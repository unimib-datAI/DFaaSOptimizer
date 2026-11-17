"""Fused CUDA search and device plateau; host polls bounded batches only."""
from functools import lru_cache
import math
import time
import numpy as np
from centralized_vns.core import Options
from centralized_vns.moves import score_move
from centralized_vns_r.core import validate_plateau
from centralized_vns_r.moves import generate_move, incoming_index


def estimate_device_bytes(p,o,patience,batch_size):
  b,c,n,f,e=o.pool_size,o.candidates,*p.load.shape,len(p.src)
  # Include upload staging and rolling history; no host-sized population is built.
  return sum(a.nbytes for a in (*p.arrays(),p.starts))+8*(
    e+n+1 + e+2*n*f + 2*b*e+2*b*n*f+4*b+8*b*c+
    (patience+1)*b+3+batch_size)


def validate_execution(o,patience,tolerance,scope,batch_size,threads):
  o.validate()
  if o.backend!='cuda':
    raise ValueError('Spark execution requires backend cuda')
  validate_plateau(patience,tolerance,o.shake_after)
  if scope not in ('all','global'):
    raise ValueError('plateau_scope must be all or global')
  if type(batch_size) is not int or batch_size<1:
    raise ValueError('batch_size must be a positive integer')
  if type(threads) is not int or threads not in (64,128,256):
    raise ValueError('threads must be 64, 128 or 256')


@lru_cache(maxsize=3)
def kernels(threads):
  global cuda
  from numba import cuda as cuda_module
  from numba import config
  cuda=cuda_module
  generate=cuda.jit(device=True,fastmath=False,opt=True)(generate_move)
  score=cuda.jit(device=True,fastmath=False,opt=True)(score_move)
  if config.ENABLE_CUDASIM:
    def multiply(a,b):
      return float(a)*float(b)
  else:
    from numba.cuda.libdevice import dmul_rn as multiply

  @cuda.jit(fastmath=False)
  def initialize(y,inc,z,best_y,value,best,level,stale,history,initial_y,initial_inc,initial_z,welfare):
    p=cuda.blockIdx.x;lane=cuda.threadIdx.x
    for e in range(lane,y.shape[1],cuda.blockDim.x):
      y[p,e]=initial_y[e];best_y[p,e]=initial_y[e]
    for index in range(lane,inc.shape[1]*inc.shape[2],cuda.blockDim.x):
      n,f=index//inc.shape[2],index%inc.shape[2]
      inc[p,n,f]=initial_inc[n,f];z[p,n,f]=initial_z[n,f]
    if lane==0:
      value[p]=welfare;best[p]=welfare;history[0,p]=welfare
      level[p]=0;stale[p]=0

  @cuda.jit(fastmath=False)
  def search(seed,iteration,shake_after,status,levels,stale,moves,y,inc,z,value,best,best_y,
             load,x,capacity,memory,ram,src,dst,fun,reward,starts,in_edges,in_starts):
    # Uniform block guard: kernels queued after the first plateau do no search work.
    if status[0]:
      return
    p=cuda.blockIdx.x;lane=cuda.threadIdx.x
    choices=cuda.shared.array(threads,np.int64)
    scores=cuda.shared.array(threads,np.float64)
    improved=cuda.shared.array(1,np.int64)
    level=3 if stale[p]>=shake_after*3 else levels[p]
    chosen=-1;gain=-math.inf
    for c in range(lane,moves.shape[1],cuda.blockDim.x):
      generate(seed,iteration,p,c,level,moves,y,inc,z,x,capacity,memory,ram,
               src,dst,fun,starts,in_edges,in_starts)
      g=score(p,c,moves,y,inc,z,load,x,capacity,memory,ram,src,dst,fun,reward)
      if math.isfinite(g):
        # Explicit rounded product prevents default NVVM FMA contraction in gains.
        g=0.
        for a in range(4):
          e=moves[p,c,a,0]
          if e>=0:
            g+=multiply(float(moves[p,c,a,1]),reward[e])
      nonempty=False
      for a in range(4):
        if moves[p,c,a,1]!=0:
          nonempty=True
      if math.isfinite(g) and nonempty and (chosen<0 or (level!=3 and g>gain)):
        chosen=c;gain=g
    choices[lane]=chosen;scores[lane]=gain
    cuda.syncthreads()
    width=cuda.blockDim.x//2
    while width>0:
      if lane<width:
        other=choices[lane+width];g=scores[lane+width]
        chosen=choices[lane];gain=scores[lane]
        if other>=0 and (chosen<0 or (level==3 and other<chosen) or
            (level!=3 and (g>gain or (g==gain and other<chosen)))):
          choices[lane]=other;scores[lane]=g
      cuda.syncthreads()
      width//=2
    if lane==0:
      improved[0]=0
      chosen=choices[0];gain=scores[0]
      if chosen>=0 and (gain>1e-12 or level==3):
        # Ordered updates preserve repeated-edge semantics in compound moves.
        for a in range(4):
          e,q=moves[p,chosen,a,0],moves[p,chosen,a,1]
          if e>=0:
            y[p,e]+=q;inc[p,dst[e],fun[e]]+=q;z[p,src[e],fun[e]]-=q
        value[p]+=gain
        if value[p]>best[p]+1e-12:
          best[p]=value[p];improved[0]=1
        levels[p]=0;stale[p]=0
      else:
        levels[p]=(levels[p]+1)%3;stale[p]+=1
        if level==3:
          stale[p]=0
    cuda.syncthreads()
    if improved[0]:
      for e in range(lane,y.shape[1],cuda.blockDim.x):
        best_y[p,e]=y[p,e]

  @cuda.jit(fastmath=False)
  def monitor(iteration,iterations,patience,tolerance,global_scope,status,best,history,trace,slot):
    if status[0]:
      return
    lane=cuda.threadIdx.x
    maxima=cuda.shared.array(threads,np.float64)
    references=cuda.shared.array(threads,np.float64)
    indices=cuda.shared.array(threads,np.int64)
    active=cuda.shared.array(threads,np.int64)
    current=-math.inf;reference=-math.inf;winner=-1;changed=0
    ready=patience>0 and iteration>=patience
    for p in range(lane,best.size,cuda.blockDim.x):
      v=best[p]
      if winner<0 or v>current:
        current=v;winner=p
      if ready:
        old=history[(iteration-patience)%(patience+1),p]
        reference=max(reference,old)
        if (v-old)/max(1.,abs(old))>tolerance:
          changed=1
      history[iteration%(patience+1),p]=v
    maxima[lane]=current;references[lane]=reference
    indices[lane]=winner;active[lane]=changed
    cuda.syncthreads()
    width=cuda.blockDim.x//2
    while width>0:
      if lane<width:
        v=maxima[lane+width];other=indices[lane+width]
        if other>=0 and (indices[lane]<0 or v>maxima[lane] or
            (v==maxima[lane] and other<indices[lane])):
          maxima[lane]=v;indices[lane]=other
        references[lane]=max(references[lane],references[lane+width])
        active[lane]=max(active[lane],active[lane+width])
      cuda.syncthreads()
      width//=2
    if lane==0:
      status[1]=iteration;status[2]=indices[0];trace[slot]=maxima[0]
      plateau=active[0]==0
      if global_scope and ready:
        plateau=(maxima[0]-references[0])/max(1.,abs(references[0]))<=tolerance
      if ready and plateau and iteration<iterations:
        status[0]=1
  return cuda,initialize,search,monitor


class CUDA:
  def __init__(self,p,o,initial_y,*,patience=200,batch_size=16,threads=128):
    validate_execution(o,patience,0.,'all',batch_size,threads)
    self.device_bytes=estimate_device_bytes(p,o,patience,batch_size)
    if self.device_bytes>o.max_device_bytes:
      raise ValueError(f'CUDA memory budget exceeded: {self.device_bytes} bytes')
    checked=p.check(initial_y)
    if not math.isfinite(checked['welfare']):
      raise ValueError('initial welfare must be finite')
    self.o,self.patience,self.batch_size,self.threads=o,patience,batch_size,threads
    self.cuda,init,self.search,self.monitor=kernels(threads)
    if not self.cuda.is_available():
      raise RuntimeError('CUDA unavailable; Spark runner does not fall back to CPU')
    b,c,n,f,e=o.pool_size,o.candidates,*p.load.shape,len(p.src)
    transfer=self.cuda.to_device;allocate=self.cuda.device_array
    self.data=tuple(transfer(a) for a in p.arrays());self.starts=transfer(p.starts)
    self.in_edges,self.in_starts=[transfer(a) for a in incoming_index(p)]
    self.y=allocate((b,e),np.int64);self.best_y=allocate((b,e),np.int64)
    self.inc=allocate((b,n,f),np.int64);self.z=allocate((b,n,f),np.int64)
    self.level=allocate(b,np.int64);self.stale=allocate(b,np.int64)
    self.value=allocate(b,np.float64);self.best=allocate(b,np.float64)
    self.moves=allocate((b,c,4,2),np.int64)
    self.history=allocate((patience+1,b),np.float64)
    self.status=transfer(np.zeros(3,dtype=np.int64))
    self.trace=allocate(batch_size,np.float64)
    self.host_status=np.zeros(3,dtype=np.int64)
    init[b,threads](self.y,self.inc,self.z,self.best_y,self.value,self.best,self.level,self.stale,
      self.history,transfer(np.asarray(initial_y,dtype=np.int64)),transfer(checked['incoming']),
      transfer(checked['z']),checked['welfare'])

  def advance(self,iteration,count,tolerance,scope):
    if not 1<=count<=self.batch_size:
      raise ValueError('batch count outside allocation')
    o=self.o
    for slot in range(count):
      step=iteration+slot
      self.search[o.pool_size,self.threads](o.seed,step,o.shake_after,self.status,self.level,
        self.stale,self.moves,self.y,self.inc,self.z,self.value,self.best,self.best_y,
        *self.data,self.starts,self.in_edges,self.in_starts)
      self.monitor[1,self.threads](step+1,o.iterations,self.patience,tolerance,scope=='global',
        self.status,self.best,self.history,self.trace,slot)
    self.status.copy_to_host(self.host_status)
    completed=int(self.host_status[1])-iteration
    if completed<1:
      raise RuntimeError('cannot advance a stopped CUDA search')
    return float(self.trace[completed-1:completed].copy_to_host()[0])

  def result(self):
    index=int(self.host_status[2])
    return self.best_y[index:index+1].copy_to_host()[0]


def solve(p,options=None,initial_y=None,*,patience=200,plateau_tolerance=1e-6,
          plateau_scope='all',batch_size=16,threads=128,time_budget='total'):
  o=options or Options(backend='cuda')
  validate_execution(o,patience,plateau_tolerance,plateau_scope,batch_size,threads)
  if time_budget not in ('total','search'):
    raise ValueError('time_budget must be total or search')
  start=time.perf_counter()
  y=np.zeros(len(p.src),dtype=np.int64) if initial_y is None else np.asarray(initial_y).copy()
  initial=p.check(y)
  if not math.isfinite(initial['welfare']):
    raise ValueError('initial welfare must be finite')
  engine=CUDA(p,o,y,patience=patience,batch_size=batch_size,threads=threads)
  engine.cuda.synchronize()
  search_start=time.perf_counter()
  budget_start=start if time_budget=='total' else search_start
  trace=[dict(iteration=0,elapsed_s=time.perf_counter()-start,best_welfare=initial['welfare'])]
  completed=0;reason='iteration_limit';batch_seconds=0.;previous_count=1
  while completed<o.iterations:
    now=time.perf_counter();remaining=o.time_limit-(now-budget_start)
    if remaining<=0:
      reason='time_limit';break
    count=min(batch_size if batch_seconds else 1,o.iterations-completed)
    if batch_seconds:
      count=min(count,max(1,int(remaining/(batch_seconds/previous_count))))
    value=engine.advance(completed,count,plateau_tolerance,plateau_scope)
    previous_count=int(engine.host_status[1])-completed
    completed=int(engine.host_status[1]);elapsed=time.perf_counter()-start
    batch_seconds=time.perf_counter()-now
    trace.append(dict(iteration=completed,elapsed_s=elapsed,best_welfare=value))
    if time.perf_counter()-budget_start>=o.time_limit:
      reason='time_limit';break
    if engine.host_status[0]:
      reason='plateau';break
  search_elapsed=time.perf_counter()-search_start
  y=engine.result();checked=p.check(y)
  if not math.isfinite(checked['welfare']):
    raise RuntimeError('CUDA result has nonfinite independently checked welfare')
  return dict(algorithm='VNS-R',backend='cuda',x=p.x.copy(),y=y,**checked,
    initial_welfare=initial['welfare'],trace=trace,elapsed_s=time.perf_counter()-start,
    iterations_completed=completed,candidates_evaluated=completed*o.pool_size*o.candidates,
    termination_reason=reason,plateau=dict(patience=patience,tolerance=plateau_tolerance,scope=plateau_scope),
    execution=dict(variant='fused_cuda',threads=threads,batch_size=batch_size,
      device_bytes=engine.device_bytes,host_checkpoints=len(trace)-1,
      setup_s=search_start-start,search_elapsed_s=search_elapsed,time_budget=time_budget,
      trace='real host checkpoints; plateau evaluated on GPU every iteration',
      deadline='checked between batches; an in-flight batch may overrun'))
