"""Optional resident CUDA population. No CUDA import occurs in CPU execution."""
from functools import lru_cache
import math
import numpy as np
from .moves import generate_move, score_move


@lru_cache(maxsize=1)
def kernels():
  global cuda
  from numba import cuda as cuda_module
  # CUDA simulator resolves the CUDA module in kernel globals.
  cuda=cuda_module
  generate=cuda_module.jit(device=True)(generate_move)
  score=cuda_module.jit(device=True)(score_move)

  @cuda_module.jit
  def candidates(seed,iteration,shake_after,levels,stale,moves,gains,y,inc,z,
                 load,x,capacity,memory,ram,src,dst,fun,reward,starts):
    index=cuda.grid(1)
    p,c=index//gains.shape[1],index%gains.shape[1]
    if p>=gains.shape[0]:
      return
    level=3 if stale[p]>=shake_after*3 else levels[p]
    generate(seed,iteration,p,c,level,moves,y,inc,z,x,capacity,memory,ram,src,dst,fun,starts)
    gains[p,c]=score(p,c,moves,y,inc,z,load,x,capacity,memory,ram,src,dst,fun,reward)

  @cuda_module.jit
  def accept(shake_after,levels,stale,moves,gains,y,inc,z,value,best,best_y,src,dst,fun):
    p=cuda.blockIdx.x
    lane=cuda.threadIdx.x
    if p>=y.shape[0]:
      return
    choices=cuda.shared.array(64,np.int64)
    scores=cuda.shared.array(64,np.float64)
    improved=cuda.shared.array(1,np.int64)
    level=3 if stale[p]>=shake_after*3 else levels[p]
    chosen=-1;gain=-math.inf
    for c in range(lane,gains.shape[1],cuda.blockDim.x):
      nonempty=False
      for a in range(4):
        if moves[p,c,a,1]!=0:
          nonempty=True
      g=gains[p,c]
      if math.isfinite(g) and nonempty and (chosen<0 or g>gain):
        chosen,gain=c,g
        if level==3:
          break
    choices[lane]=chosen;scores[lane]=gain
    cuda.syncthreads()
    width=cuda.blockDim.x//2
    while width>0:
      if lane<width:
        other=choices[lane+width];g=scores[lane+width]
        chosen=choices[lane];gain=scores[lane]
        # Compare indices explicitly: lane order is not candidate order after striding.
        if other>=0 and (chosen<0 or (level==3 and other<chosen) or
            (level!=3 and (g>gain or (g==gain and other<chosen)))):
          choices[lane]=other;scores[lane]=g
      cuda.syncthreads()
      width//=2
    if lane==0:
      improved[0]=0
      chosen=choices[0];gain=scores[0]
      if chosen>=0 and (gain>1e-12 or level==3):
        # Four updates stay ordered, including repeated edges within a compound move.
        for a in range(4):
          e,q=moves[p,chosen,a,0],moves[p,chosen,a,1]
          if e>=0:
            y[p,e]+=q
            inc[p,dst[e],fun[e]]+=q
            z[p,src[e],fun[e]]-=q
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
  return cuda_module,candidates,accept


class CUDA:
  def __init__(self,p,o):
    from .core import State
    self.o=o;self.evaluated=0
    # Bound all persistent arrays before either host population or device allocation.
    b,c,n,f,e=o.pool_size,o.candidates,*p.load.shape,len(p.src)
    self.device_bytes=sum(a.nbytes for a in (*p.arrays(),p.starts))+8*(
      2*b*e+2*b*n*f+4*b+b*c*9)
    if self.device_bytes>o.max_device_bytes:
      raise ValueError(f'CUDA memory budget exceeded: {self.device_bytes} bytes')
    self.cuda,self.candidate_kernel,self.accept_kernel=kernels()
    if not self.cuda.is_available():
      raise RuntimeError('CUDA unavailable; select --backend cpu or run with NUMBA_ENABLE_CUDASIM=1')
    transfer=self.cuda.to_device
    self.data=tuple(transfer(a) for a in p.arrays());self.starts=transfer(p.starts)
    state=State.empty(p,b)
    self.y,self.inc,self.z=[transfer(a) for a in (state.y,state.incoming,state.rejected)]
    self.best_y=transfer(state.y)
    self.level=transfer(np.zeros(b,dtype=np.int64));self.stale=transfer(np.zeros(b,dtype=np.int64))
    self.value=transfer(np.full(b,p.base_welfare));self.best=transfer(np.full(b,p.base_welfare))
    self.moves=self.cuda.device_array((b,c,4,2),dtype=np.int64)
    self.gains=self.cuda.device_array((b,c),dtype=np.float64)

  def step(self,iteration):
    o=self.o
    self.candidate_kernel[(o.pool_size*o.candidates+63)//64,64](
      o.seed,iteration,o.shake_after,self.level,self.stale,self.moves,self.gains,
      self.y,self.inc,self.z,*self.data,self.starts)
    self.accept_kernel[o.pool_size,64](
      o.shake_after,self.level,self.stale,self.moves,self.gains,
      self.y,self.inc,self.z,self.value,self.best,self.best_y,*self.data[5:8])
    self.evaluated+=o.pool_size*o.candidates
    return self.best.copy_to_host()  # Only pool_size scalars per iteration.

  def result(self):
    index=int(np.argmax(self.best.copy_to_host()))
    return self.best_y[index:index+1].copy_to_host()[0]
