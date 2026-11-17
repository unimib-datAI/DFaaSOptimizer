"""GPU integer neighborhoods using exact float64 replica boundaries from the host."""
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
import numpy as np
from centralized_vns.metal import Metal as ReferenceMetal
from .moves import incoming_index


@lru_cache(maxsize=1)
def kernel():
  import mlx.core as mx
  return mx.fast.metal_kernel(name='vns_r_candidates',
    input_names=['load','x','ram','src','dst','fun','starts','costs','memory',
                 'limits','in_edges','in_starts','y','inc','z','space','fill','release','levels','params'],
    output_names=['moves','valid'],source=Path(__file__).with_suffix('.metal').read_text())


class Metal(ReferenceMetal):
  def __init__(self,p,o):
    n,f=p.load.shape
    self.replica_stride=int(p.ram.max()//p.memory.min())+1
    extra=8*(n*f*self.replica_stride+f+len(p.src)+n+1+2*o.pool_size*n*f+1)
    super().__init__(p,replace(o,max_device_bytes=o.max_device_bytes-extra))
    self.o=o;self.device_bytes+=extra;self.kernel=kernel()
    # shortcut: dense replica limits, use sparse thresholds if RAM/memory exceeds the cap.
    limits=np.empty((n,f,self.replica_stride),dtype=np.int64)
    replicas=np.arange(self.replica_stride,dtype=np.float64)
    for i in range(n):
      for g in range(f):
        with np.errstate(over='ignore'):
          limits[i,g]=np.floor(np.minimum(self.stride-1,(replicas+1e-9)*p.capacity[i,g])).astype(np.int64)
        limits[i,g,0]=0
    self.data += [self.mx.array(a) for a in (p.memory,limits,*incoming_index(p))]
    self.mx.eval(*self.data)

  def step(self,iteration):
    p,o,s,mx=self.p,self.o,self.state,self.mx
    processing=p.x+s.incoming
    replicas=np.maximum(processing>0,np.ceil(processing/p.capacity-1e-9))
    used=replicas*p.memory
    other=used.sum(axis=2,keepdims=True)-used
    available=np.floor((p.ram[None,:,None]-other)/p.memory)
    with np.errstate(over='ignore'):
      limit=np.where(available>0,(available+1e-9)*p.capacity,0.)
      boundary=(replicas+1e-9)*p.capacity
      boundary=np.where(boundary<processing+1,(replicas+1+1e-9)*p.capacity,boundary)
      lower=np.where(replicas>1,np.minimum(processing,(replicas-1+1e-9)*p.capacity),0.)
    space=np.maximum(0,np.floor(np.minimum(limit,self.stride-1))-processing).astype(np.int64)
    fill=np.maximum(0,np.floor(np.minimum(boundary,self.stride-1))-processing).astype(np.int64)
    release=np.maximum(0,processing-np.floor(lower)).astype(np.int64)
    levels=np.where(self.stale>=o.shake_after*3,3,self.level)
    params=np.array([o.seed,iteration,o.candidates,*p.load.shape,self.stride,self.replica_stride],dtype=np.int64)
    outputs=self.kernel(inputs=self.data+[mx.array(a) for a in
      (s.y,s.incoming,s.rejected,space,fill,release,levels,params)],
      grid=(o.pool_size*o.candidates,1,1),
      threadgroup=(min(64,o.pool_size*o.candidates),1,1),
      output_shapes=[self.moves.shape,(o.pool_size,o.candidates)],
      output_dtypes=[mx.int64,mx.bool_],stream=mx.gpu)
    mx.eval(*outputs)
    self.moves=np.array(outputs[0]);valid=np.array(outputs[1])
    gains=np.zeros((o.pool_size,o.candidates),dtype=np.float64)
    if len(p.src):
      for a in range(4):
        edges=self.moves[:,:,a,0];q=self.moves[:,:,a,1]
        gains+=np.where(edges>=0,q*p.reward[np.maximum(edges,0)],0.)
    gains[~valid]=-np.inf
    for index in range(o.pool_size):
      self.accept(index,levels[index],gains[index])
    self.evaluated+=o.pool_size*o.candidates
    return self.best.copy()
