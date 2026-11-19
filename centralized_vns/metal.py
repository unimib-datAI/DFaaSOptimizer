"""Integer Metal candidates/feasibility with reference float64 CPU decisions."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from .core import CPU


@lru_cache(maxsize=1)
def kernel():
  import mlx.core as mx
  if not mx.metal.is_available():
    raise RuntimeError('Apple Metal GPU unavailable')
  return mx.fast.metal_kernel(name='vns_candidates',
    input_names=['load','x','ram','src','dst','fun','starts','costs',
                 'y','inc','z','space','levels','params'],
    output_names=['moves','valid'], source=Path(__file__).with_suffix('.metal').read_text())


class Metal(CPU):
  def __init__(self,p,o):
    n,f=p.load.shape; b,c=o.pool_size,o.candidates; e=len(p.src)
    self.stride=int(p.load.sum(axis=0).max())+1
    # shortcut: dense processing tables, use sparse thresholds if real inputs exceed this cap.
    self.device_bytes=8*(n*f*self.stride+2*n*f+n+3*e+n*f+1+
                         b*e+3*b*n*f+b+b*c*8+6)+b*c
    if self.device_bytes>o.max_device_bytes:
      raise ValueError(f'Metal memory budget exceeded: {self.device_bytes} bytes')
    import mlx.core as mx
    self.mx=mx; self.kernel=kernel()
    super().__init__(p,o)
    costs=np.empty((n,f,self.stride),dtype=np.int64)
    processing=np.arange(self.stride,dtype=np.float64)
    for i in range(n):
      for g in range(f):
        with np.errstate(over='ignore',invalid='ignore'):
          ratio=processing/p.capacity[i,g]
          replicas=np.maximum(processing>0,np.ceil(ratio-1e-9))
          used=replicas*p.memory[g]
        used=np.where((ratio>2**40)|~np.isfinite(used),p.ram[i]+1,used)
        costs[i,g]=np.minimum(used,p.ram[i]+1).astype(np.int64)
    self.data=[mx.array(a) for a in (p.load,p.x,p.ram,p.src,p.dst,p.fun,p.starts,costs)]
    mx.eval(*self.data)

  def step(self,iteration):
    p,o,s,mx=self.p,self.o,self.state,self.mx
    # The generator's replica arithmetic stays on CPU in exactly float64.
    processing=p.x+s.incoming
    replicas=np.maximum(processing>0,np.ceil(processing/p.capacity-1e-9))
    used=replicas*p.memory
    other=used.sum(axis=2,keepdims=True)-used
    available=np.floor((p.ram[None,:,None]-other)/p.memory)
    with np.errstate(over='ignore'):
      limit=np.where(available>0,(available+1e-9)*p.capacity,0.)
    limit=np.minimum(limit,self.stride-1)
    space=np.maximum(0,np.floor(limit)-processing).astype(np.int64)
    levels=np.where(self.stale>=o.shake_after*3,3,self.level)
    params=np.array([o.seed,iteration,o.candidates,*p.load.shape,self.stride],dtype=np.int64)
    outputs=self.kernel(inputs=self.data+[mx.array(a) for a in
      (s.y,s.incoming,s.rejected,space,levels,params)],
      grid=(o.pool_size*o.candidates,1,1),
      threadgroup=(min(64,o.pool_size*o.candidates),1,1),
      output_shapes=[self.moves.shape,(o.pool_size,o.candidates)],
      output_dtypes=[mx.int64,mx.bool_],stream=mx.gpu)
    mx.eval(*outputs)
    self.moves=np.array(outputs[0]); valid=np.array(outputs[1])
    gains=np.zeros((o.pool_size,o.candidates),dtype=np.float64)
    if len(p.src):
      for a in range(4):
        edges=self.moves[:,:,a,0]; q=self.moves[:,:,a,1]
        gains+=np.where(edges>=0,q*p.reward[np.maximum(edges,0)],0.)
    gains[~valid]=-np.inf
    for index in range(o.pool_size):
      self.accept(index,levels[index],gains[index])
    self.evaluated+=o.pool_size*o.candidates
    return self.best.copy()
