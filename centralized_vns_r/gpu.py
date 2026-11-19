"""Resident CUDA VNS-R candidates with the baseline parallel acceptance."""
from dataclasses import replace
from functools import lru_cache
from centralized_vns.gpu import CUDA as ReferenceCUDA, kernels as reference_kernels
from centralized_vns.moves import score_move
from .moves import generate_move,incoming_index


@lru_cache(maxsize=1)
def kernels():
  global cuda
  cuda,_,accept=reference_kernels()
  generate=cuda.jit(device=True)(generate_move)
  score=cuda.jit(device=True)(score_move)

  @cuda.jit
  def candidates(seed,iteration,shake_after,levels,stale,moves,gains,y,inc,z,
                 load,x,capacity,memory,ram,src,dst,fun,reward,starts,in_edges,in_starts):
    index=cuda.grid(1)
    p,c=index//gains.shape[1],index%gains.shape[1]
    if p>=gains.shape[0]:
      return
    level=3 if stale[p]>=shake_after*3 else levels[p]
    generate(seed,iteration,p,c,level,moves,y,inc,z,x,capacity,memory,ram,
             src,dst,fun,starts,in_edges,in_starts)
    gains[p,c]=score(p,c,moves,y,inc,z,load,x,capacity,memory,ram,src,dst,fun,reward)
  return cuda,candidates,accept


class CUDA(ReferenceCUDA):
  def __init__(self,p,o):
    extra=8*(len(p.src)+len(p.load)+1)
    super().__init__(p,replace(o,max_device_bytes=o.max_device_bytes-extra))
    self.o=o;self.device_bytes+=extra
    self.in_edges,self.in_starts=[self.cuda.to_device(a) for a in incoming_index(p)]
    self.cuda,self.candidate_kernel,self.accept_kernel=kernels()

  def step(self,iteration):
    o=self.o
    self.candidate_kernel[(o.pool_size*o.candidates+63)//64,64](
      o.seed,iteration,o.shake_after,self.level,self.stale,self.moves,self.gains,
      self.y,self.inc,self.z,*self.data,self.starts,self.in_edges,self.in_starts)
    self.accept_kernel[o.pool_size,64](
      o.shake_after,self.level,self.stale,self.moves,self.gains,
      self.y,self.inc,self.z,self.value,self.best,self.best_y,*self.data[5:8])
    self.evaluated+=o.pool_size*o.candidates
    return self.best.copy_to_host()
