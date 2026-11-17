"""CPU reference, independent feasibility checker and population search driver."""
from dataclasses import dataclass
import math
import time
import numpy as np
from .moves import generate_move, score_move


@dataclass
class Problem:
  load: object
  x: object
  capacity: object
  memory: object
  ram: object
  alpha: object
  gamma: object
  edges: object
  beta: object

  def __post_init__(self):
    for key in ('load','x','capacity','memory','ram','alpha','gamma','beta'):
      a = np.asarray(getattr(self,key),dtype=np.float64)
      if not np.all(np.isfinite(a)) or np.any(a < 0):
        raise ValueError(f'{key} must be finite and nonnegative')
      if key in ('load','x','memory','ram'):
        if np.any(a != np.floor(a)) or np.any(a > 2**40):
          raise ValueError(f'{key} must contain bounded integers')
        a = a.astype(np.int64)
      setattr(self,key,np.ascontiguousarray(a))
    if self.load.ndim != 2 or 0 in self.load.shape:
      raise ValueError('load must be a nonempty node/function matrix')
    n,f = self.load.shape
    if sum(map(int,self.load.flat))>2**52:
      raise ValueError('aggregate load exceeds exact integer numerical range')
    for key in ('x','capacity','alpha','gamma'):
      if getattr(self,key).shape != (n,f):
        raise ValueError(f'{key} shape mismatch')
    if self.memory.shape != (f,) or self.ram.shape != (n,):
      raise ValueError('memory/ram shape mismatch')
    if np.any(self.memory <= 0) or np.any(self.capacity <= 0) or np.any(self.x > self.load):
      raise ValueError('positive capacities/memory and x <= load required')
    edges = np.asarray(self.edges)
    if edges.size == 0:
      edges = np.empty((0,3),dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 3 or np.any(edges != np.floor(edges)):
      raise ValueError('edges must be integer triples')
    edges = edges.astype(np.int64)
    if self.beta.shape != (len(edges),):
      raise ValueError('one beta per edge required')
    if len(edges) and (np.any(edges < 0) or np.any(edges[:,:2] >= n) or
                      np.any(edges[:,2] >= f) or np.any(edges[:,0] == edges[:,1])):
      raise ValueError('invalid/self edge')
    if len(set(map(tuple,edges))) != len(edges):
      raise ValueError('duplicate edges')
    order = sorted(range(len(edges)),key=lambda k:(edges[k,0],edges[k,2],edges[k,1]))
    self.edges = edges[order]
    self.beta = self.beta[order]
    self.src,self.dst,self.fun = [np.ascontiguousarray(self.edges[:,k]) for k in range(3)]
    counts=np.bincount(self.src*f+self.fun,minlength=n*f)
    self.starts=np.concatenate(([0],np.cumsum(counts))).astype(np.int64)
    denom=np.maximum(self.load,1)
    self.reward=np.ascontiguousarray((self.beta+self.gamma[self.src,self.fun])/denom[self.src,self.fun])
    self.base_welfare=float(np.sum((self.alpha*self.x-self.gamma*(self.load-self.x))/denom))
    if not np.isfinite(self.base_welfare) or not np.all(np.isfinite(self.reward)):
      raise ValueError('welfare outside finite numerical range')
    self.check(np.zeros(len(edges),dtype=np.int64))

  def check(self, y):
    """Independent full reconstruction; no cached move state is trusted."""
    y=np.asarray(y)
    if y.shape != (len(self.src),) or not np.all(np.isfinite(y)) or np.any(y<0) or np.any(y!=np.floor(y)):
      raise ValueError('invalid flows')
    if np.any(y > self.load[self.src,self.fun]-self.x[self.src,self.fun]):
      raise ValueError('flow exceeds source residual load')
    incoming=np.zeros_like(self.load);outgoing=np.zeros_like(self.load)
    np.add.at(incoming,(self.dst,self.fun),y.astype(np.int64))
    np.add.at(outgoing,(self.src,self.fun),y.astype(np.int64))
    z=self.load-self.x-outgoing
    if np.any(z<0) or np.any((incoming>0)&(outgoing>0)):
      raise ValueError('conservation or sender/receiver role violated')
    processing=self.x+incoming
    with np.errstate(over='ignore'):
      required=np.maximum(processing>0,np.ceil(processing/self.capacity-1e-9))
    if not np.all(np.isfinite(required)) or np.any(required>2**40):
      raise ValueError('replica requirement outside supported range')
    replicas=required.astype(np.int64)
    # Float products cannot wrap; any feasible integer sum is <= RAM <= 2**40.
    if np.any(replicas.astype(np.float64)@self.memory>self.ram):
      raise ValueError('memory capacity violated')
    welfare=float(np.sum((self.alpha*self.x-self.gamma*z)/np.maximum(self.load,1))+
                  np.sum(self.beta*y/np.maximum(self.load[self.src,self.fun],1)))
    return dict(welfare=welfare,z=z,replicas=replicas,incoming=incoming)

  def arrays(self):
    return (self.load,self.x,self.capacity,self.memory,self.ram,self.src,self.dst,self.fun,self.reward)


@dataclass
class State:
  y: np.ndarray
  incoming: np.ndarray
  rejected: np.ndarray

  @classmethod
  def empty(cls,p,size):
    return cls(np.zeros((size,len(p.src)),dtype=np.int64),
               np.zeros((size,*p.load.shape),dtype=np.int64),
               np.repeat((p.load-p.x)[None],size,axis=0))


def evaluate(p,state,index,move):
  moves=np.asarray(move,dtype=np.int64).reshape(1,1,4,2)
  if np.any(moves[:,:,:,0] >= len(p.src)) or np.any(moves[:,:,:,0] < -1):
    raise ValueError('invalid move edge')
  return score_move(0,0,moves,state.y[index:index+1],state.incoming[index:index+1],
                    state.rejected[index:index+1],*p.arrays())


def apply_move(p,state,index,move):
  if not math.isfinite(evaluate(p,state,index,move)):
    raise ValueError('infeasible move')
  for e,q in move:
    if e>=0:
      state.y[index,e]+=q
      state.incoming[index,p.dst[e],p.fun[e]]+=q
      state.rejected[index,p.src[e],p.fun[e]]-=q


@dataclass
class Options:
  pool_size: int = 32
  candidates: int = 128
  iterations: int = 200
  seed: int = 4850
  time_limit: float = 300.0
  backend: str = 'cpu'
  shake_after: int = 3
  max_device_bytes: int = 512*1024*1024

  def validate(self):
    for k in ('pool_size','candidates','shake_after','max_device_bytes'):
      if type(getattr(self,k)) is not int or getattr(self,k)<1:
        raise ValueError(f'{k} must be a positive integer')
    if type(self.iterations) is not int or self.iterations<0:
      raise ValueError('iterations must be a nonnegative integer')
    if type(self.seed) is not int or not 0<=self.seed<2147483646:
      raise ValueError('seed must be in [0, 2147483646)')
    if not math.isfinite(self.time_limit) or self.time_limit<=0:
      raise ValueError('time_limit must be positive and finite')
    if self.backend not in ('cpu','cuda','metal'):
      raise ValueError('backend must be cpu, cuda or metal')


def select_candidate(moves,gains,level):
  gains=np.asarray(gains,dtype=np.float64)
  valid=np.isfinite(gains)&np.any(moves[:,:,1]!=0,axis=1)
  if not np.any(valid):
    return -1
  # argmax keeps the first index, including exact ties and shaking order.
  return int(np.argmax(valid if level==3 else np.where(valid,gains,-np.inf)))


class CPU:
  def __init__(self,p,o):
    self.p,self.o=p,o
    self.state=State.empty(p,o.pool_size)
    self.moves=np.full((o.pool_size,o.candidates,4,2),-1,dtype=np.int64)
    self.level=np.zeros(o.pool_size,dtype=np.int64)
    self.stale=np.zeros(o.pool_size,dtype=np.int64)
    self.value=np.full(o.pool_size,p.base_welfare)
    self.best=self.value.copy();self.best_y=self.state.y.copy()
    self.evaluated=0

  def step(self,iteration):
    p,o,s=self.p,self.o,self.state
    for index in range(o.pool_size):
      level=3 if self.stale[index]>=o.shake_after*3 else self.level[index]
      gains=[]
      for c in range(o.candidates):
        generate_move(o.seed,iteration,index,c,level,self.moves,s.y,s.incoming,s.rejected,
                      p.x,p.capacity,p.memory,p.ram,p.src,p.dst,p.fun,p.starts)
        gains.append(score_move(index,c,self.moves,s.y,s.incoming,s.rejected,*p.arrays()))
      self.accept(index,level,gains)
    self.evaluated+=o.pool_size*o.candidates
    return self.best.copy()

  def accept(self,index,level,gains):
    # Shared float64 ordering and archive updates for CPU and hybrid Metal.
    chosen=select_candidate(self.moves[index],gains,level)
    gain=gains[chosen] if chosen>=0 else -math.inf
    if chosen>=0 and (gain>1e-12 or level==3):
      apply_move(self.p,self.state,index,self.moves[index,chosen])
      self.value[index]+=gain
      if self.value[index]>self.best[index]+1e-12:
        self.best[index]=self.value[index];self.best_y[index]=self.state.y[index]
      self.level[index]=0;self.stale[index]=0
    else:
      self.level[index]=(self.level[index]+1)%3;self.stale[index]+=1
      if level==3:
        self.stale[index]=0

  def result(self):
    index=int(np.argmax(self.best))
    return self.best_y[index].copy()


def solve(p,options=None):
  o=options or Options();o.validate()
  start=time.perf_counter()
  if o.backend=='cuda':
    from .gpu import CUDA
    engine=CUDA(p,o)
  elif o.backend=='metal':
    from .metal import Metal
    engine=Metal(p,o)
  else:
    engine=CPU(p,o)
  trace=[dict(iteration=0,elapsed_s=time.perf_counter()-start,best_welfare=p.base_welfare)]
  for iteration in range(o.iterations):
    if time.perf_counter()-start>=o.time_limit:
      break
    best=engine.step(iteration)
    trace.append(dict(iteration=iteration+1,elapsed_s=time.perf_counter()-start,
                      best_welfare=float(np.max(best))))
  y=engine.result();checked=p.check(y)
  return dict(y=y,**checked,trace=trace,elapsed_s=time.perf_counter()-start,
              candidates_evaluated=engine.evaluated,backend=o.backend,
              initial_welfare=p.base_welfare)
