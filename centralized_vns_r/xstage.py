"""CPU local-plan refinement, guarded by original local-gain floors."""
from dataclasses import replace
import math
import time

import numpy as np


def local_gains(p, x):
  """Return each node's alpha-weighted local service, excluding rejection/flows."""
  x=np.asarray(x,dtype=np.float64)
  if (x.shape!=p.load.shape or not np.all(np.isfinite(x)) or
      np.any(x<0) or np.any(x>p.load) or np.any(x!=np.floor(x))):
    raise ValueError('invalid local plan')
  with np.errstate(over='ignore',invalid='ignore'):
    gains=np.sum(p.alpha*x/np.maximum(p.load,1),axis=1,dtype=np.float64)
  if not np.all(np.isfinite(gains)):
    raise ValueError('local gains must be finite')
  return gains


def _room(p, x, incoming, node, fun):
  processing=x[node]+incoming[node]
  replicas=np.maximum(processing>0,np.ceil(processing/p.capacity[node]-1e-9))
  free=max(0,int(p.ram[node]-replicas@p.memory))
  count=int(replicas[fun])+free//int(p.memory[fun])
  if not count:
    return 0
  # Match the independent checker's replica tolerance, then cap before int conversion.
  with np.errstate(over='ignore'):
    limit=p.capacity[node,fun]*(count+1e-9)
  limit=min(limit,float(p.load.sum()))
  return max(0,int(math.floor(limit))-int(processing[fun]))


def _proposals(p, x, y, checked, floor, rng, available):
  incoming=checked['incoming']
  gains=local_gains(p,x)
  for node in rng.permutation(p.load.shape[0]):
    for fun in rng.permutation(p.load.shape[1]):
      if not available():
        return
      maximum=min(int(p.load[node,fun]-x[node,fun]),_room(p,x,incoming,node,fun))
      for amount in sorted({1,maximum}):
        if 0<amount<=maximum:
          trial=x.copy();trial[node,fun]+=amount
          yield trial,_trim_outgoing(p,trial,y,node,fun),node
      if x[node,fun]==0:
        continue
      count=int(checked['replicas'][node,fun])
      boundary=0 if count<=1 else math.floor(p.capacity[node,fun]*(count-1+1e-9))
      removal=int(x[node,fun]+incoming[node,fun])-boundary
      for amount in sorted({1,removal,int(x[node,fun])}):
        if not available():
          return
        if not 0<amount<=x[node,fun]:
          continue
        dropped=x.copy();dropped[node,fun]-=amount
        yield dropped,y.copy(),node
        for other in rng.permutation(p.load.shape[1]):
          if not available():
            return
          if other==fun:
            continue
          maximum=min(int(p.load[node,other]-x[node,other]),
                      _room(p,dropped,incoming,node,other))
          unit=p.alpha[node,other]/max(p.load[node,other],1)
          lost=amount*p.alpha[node,fun]/max(p.load[node,fun],1)
          deficit=max(0.,float(floor[node]-gains[node]+lost-1e-12))
          compensation=0
          if unit>0 and deficit<=unit*maximum:
            compensation=int(math.ceil(deficit/unit))
          for added in sorted({1,maximum,compensation}):
            if 0<added<=maximum:
              trial=dropped.copy();trial[node,other]+=added
              yield trial,_trim_outgoing(p,trial,y,node,other),node


def _trim_outgoing(p, x, y, node, fun):
  trial=y.copy()
  edges=range(int(p.starts[node*p.load.shape[1]+fun]),
              int(p.starts[node*p.load.shape[1]+fun+1]))
  excess=int(sum(int(y[e]) for e in edges)-(p.load[node,fun]-x[node,fun]))
  for edge in sorted(edges,key=lambda e:p.reward[e]):
    if excess<=0:
      break
    removed=min(excess,int(trial[edge]));trial[edge]-=removed;excess-=removed
  return trial


def refine_x(p, y, *, iterations=1000, time_limit=15., seed=4850):
  """Improve a checked incumbent without lowering any initial local-gain floor.

  ``iterations`` bounds all attempted transactions, including insertion repairs.
  The deadline is checked between proposals/checks; a single full check may overrun.
  """
  if type(iterations) is not int or iterations<0:
    raise ValueError('iterations must be a nonnegative integer')
  if type(seed) is not int or not 0<=seed<2147483646:
    raise ValueError('seed must be in [0, 2147483646)')
  if not math.isfinite(time_limit) or time_limit<0:
    raise ValueError('time_limit must be nonnegative and finite')
  start=time.perf_counter();deadline=start+time_limit
  # Revalidate mutable Problem arrays and recompute all derived welfare data.
  p=replace(p)
  checked=p.check(y)
  if not math.isfinite(checked['welfare']):
    raise ValueError('incumbent welfare must be finite')
  x=p.x.copy();y=np.asarray(y,dtype=np.int64).copy()
  floor=local_gains(p,x).copy();initial=checked['welfare']
  attempted=accepted=0;rng=np.random.default_rng(seed)
  trace=[dict(iteration=0,elapsed_s=time.perf_counter()-start,best_welfare=initial)]

  def available():
    return attempted<iterations and time.perf_counter()<deadline

  while available():
    improved=False
    for trial_x,trial_y,node in _proposals(p,x,y,checked,floor,rng,available):
      if not available():
        break
      attempted+=1
      try:
        gains=local_gains(p,trial_x)
        if np.any(gains<floor-1e-12):
          continue
        trial_p=replace(p,x=trial_x)
        trial=trial_p.check(trial_y)
      except ValueError:
        continue
      # A local swap can pay for itself by freeing a replica for incoming work.
      edges=np.flatnonzero(p.dst==node)
      for edge in sorted(edges,key=lambda e:-p.reward[e]):
        if not available():
          break
        src,fun=int(p.src[edge]),int(p.fun[edge])
        if p.reward[edge]<=0 or trial['incoming'][src,fun]>0:
          continue
        outgoing=p.load[node,fun]-trial_x[node,fun]-trial['z'][node,fun]
        if outgoing>0:
          continue
        amount=min(int(trial['z'][src,fun]),
                   _room(p,trial_x,trial['incoming'],node,fun))
        if amount<=0:
          continue
        repaired=trial_y.copy();repaired[edge]+=amount;attempted+=1
        try:
          repaired_check=trial_p.check(repaired)
        except ValueError:
          continue
        trial_y,trial=repaired,repaired_check
      if math.isfinite(trial['welfare']) and trial['welfare']>checked['welfare']+1e-12:
        x,y,checked=trial_x,trial_y,trial;accepted+=1;improved=True
        trace.append(dict(iteration=attempted,elapsed_s=time.perf_counter()-start,
                          best_welfare=checked['welfare']))
        break
    if not improved:
      break
  return dict(algorithm='VNS-RX',backend='cpu',x=x,y=y,**checked,
              local_gain_floor=floor,local_gain=local_gains(p,x),
              attempted=attempted,accepted=accepted,trace=trace,
              initial_welfare=initial,elapsed_s=time.perf_counter()-start)
