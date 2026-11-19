"""Replica-boundary and relocation/insertion neighborhoods for VNS-R."""
import math
import numpy as np


def incoming_index(p):
  return (np.argsort(p.dst,kind="stable").astype(np.int64),
          np.concatenate(([0],np.cumsum(np.bincount(p.dst,minlength=len(p.load))))).astype(np.int64))


def generate_move(seed, iteration, p, c, level, moves, y, incoming, rejected,
                  x, capacity, memory, ram, src, dst, fun, starts, in_edges, in_starts):
  for a in range(4):
    moves[p, c, a, 0] = -1
    moves[p, c, a, 1] = 0
  ne = src.size
  if ne == 0:
    return
  # Stateless Park-Miller streams: identical integer arithmetic on CPU/CUDA.
  rng = (seed % 2147483646 + 1 + 104729*p + 13007*iteration + 433*c) % 2147483647
  rng = (rng * 16807) % 2147483647
  e = (rng + c) % ne
  i, j, f = src[e], dst[e], fun[e]
  if level == 0:
    used_other = 0.0
    for g in range(x.shape[1]):
      if g != f:
        used_other += max(1 if x[j,g]+incoming[p,j,g]>0 else 0, math.ceil((x[j,g]+incoming[p,j,g])/capacity[j,g]-1e-9))*float(memory[g])
    replicas = math.floor((ram[j]-used_other)/memory[f])
    needed = x[j,f] + incoming[p,j,f] + rejected[p,i,f]
    limit = min(float(needed), float(replicas+1e-9)*float(capacity[j,f])) if replicas>0 else 0.0
    space = max(0, math.floor(limit)-x[j,f]-incoming[p,j,f])
    q = min(rejected[p,i,f], space)
    if c % 3 == 0:
      q = min(q, 1)
    elif c % 3 == 1:
      processing=x[j,f]+incoming[p,j,f]
      replicas=max(1 if processing>0 else 0,math.ceil(processing/capacity[j,f]-1e-9))
      boundary=min(float(processing+q),float(replicas+1e-9)*capacity[j,f])
      if boundary<processing+1:
        boundary=min(float(processing+q),float(replicas+1+1e-9)*capacity[j,f])
      q=min(q,max(0,math.floor(boundary)-processing))
    moves[p,c,0,0], moves[p,c,0,1] = e, q
  elif level == 3:  # Feasible destruction for shaking; no automatic acceptance.
    # Increase destruction from one to four whole flows, then cycle.
    # Whole flows can release replicas; single requests often cannot.
    strength = 1 + (iteration // 9) % 4
    for a in range(strength):
      rng = (rng * 16807) % 2147483647
      edge = rng % ne
      duplicate = False
      for b in range(a):
        if moves[p,c,b,0] == edge:
          duplicate = True
      if not duplicate:
        moves[p,c,a,0], moves[p,c,a,1] = edge, -y[p,edge]
  elif level == 1:
    begin, end = starts[i*x.shape[1]+f], starts[i*x.shape[1]+f+1]
    rng = (rng * 16807) % 2147483647
    target = begin + rng % (end-begin)
    q = y[p,e]
    if c % 3 == 0:
      q = min(q, 1)
    elif c % 3 == 1:
      processing=x[j,f]+incoming[p,j,f]
      replicas=max(1 if processing>0 else 0,math.ceil(processing/capacity[j,f]-1e-9))
      limit=min(float(processing),float(replicas-1+1e-9)*capacity[j,f]) if replicas>1 else 0.
      q=min(q,max(0,processing-math.floor(limit)))
    moves[p,c,0,0], moves[p,c,0,1] = e, -q
    moves[p,c,1,0], moves[p,c,1,1] = target, q
  elif c % 4 >= 2:
    # Relocate an existing incoming flow, then use the released memory at j.
    begin,end=in_starts[j],in_starts[j+1]
    if begin==end or rejected[p,i,f]==0:
      return
    rng=(rng*16807)%2147483647
    other=in_edges[begin+rng%(end-begin)]
    origin,g=src[other],fun[other]
    begin,end=starts[origin*x.shape[1]+g],starts[origin*x.shape[1]+g+1]
    rng=(rng*16807)%2147483647
    target=begin+rng%(end-begin)
    if dst[target]==j or y[p,other]==0:
      return
    q1=y[p,other]
    if c%4==2:
      processing=x[j,g]+incoming[p,j,g]
      replicas=max(1 if processing>0 else 0,math.ceil(processing/capacity[j,g]-1e-9))
      limit=min(float(processing),float(replicas-1+1e-9)*capacity[j,g]) if replicas>1 else 0.
      q1=min(q1,max(0,processing-math.floor(limit)))
    used_other=0.
    for h in range(x.shape[1]):
      if h!=f:
        processing=x[j,h]+incoming[p,j,h]-(q1 if h==g else 0)
        used_other+=max(1 if processing>0 else 0,math.ceil(processing/capacity[j,h]-1e-9))*float(memory[h])
    replicas=math.floor((ram[j]-used_other)/memory[f])
    processing=x[j,f]+incoming[p,j,f]-(q1 if f==g else 0)
    needed=processing+rejected[p,i,f]
    limit=min(float(needed),float(replicas+1e-9)*capacity[j,f]) if replicas>0 else 0.
    q2=min(rejected[p,i,f],max(0,math.floor(limit)-processing))
    moves[p,c,0,0],moves[p,c,0,1]=other,-q1
    moves[p,c,1,0],moves[p,c,1,1]=target,q1
    moves[p,c,2,0],moves[p,c,2,1]=e,q2
  else:
    rng = (rng * 16807) % 2147483647
    other = rng % ne
    if (fun[other] == f and src[other] == i) or dst[other] == j:
      return
    target1, target2 = -1, -1
    for a in range(starts[i*x.shape[1]+f], starts[i*x.shape[1]+f+1]):
      if dst[a] == dst[other]:
        target1 = a
    origin, f2 = src[other], fun[other]
    for a in range(starts[origin*x.shape[1]+f2], starts[origin*x.shape[1]+f2+1]):
      if dst[a] == j:
        target2 = a
    if target1 < 0 or target2 < 0:
      return
    # Independent block sizes also allow memory exchanges between unequal loads.
    q1, q2 = y[p,e], y[p,other]
    if c % 2 == 0:
      q1, q2 = min(q1,1), min(q2,1)
    moves[p,c,0,0], moves[p,c,0,1] = e, -q1
    moves[p,c,1,0], moves[p,c,1,1] = target1, q1
    moves[p,c,2,0], moves[p,c,2,1] = other, -q2
    moves[p,c,3,0], moves[p,c,3,1] = target2, q2
