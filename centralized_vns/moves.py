"""Scalar numerical routines shared by Python and lazily compiled CUDA kernels.

A move has four (edge, signed integer flow) entries; -1 is an unused entry.
All entries are evaluated atomically, including repeated edges/nodes.
"""
import math


def score_move(p, candidate, moves, y, incoming, rejected, load, x, capacity,
               memory, ram, src, dst, fun, reward):
  gain = 0.0
  for a in range(4):
    e = moves[p, candidate, a, 0]
    if e < 0:
      continue
    change = 0
    for b in range(4):
      if moves[p, candidate, b, 0] == e:
        change += moves[p, candidate, b, 1]
    if y[p, e] + change < 0:
      return -math.inf
    gain += moves[p, candidate, a, 1] * reward[e]
    # Check every affected endpoint, incorporating the complete compound move.
    for endpoint in range(2):
      node = src[e] if endpoint == 0 else dst[e]
      used = 0.0
      for f in range(x.shape[1]):
        di, do = 0, 0
        for b in range(4):
          other = moves[p, candidate, b, 0]
          if other >= 0 and fun[other] == f:
            q = moves[p, candidate, b, 1]
            if src[other] == node:
              do += q
            if dst[other] == node:
              di += q
        inc = incoming[p, node, f] + di
        z = rejected[p, node, f] - do
        outgoing = load[node, f] - x[node, f] - z
        if inc < 0 or z < 0 or outgoing < 0 or (inc > 0 and outgoing > 0):
          return -math.inf
        # Match LSP's 1e-9 replica-ratio tolerance, but positive load needs a replica.
        processing = x[node, f] + inc
        ratio = float(processing) / float(capacity[node, f])
        if not math.isfinite(ratio) or ratio > 2**40:
          return -math.inf
        replicas = max(1 if processing > 0 else 0, math.ceil(ratio - 1e-9))
        used += float(replicas) * memory[f]
      if used > ram[node]:
        return -math.inf
  return gain


def generate_move(seed, iteration, p, c, level, moves, y, incoming, rejected,
                  x, capacity, memory, ram, src, dst, fun, starts):
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
    if c % 2 == 0:
      q = min(q, 1)
    moves[p,c,0,0], moves[p,c,0,1] = e, -q
    moves[p,c,1,0], moves[p,c,1,1] = target, q
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
