"""Sample positive integer transfers inside a dyadic flow region, on CPU or CUDA."""

import math
from functools import lru_cache
import numpy as np
from centralized_vns.moves import score_move
from centralized_vns.core import CPU
from centralized_vns_frontier.pool import Pool
from centralized_vns_highs.mip import ram_receive_limit


def interval(e, bit, cut, code, depth, cap):
  k = bit[e]
  if 0 <= k < depth:
    if code & (1 << k):
      return cut[e] + 1, cap[e]
    return 0, min(cut[e], cap[e])
  return 0, cap[e]


def make_generator(device):
  limits = device(interval)

  @device
  def generate(
    seed,
    iteration,
    p,
    c,
    level,
    moves,
    y,
    inc,
    z,
    load,
    x,
    capacity,
    memory,
    ram,
    src,
    dst,
    fun,
    starts,
    in_edges,
    in_starts,
    bit,
    cut,
    code,
    depth,
    eligible,
    cap,
  ):
    for k in range(4):
      moves[p, c, k, 0], moves[p, c, k, 1] = -1, 0
    if depth < 0 or not eligible.size:
      return False
    rng = (seed + 1 + 104729 * p + 13007 * iteration + 433 * c) % 2147483647
    for retry in range(8):
      rng = (rng * 16807 + 1) % 2147483647
      e = eligible[(rng + c) % eligible.size]
      s, j, f = src[e], dst[e], fun[e]
      lo, hi = limits(e, bit, cut, code, depth, cap)
      old = -1
      if level == 3:
        q = y[p, e] - lo
        if q > 0:
          moves[p, c, 0, 0], moves[p, c, 0, 1] = e, -q
          return True
        continue
      if level == 1:
        # Relocate a strictly positive amount to another destination of the source.
        movable = y[p, e] - lo
        begin, end = starts[s * x.shape[1] + f], starts[s * x.shape[1] + f + 1]
        if movable <= 0 or end - begin < 2:
          continue
        old = e
        rng = (rng * 16807 + 1) % 2147483647
        e = begin + rng % (end - begin - 1)
        if e >= old:
          e += 1
        j = dst[e]
        lo, hi = limits(e, bit, cut, code, depth, cap)
        q = movable
      else:
        q = z[p, s, f]
      q = min(q, hi - y[p, e])
      if q <= 0 or inc[p, s, f] > 0 or z[p, j, f] < load[j, f] - x[j, f]:
        continue
      released = 0
      other_f = -1
      if level == 2:
        # Replace another function's incoming traffic at the same receiver.
        begin, end = in_starts[j], in_starts[j + 1]
        rng = (rng * 16807 + 1) % 2147483647
        old = in_edges[begin + rng % (end - begin)]
        other_f = fun[old]
        if other_f == f:
          continue
        lower, _ = limits(old, bit, cut, code, depth, cap)
        released = y[p, old] - lower
        if released <= 0:
          continue
      used = 0
      for g in range(x.shape[1]):
        if g != f:
          work = x[j, g] + inc[p, j, g] - (released if g == other_f else 0)
          replicas = max(1 if work > 0 else 0, math.ceil(work / capacity[j, g] - 1e-9))
          used += replicas * memory[g]
      replicas = (ram[j] - used) // memory[f]
      work = x[j, f] + inc[p, j, f]
      if replicas <= 0:
        continue
      # Cap before conversion even when service capacity is extremely large.
      room = math.floor(min(float(work + q), (float(replicas) + 1e-9) * capacity[j, f])) - work
      q = min(q, room)
      if q <= 0:
        continue
      if c % 3 == 0:
        q = 1
      elif c % 3 == 1:
        copies = max(1 if work > 0 else 0, math.ceil(work / capacity[j, f] - 1e-9))
        boundary = min(float(work + q), (copies + 1e-9) * capacity[j, f])
        if boundary < work + 1:
          boundary = min(float(work + q), (copies + 1 + 1e-9) * capacity[j, f])
        q = min(q, max(1, math.floor(boundary) - work))
      moves[p, c, 0, 0], moves[p, c, 0, 1] = e, q
      if old >= 0:
        moves[p, c, 1, 0], moves[p, c, 1, 1] = old, -(q if level == 1 else released)
      return True
    return False

  return generate


generate_cpu = make_generator(lambda fn: fn)


@lru_cache(maxsize=1)
def kernel():
  global cuda
  from numba import cuda as module

  cuda = module
  generate = make_generator(cuda.jit(device=True))
  score = cuda.jit(device=True)(score_move)

  @cuda.jit
  def candidates(
    seed,
    iteration,
    shake,
    levels,
    stale,
    moves,
    gains,
    y,
    inc,
    z,
    load,
    x,
    capacity,
    memory,
    ram,
    src,
    dst,
    fun,
    reward,
    starts,
    in_edges,
    in_starts,
    bit,
    cut,
    codes,
    depths,
    eligible,
    cap,
  ):
    idx = cuda.grid(1)
    p, c = idx // gains.shape[1], idx % gains.shape[1]
    if p < gains.shape[0]:
      level = 3 if stale[p] >= 3 * shake else levels[p]
      valid = generate(
        seed,
        iteration,
        p,
        c,
        level,
        moves,
        y,
        inc,
        z,
        load,
        x,
        capacity,
        memory,
        ram,
        src,
        dst,
        fun,
        starts,
        in_edges,
        in_starts,
        bit,
        cut,
        codes[p],
        depths[p],
        eligible,
        cap,
      )
      gains[p, c] = (
        score(p, c, moves, y, inc, z, load, x, capacity, memory, ram, src, dst, fun, reward)
        if valid
        else -math.inf
      )

  return candidates


class TreePool(Pool):
  def __init__(self, p, o, full, depth):
    from dataclasses import replace

    cap = (p.load - p.x)[p.src, p.fun].copy()
    supply = np.zeros_like(p.x)
    np.add.at(supply, (p.dst, p.fun), cap)
    baseline = p.check(np.zeros(len(cap), np.int64))['replicas']
    receiving = ram_receive_limit(p, supply.ravel(), baseline).reshape(p.x.shape)
    cap = np.minimum(cap, receiving[p.dst, p.fun])
    eligible = np.flatnonzero(cap > 0)
    self.depths = np.full(o.pool_size, depth, np.int64)
    reserve = cap.nbytes + eligible.nbytes + self.depths.nbytes
    super().__init__(p, replace(o, max_device_bytes=o.max_device_bytes - reserve), full)
    self.cap, self.eligible = cap, eligible
    if o.backend == 'cuda':
      e = self.engine
      e.candidate_kernel = kernel()
      e.domains += tuple(e.cuda.to_device(a) for a in (self.depths, eligible, cap))

  def assign(self):
    if self.backend == 'cuda':
      self.engine.domains[2].copy_to_device(self.branches)
      self.engine.domains[3].copy_to_device(self.depths)

  def step(self, iteration):
    e = self.engine
    if self.backend == 'cuda':
      return e.step(iteration)
    p, o, s = e.p, e.o, e.state
    for i in range(o.pool_size):
      level = 3 if e.stale[i] >= 3 * o.shake_after else e.level[i]
      gains = np.full(o.candidates, -np.inf)
      for c in range(o.candidates):
        if generate_cpu(
          o.seed,
          iteration,
          i,
          c,
          level,
          e.moves,
          s.y,
          s.incoming,
          s.rejected,
          p.load,
          p.x,
          p.capacity,
          p.memory,
          p.ram,
          p.src,
          p.dst,
          p.fun,
          p.starts,
          e.in_edges,
          e.in_starts,
          e.bit,
          e.cut,
          self.branches[i],
          self.depths[i],
          self.eligible,
          self.cap,
        ):
          gains[c] = score_move(i, c, e.moves, s.y, s.incoming, s.rejected, *p.arrays())
      CPU.accept(e, i, level, gains)
    e.evaluated += o.pool_size * o.candidates
    return e.best.copy()
