"""Node-local flow moves, coordinated evictions and parallel selection on CUDA."""

import math
from functools import lru_cache
import numpy as np

from centralized_vns.core import Options
from centralized_vns_r.gpu import CUDA
from centralized_vns_highs.mip import ram_receive_limit
from .heuristic import NodeSearch


@lru_cache(maxsize=1)
def node_kernel():
  global cuda
  from numba import cuda as device
  from centralized_vns_r.moves import generate_move
  from centralized_vns.moves import score_move
  from centralized_vns_frontier.moves import changes_flow

  cuda = device
  generate = cuda.jit(device=True)(generate_move)
  score = cuda.jit(device=True)(score_move)
  changed = cuda.jit(device=True)(changes_flow)

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
    cap,
    memory,
    ram,
    src,
    dst,
    fun,
    reward,
    starts,
    in_edges,
    in_starts,
    eligible,
    lo,
    hi,
  ):
    idx = cuda.grid(1)
    p, c = idx // gains.shape[1], idx % gains.shape[1]
    if p >= gains.shape[0]:
      return
    level = 3 if stale[p] >= shake * 3 else levels[p]
    if level == 2 and c % 2 == 0 and eligible.size:
      for k in range(4):
        moves[p, c, k, 0], moves[p, c, k, 1] = -1, 0
      rng = (
        (seed % 2147483646 + 1 + 104729 * p + 13007 * iteration + 433 * c) % 2147483647 * 16807
      ) % 2147483647
      edge = eligible[(rng + c) % eligible.size]
      i, j, f = src[edge], dst[edge], fun[edge]
      want = z[p, i, f]
      if want > 0 and inc[p, i, f] == 0 and z[p, j, f] == load[j, f] - x[j, f]:
        begin = in_starts[j]
        size = in_starts[j + 1] - begin
        offset = (rng * 16807 % 2147483647) % size
        count = 0
        for n in range(size):
          d = in_edges[begin + (offset + n) % size]
          g = fun[d]
          if g == f or y[p, d] == 0:
            continue
          work = x[j, g] + inc[p, j, g]
          reps = max(1, math.ceil(work / cap[j, g] - 1e-9))
          needed = (
            work - min(work, math.floor(min(float(work), (reps - 1 + 1e-9) * cap[j, g])))
            if reps > 1
            else work
          )
          if count and fun[moves[p, c, 1, 0]] == g:
            needed += moves[p, c, 1, 1]
          q = y[p, d] if c % 8 == 0 else min(y[p, d], max(0, needed))
          if q:
            count += 1
            moves[p, c, count, 0], moves[p, c, count, 1] = d, -q
            if count == 2 or c % 4 == 2:
              break
        old_used, used = 0, 0
        for g in range(x.shape[1]):
          if g == f:
            continue
          work = x[j, g] + inc[p, j, g]
          old_used += max(int(work > 0), math.ceil(work / cap[j, g] - 1e-9)) * memory[g]
          for k in range(1, count + 1):
            if fun[moves[p, c, k, 0]] == g:
              work += moves[p, c, k, 1]
          used += max(int(work > 0), math.ceil(work / cap[j, g] - 1e-9)) * memory[g]
        work = x[j, f] + inc[p, j, f]
        before = (ram[j] - old_used) // memory[f]
        after = (ram[j] - used) // memory[f]
        limit = work + want
        old_space = (
          max(0, math.floor(min(float(limit), (before + 1e-9) * cap[j, f])) - work) if before else 0
        )
        admit = (
          max(0, math.floor(min(float(limit), (after + 1e-9) * cap[j, f])) - work) if after else 0
        )
        if count and admit > old_space:
          moves[p, c, 0, 0], moves[p, c, 0, 1] = edge, admit
        else:
          gains[p, c] = -math.inf
          return
    else:
      generate(
        seed,
        iteration,
        p,
        c,
        level,
        moves,
        y,
        inc,
        z,
        x,
        cap,
        memory,
        ram,
        src,
        dst,
        fun,
        starts,
        in_edges,
        in_starts,
      )
    gains[p, c] = -math.inf
    if not changed(moves, p, c):
      return
    for k in range(4):
      e = moves[p, c, k, 0]
      if e >= 0:
        v = y[p, e]
        for a in range(4):
          if moves[p, c, a, 0] == e:
            v += moves[p, c, a, 1]
        if v < lo[e] or v > hi[e]:
          return
    gains[p, c] = score(p, c, moves, y, inc, z, load, x, cap, memory, ram, src, dst, fun, reward)

  return candidates


class NodePool(CUDA):
  def __init__(self, p, o):
    super().__init__(p, o)
    supply = (p.load - p.x)[p.src, p.fun]
    incoming = np.zeros_like(p.x)
    np.add.at(incoming, (p.dst, p.fun), supply)
    base = p.check(np.zeros_like(supply))["replicas"]
    cap = ram_receive_limit(p, incoming.ravel(), base).reshape(p.x.shape)
    self.eligible = self.cuda.to_device(np.flatnonzero((supply > 0) & (cap[p.dst, p.fun] > 0)))
    self.lo, self.hi = (self.cuda.device_array_like(self.data[5]) for _ in range(2))
    self.node_candidates = node_kernel()

  def advance(self, it, count):
    o = self.o
    self.node_candidates[(o.pool_size * o.candidates + 63) // 64, 64](
      o.seed,
      it,
      o.shake_after,
      self.level,
      self.stale,
      self.moves,
      self.gains,
      self.y,
      self.inc,
      self.z,
      *self.data,
      self.starts,
      self.in_edges,
      self.in_starts,
      self.eligible,
      self.lo,
      self.hi,
    )
    self.accept_kernel[o.pool_size, 64](
      o.shake_after,
      self.level,
      self.stale,
      self.moves,
      self.gains,
      self.y,
      self.inc,
      self.z,
      self.value,
      self.best,
      self.best_y,
      *self.data[5:8],
    )
    return float(self.best.copy_to_host().max())


class NodeCUDA(NodeSearch):
  def engine(self, lo, hi):
    if self.pool is None:
      self.pool = NodePool(
        self.shared.p,
        Options(
          backend="cuda", pool_size=64, candidates=512, max_device_bytes=3 * 1024**3, seed=4850
        ),
      )
    self.pool.lo.copy_to_device(lo)
    self.pool.hi.copy_to_device(hi)
    return self.pool

  def seed_slots(self, slots, y, checked):
    e = self.pool
    for target, value in (
      (e.y, y),
      (e.best_y, y),
      (e.inc, checked["incoming"]),
      (e.z, checked["z"]),
      (e.best, checked["welfare"]),
      (e.value, checked["welfare"]),
      (e.level, 0),
      (e.stale, 0),
    ):
      # Strided CUDA slices cannot receive a single contiguous host upload.
      ids = range(*slots.indices(e.o.pool_size))
      if ids.step == 1:
        dest = target[ids.start : ids.stop]
        dest.copy_to_device(np.full(dest.shape, value, dtype=target.dtype))
      else:
        for i in ids:
          dest = target[i : i + 1]
          dest.copy_to_device(np.full(dest.shape, value, dtype=target.dtype))
