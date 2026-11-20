"""CUDA branch filtering and component path relinking on resident topology."""

from functools import lru_cache
import math
import numpy as np
from centralized_vns_r.gpu import CUDA


@lru_cache(maxsize=1)
def kernels():
  global cuda
  from numba import cuda as module

  cuda = module
  from centralized_vns_r.moves import generate_move
  from centralized_vns.moves import score_move
  from .moves import changes_flow

  generate = cuda.jit(device=True)(generate_move)
  score = cuda.jit(device=True)(score_move)
  changed = cuda.jit(device=True, inline=True)(changes_flow)

  @cuda.jit
  def candidates(
    seed,
    iteration,
    shake_after,
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
    branches,
  ):
    idx = cuda.grid(1)
    p, c = idx // gains.shape[1], idx % gains.shape[1]
    if p >= gains.shape[0]:
      return
    level = 3 if stale[p] >= shake_after * 3 else levels[p]
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
      capacity,
      memory,
      ram,
      src,
      dst,
      fun,
      starts,
      in_edges,
      in_starts,
    )
    if not changed(moves, p, c):
      gains[p, c] = -math.inf
      return
    for k in range(4):
      e = moves[p, c, k, 0]
      if e >= 0 and bit[e] >= 0:
        v = y[p, e]
        for a in range(4):
          if moves[p, c, a, 0] == e:
            v += moves[p, c, a, 1]
        if (v > cut[e]) != (((branches[p] >> bit[e]) & 1) != 0):
          gains[p, c] = -math.inf
          return
    gains[p, c] = score(p, c, moves, y, inc, z, load, x, capacity, memory, ram, src, dst, fun, reward)

  @cuda.jit
  def path(src, dst, reward, start, guide, labels, gains, out):
    # One cooperative block per parental pair. Each pair owns its labels and gain sums.
    pair, lane = cuda.blockIdx.x, cuda.threadIdx.x
    again = cuda.shared.array(1, np.int32)
    for n in range(lane, labels.shape[1], cuda.blockDim.x):
      labels[pair, n], gains[pair, n] = n, 0.0
    cuda.syncthreads()
    while True:
      if lane == 0:
        again[0] = 0
      cuda.syncthreads()
      for e in range(lane, src.size, cuda.blockDim.x):
        if start[pair, e] != guide[pair, e]:
          a, b = src[e], dst[e]
          low = min(labels[pair, a], labels[pair, b])
          old_a = cuda.atomic.min(labels, (pair, a), low)
          old_b = cuda.atomic.min(labels, (pair, b), low)
          if old_a > low or old_b > low:
            cuda.atomic.max(again, 0, 1)
      cuda.syncthreads()
      changed = again[0]
      cuda.syncthreads()
      if not changed:
        break
    for e in range(lane, src.size, cuda.blockDim.x):
      if start[pair, e] != guide[pair, e]:
        cuda.atomic.add(
          gains, (pair, labels[pair, src[e]]), (guide[pair, e] - start[pair, e]) * reward[e]
        )
    cuda.syncthreads()
    for e in range(lane, src.size, cuda.blockDim.x):
      out[pair, e] = guide[pair, e] if gains[pair, labels[pair, src[e]]] > 0 else start[pair, e]

  return candidates, path


class BoundedCUDA(CUDA):
  def __init__(self, p, o, bit, cut, branches):
    super().__init__(p, o)
    self.domains = tuple(self.cuda.to_device(a) for a in (bit, cut, branches))
    self.candidate_kernel, self.path = kernels()

  def step(self, iteration):
    o = self.o
    self.candidate_kernel[(o.pool_size * o.candidates + 63) // 64, 64](
      o.seed,
      iteration,
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
      *self.domains,
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
    self.evaluated += o.pool_size * o.candidates
    return self.best.copy_to_host()

  def relink(self, starts, guides):
    device = self.cuda
    nodes = self.data[0].shape[0]
    bytes_per_pair = 8 * (3 * starts.shape[1] + 2 * nodes)
    batch = min(len(starts), (self.o.max_device_bytes - self.device_bytes) // bytes_per_pair)
    if batch < 1:
      raise ValueError('CUDA memory budget insufficient for a relinking pair')
    a = device.device_array((batch, starts.shape[1]), np.int64)
    g = device.device_array_like(a)
    out = device.device_array_like(a)
    labels = device.device_array((batch, nodes), np.int64)
    gains = device.device_array((batch, nodes), np.float64)
    children = np.empty_like(starts)
    for i in range(0, len(starts), batch):
      count = min(batch, len(starts) - i)
      a[:count].copy_to_device(starts[i : i + count])
      g[:count].copy_to_device(guides[i : i + count])
      self.path[count, 64](self.data[5], self.data[6], self.data[8], a, g, labels, gains, out)
      out[:count].copy_to_host(children[i : i + count])
    return children
