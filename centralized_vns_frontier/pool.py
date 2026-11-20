"""Branch-aware candidate acceptance and resident population state."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from centralized_vns.moves import score_move
from centralized_vns_r.core import CPU
from .moves import changes_flow


def allowed(moves, y, bit, cut, branch):
  if not len(y):
    return np.ones(len(moves), dtype=bool)
  edges, deltas = moves[:, :, 0], moves[:, :, 1]
  valid = np.ones(len(moves), dtype=bool)
  for k in range(edges.shape[1]):
    e = edges[:, k]
    split = bit[e]
    tested = (e >= 0) & (split >= 0)
    if np.any(tested):
      touched = e[tested]
      value = y[touched] + ((edges[tested] == touched[:, None]) * deltas[tested]).sum(axis=1)
      valid[tested] &= (value > cut[touched]) == (((branch >> split[tested]) & 1) != 0)
  return valid


class BoundedCPU(CPU):
  @staticmethod
  def score(index, candidate, moves, *state):
    if not changes_flow(moves, index, candidate):
      return -np.inf
    return score_move(index, candidate, moves, *state)

  def accept(self, index, level, gains):
    gains = np.asarray(gains)
    gains[
      ~allowed(self.moves[index], self.state.y[index], self.bit, self.cut, self.branches[index])
    ] = -np.inf
    super().accept(index, level, gains)


def metal_engine(p, o, bit, cut, branches):
  from centralized_vns_r import metal

  engine = metal.Metal(p, o)
  mx = engine.mx
  names = [
    'load',
    'x',
    'ram',
    'src',
    'dst',
    'fun',
    'starts',
    'costs',
    'memory',
    'limits',
    'in_edges',
    'in_starts',
    'split_bit',
    'split_cut',
    'branches',
    'y',
    'inc',
    'z',
    'space',
    'fill',
    'release',
    'levels',
    'params',
  ]
  engine.data.extend([mx.array(bit), mx.array(cut), mx.array(branches)])
  source = (
    '#define FRONTIER_REQUIRE_CHANGE 1\n'
    + Path(metal.__file__).with_suffix('.metal').read_text()
    + """
    for(long a=0;a<4 && valid[idx];++a) {
      long e=edges[a];
      if(e<0 || split_bit[e]<0) continue;
      long v=y[p*ne+e];
      for(long b=0;b<4;++b) if(edges[b]==e) v+=delta[b];
      if ((v>split_cut[e]) != bool((branches[p]>>split_bit[e])&1L)) valid[idx]=false;
    }
  """
  )
  engine.kernel = mx.fast.metal_kernel(
    name='frontier_candidates', input_names=names, output_names=['moves', 'valid'], source=source
  )
  return engine


class Pool:
  def __init__(self, p, o, regions):
    self.backend = o.backend
    self.branches = np.arange(o.pool_size, dtype=np.int64) % regions.count
    bit = np.full(len(p.src), -1, np.int64)
    cut = np.zeros_like(bit)
    bit[regions.edges] = np.arange(len(regions.edges))
    cut[regions.edges] = regions.cuts
    if o.backend != 'cpu':
      o = replace(
        o, max_device_bytes=o.max_device_bytes - bit.nbytes - cut.nbytes - self.branches.nbytes
      )
    if o.backend == 'metal':
      self.engine = metal_engine(p, o, bit, cut, self.branches)
    elif o.backend == 'cuda':
      from .gpu import BoundedCUDA

      self.engine = BoundedCUDA(p, o, bit, cut, self.branches)
    else:
      self.engine = BoundedCPU(p, o)
      self.engine.bit, self.engine.cut, self.engine.branches = bit, cut, self.branches

  def assign(self):
    if self.backend == 'metal':
      self.engine.data[-1] = self.engine.mx.array(self.branches)
    elif self.backend == 'cuda':
      self.engine.domains[-1].copy_to_device(self.branches)

  def seed(self, slot, y, checked):
    e = self.engine
    state = e if self.backend == 'cuda' else e.state
    inc = e.inc if self.backend == 'cuda' else state.incoming
    rejected = e.z if self.backend == 'cuda' else state.rejected
    for target, v in (
      (state.y, y),
      (e.best_y, y),
      (inc, checked['incoming']),
      (rejected, checked['z']),
      (e.value, checked['welfare']),
      (e.best, checked['welfare']),
      (e.level, 0),
      (e.stale, 0),
    ):
      if self.backend == 'cuda':
        target[slot : slot + 1].copy_to_device(np.asarray(v, dtype=target.dtype)[None])
      else:
        target[slot] = v

  def record(self, slot):
    archive = self.engine.best_y
    return (
      archive[slot : slot + 1].copy_to_host()[0] if self.backend == 'cuda' else archive[slot].copy()
    )
