"""An exhaustive partition of integer flows in the original model coordinates."""

from time import monotonic

import highspy
import numpy as np

from centralized_vns_highs.mip import apply_rows, fixed_model, require_ok
from .pair_cuts import relaxation, separate


class Regions:
  def __init__(self, edges, cuts):
    self.edges = np.asarray(edges, dtype=np.int64)
    self.cuts = np.asarray(cuts, dtype=np.int64)
    self.bits = 1 << np.arange(len(edges), dtype=np.int64)
    self.count = 1 << len(edges)

  def identify(self, y):
    return (y[..., self.edges] > self.cuts) @ self.bits

  def seed(self, p, branch, parent):
    right = (branch & self.bits) != 0
    differs = (parent[self.edges] > self.cuts) != right
    y = parent.copy()
    # Removing traffic incident to a newly forced edge frees its baseline RAM.
    forced = self.edges[right & differs]
    if forced.size:
      nodes = np.r_[p.src[forced], p.dst[forced]]
      y[np.isin(p.src, nodes) | np.isin(p.dst, nodes)] = 0
    y[self.edges[differs]] = self.cuts[differs] + right[differs]
    return y

  def constrain(self, solver, p, branch):
    for edge, cut, bit in zip(self.edges, self.cuts, self.bits):
      limit = p.load[p.src[edge], p.fun[edge]] - p.x[p.src[edge], p.fun[edge]]
      lo, hi = (cut + 1, limit) if branch & bit else (0, cut)
      require_ok(solver.changeColBounds(int(edge), float(lo), float(hi)))


def partition(
  p, max_regions, deadline, *, tighten_receive=False, pair_rows=None, group_rows=None, dual=None
):
  root = fixed_model(p, tighten_receive=tighten_receive)
  require_ok(root.setOptionValue('solve_relaxation', True))
  require_ok(root.setOptionValue('time_limit', max(0.0, deadline - monotonic())))
  status = root.run()
  if status == highspy.HighsStatus.kError:
    raise RuntimeError('root LP failed')
  if root.getModelStatus() != highspy.HighsModelStatus.kOptimal:
    return Regions([], []), np.inf
  flows = np.array(root.getSolution().col_value[: len(p.src)])
  fraction = np.abs(flows - np.rint(flows))
  edges, cuts, taken = [], [], set()
  for edge in np.argsort(-fraction, kind='stable'):
    if 2 ** (len(edges) + 1) > max_regions or fraction[edge] <= 1e-6 or monotonic() >= deadline:
      break
    ends = {p.src[edge], p.dst[edge]}
    if ends & taken:
      continue
    threshold = int(np.floor(flows[edge]))
    if threshold < 0:
      continue
    seed = np.zeros(len(flows), dtype=np.int64)
    seed[edge] = threshold + 1
    try:
      p.check(seed)
    except ValueError:
      continue
    taken.update(ends)
    edges.append(edge)
    cuts.append(threshold)
  bound = p.base_welfare - root.getObjectiveValue()
  upper = (p.load - p.x)[p.src, p.fun]
  applied = []
  for rows, groups in ((pair_rows, False), (group_rows, True)):
    if rows is None:
      continue
    rows.extend(separate(p, flows, np.zeros_like(upper), upper, deadline, groups=groups))
    if rows:
      apply_rows(root, rows)
      if not relaxation(root, deadline):
        return Regions(edges, cuts), bound
      applied.extend(rows)
      bound = min(bound, p.base_welfare - root.getObjectiveValue())
      flows = np.asarray(root.getSolution().col_value[: len(p.src)])
  if dual is not None:
    from .lagrangian import price_profiles

    dual.extend(price_profiles(p, root.getSolution().row_dual, applied))
  return Regions(edges, cuts), bound
