"""Shared regional relaxation and outward-rounded certificates for Ax <= b."""

from fractions import Fraction

import highspy
import numpy as np
from scipy.sparse import csr_matrix
from centralized_vns_frontier.pair_cuts import separate
from centralized_vns_highs.mip import apply_rows, fixed_model, require_ok


def dyadic(values):
  """Encode finite binary floats as Python integers with one shared denominator."""
  unique, inverse = np.unique(values, return_inverse=True)
  ratios = [float(v).as_integer_ratio() for v in unique]
  den = max((d for _, d in ratios), default=1)
  integers = np.array([n * (den // d) for n, d in ratios], dtype=object)
  return integers[inverse], den


class Relaxation:
  def __init__(self, problem, *, strengthen=None):
    self.solver = fixed_model(problem, tighten_receive=True)
    h = self.solver
    require_ok(h.setOptionValue('solve_relaxation', True))

    def solve_root():
      require_ok(h.run())
      if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError('optimal root relaxation required')

    solve_root()
    n = len(problem.src)
    root_flow = np.array(h.getSolution().col_value[:n])
    cap = (problem.load - problem.x)[problem.src, problem.fun]
    cuts = separate(problem, root_flow, np.zeros_like(cap), cap, np.inf)
    apply_rows(h, cuts)
    self.strengthening = strengthen(h, problem) if strengthen is not None else {}
    solve_root()
    self.cuts = len(cuts)
    self.root = h.getSolution()
    self.basis = h.getBasis()
    require_ok(h.ensureRowwise())
    lp = h.getLp()
    self.A = csr_matrix(
      (lp.a_matrix_.value_, lp.a_matrix_.index_, lp.a_matrix_.start_),
      shape=(lp.num_row_, lp.num_col_),
    )
    self.b = np.asarray(lp.row_upper_)
    self.c = -np.asarray(lp.col_cost_)
    self.lo, self.hi = np.asarray(lp.col_lower_), np.asarray(lp.col_upper_)
    if not (
      np.all(np.isneginf(lp.row_lower_))
      and np.all(np.isfinite(self.b))
      and np.all(np.isfinite(self.hi))
      and np.all(self.lo == np.floor(self.lo))
      and np.all(self.hi == np.floor(self.hi))
    ):
      raise ValueError('expected upper-bounded rows and finite integral column bounds')
    self.base = Fraction(float(problem.base_welfare))
    at = self.A.T.tocsr()
    self.nonempty = np.flatnonzero(np.diff(at.indptr))
    self.starts = at.indptr[self.nonempty]
    self.rows = at.indices
    self.a_int, self.ad = dyadic(at.data)
    self.b_int, self.bd = dyadic(self.b)
    self.c_int, self.cd = dyadic(self.c)

  def boxes(self, edges, cuts, keys):
    lo, hi = (np.tile(v, (len(keys), 1)) for v in (self.lo, self.hi))
    for i, (depth, code) in enumerate(keys):
      if not (0 <= depth <= len(edges) and 0 <= code < 1 << depth):
        raise ValueError('invalid prefix')
      for k in range(depth):
        edge, cut = edges[k], cuts[k]
        if code & (1 << k):
          lo[i, edge] = max(lo[i, edge], cut + 1)
        else:
          hi[i, edge] = min(hi[i, edge], cut)
    if np.any(lo > hi):
      raise ValueError('empty region box')
    return lo, hi

  def certificate(self, price, lo, hi):
    """Any nonnegative price yields a dual bound, even far from LP optimality."""
    price = np.asarray(price)
    if price.shape != self.b.shape or not np.isfinite(price).all():
      raise ValueError('invalid dual vector')
    y, yd = dyadic(np.maximum(price, 0))
    den = max(self.ad * yd, self.cd)
    rc = self.c_int * (den // self.cd)
    if len(self.nonempty):
      aty = np.add.reduceat(self.a_int * y[self.rows], self.starts)
      rc[self.nonempty] -= aty * (den // (self.ad * yd))
    box = np.where(rc >= 0, hi, lo).astype(np.int64).astype(object)
    ub = self.base + Fraction(int(self.b_int @ y), self.bd * yd) + Fraction(int(rc @ box), den)
    out = float(ub)
    return float(np.nextafter(out, np.inf)) if Fraction(out) < ub else out
