"""Share checked records and maintain a covering upper bound across all regions."""

import math
from threading import Event, RLock
from time import monotonic

import highspy
import numpy as np

from centralized_vns_highs.core import Record, run_highs
from centralized_vns_highs.mip import apply_rows, encode_solution, fixed_model, require_ok
from .pair_cuts import relaxation, separate
from centralized_vns_r.io import integers


class Ledger:
  def __init__(self, p, regions, y, upper, gap, start):
    self.p, self.regions, self.gap, self.start = p, regions, gap, start
    self.lock, self.stop = RLock(), Event()
    self.best = Record('initial', y.copy(), p.check(y))
    self.submission = self.best
    self.elite = [[] for _ in range(regions.count)]
    self.elite[int(regions.identify(y))].append(self.best)
    self.upper = np.full(regions.count, upper, dtype=float)
    self.closed = np.zeros(regions.count, dtype=bool)
    self.closures, self.trace = [], []
    self.observe('initial')

  def observe(self, cause):
    lower = self.best.welfare
    upper = float(self.upper.max())
    if upper < lower - 1e-7 * max(1.0, abs(lower)):
      raise ArithmeticError('covering bound below checked incumbent')
    allowed = lower + self.gap * max(1.0, abs(lower))
    for b in np.flatnonzero(~self.closed & (self.upper <= allowed)):
      self.closures.append(
        dict(
          branch=int(b),
          cause=cause,
          upper=float(self.upper[b]),
          lower=lower,
          previous_lower=self.trace[-1]['lower'] if self.trace else None,
          exact=bool(self.upper[b] <= lower),
          elapsed_s=monotonic() - self.start,
        )
      )
      self.closed[b] = True
    if self.closed.all():
      self.stop.set()
    if not self.trace or lower != self.trace[-1]['lower'] or upper != self.trace[-1]['upper']:
      self.trace.append(
        dict(elapsed_s=monotonic() - self.start, lower=lower, upper=max(lower, upper), cause=cause)
      )

  def offer(self, source, y, expected=None):
    checked = self.p.check(y)
    branch = int(self.regions.identify(y))
    if expected is not None and branch != expected:
      raise RuntimeError('solution crossed its assigned region')
    record = Record(source, y.copy(), checked)
    with self.lock:
      if record.welfare > self.upper[branch] + 1e-7 * max(1.0, abs(record.welfare)):
        raise ArithmeticError('regional bound below checked witness')
      archive = self.elite[branch]
      if (len(archive) < 3 or record.welfare >= archive[-1].welfare) and not any(
        np.array_equal(y, r.y) for r in archive
      ):
        archive.append(record)
        archive.sort(key=lambda r: -r.welfare)
        del archive[3:]
      if record.welfare > self.best.welfare:
        self.best = record
        self.submission = record if source in ('highs', 'vns') else Record('vns', record.y, checked)
        self.observe(source)
    return record

  def latest(self):
    # The baseline callback accepts records tagged 'vns' from external producers.
    with self.lock:
      return self.submission

  def tighten(self, upper, branch=None):
    if not math.isfinite(upper):
      return
    with self.lock:
      ix = slice(None) if branch is None else branch
      if np.any(self.upper[ix] > upper):
        if branch is not None and self.elite[branch]:
          witness = self.elite[branch][0].welfare
          if upper < witness - 1e-7 * max(1.0, abs(witness)):
            raise ArithmeticError('invalid branch certificate')
        self.upper[ix] = np.minimum(self.upper[ix], upper)
        self.observe('global_bound' if branch is None else 'branch_bound')

  def witness(self, b):
    with self.lock:
      if self.elite[b]:
        return self.elite[b][0]
      parent = self.best.y
    return self.offer('projected_seed', self.regions.seed(self.p, b, parent), b)


def certify_branch(book, b, deadline, *, tighten_receive=False, pair_rows=None, group_rows=None):
  p = book.p
  solver = fixed_model(p, tighten_receive=tighten_receive)
  book.regions.constrain(solver, p, b)
  counts = dict(pair_cuts=0, group_cuts=0)
  if pair_rows is not None or group_rows is not None:
    apply_rows(solver, (pair_rows or []) + (group_rows or []))
    for rows, key in ((pair_rows, 'pair_cuts'), (group_rows, 'group_cuts')):
      if rows is None:
        continue
      if not relaxation(solver, deadline):
        break
      book.tighten(p.base_welfare - solver.getObjectiveValue(), b)
      if book.closed[b]:
        break
      lp = solver.getLp()
      count = len(p.src)
      local = separate(
        p,
        solver.getSolution().col_value[:count],
        np.asarray(lp.col_lower_[:count], dtype=np.int64),
        np.asarray(lp.col_upper_[:count], dtype=np.int64),
        deadline,
        groups=(key == 'group_cuts'),
      )
      apply_rows(solver, local)
      counts[key] = len(local)
    require_ok(solver.setOptionValue('solve_relaxation', False))
    if book.closed[b] or monotonic() >= deadline:
      return dict(
        branch=b,
        status='LP bound' if book.closed[b] else 'Time limit',
        nodes=0,
        rejected=0,
        **counts,
      )
  seed = book.witness(b)
  initial = highspy.HighsSolution()
  initial.col_value = encode_solution(p, seed.y, seed.checked)
  require_ok(solver.setSolution(initial))
  sent = seed

  def query(event):
    nonlocal sent
    r = book.witness(b)
    if r is not sent:
      require_ok(event.data_in.setSolution(encode_solution(p, r.y, r.checked)))
      sent = r

  def interrupt(event):
    if book.stop.is_set() or book.closed[b] or monotonic() >= deadline:
      event.interrupt()

  errors = []

  def guarded(fn):
    def call(event):
      try:
        fn(event)
      except Exception as exc:
        errors.append(exc)
        event.interrupt()

    return call

  solver.cbMipUserSolution.subscribe(guarded(query))
  solver.cbMipInterrupt.subscribe(guarded(interrupt))
  require_ok(solver.setOptionValue('time_limit', max(0.0, deadline - monotonic())))
  status = solver.run()
  if errors:
    raise errors[0]
  state = solver.getModelStatus()
  permitted = highspy.HighsModelStatus
  if status == highspy.HighsStatus.kError or state not in (
    permitted.kOptimal,
    permitted.kTimeLimit,
    permitted.kInterrupt,
    permitted.kHighsInterrupt,
  ):
    raise RuntimeError(f'branch {b} with feasible witness: {solver.modelStatusToString(state)}')
  info = solver.getInfo()
  rejected = 0
  if info.valid and info.primal_solution_status == highspy.SolutionStatus.kSolutionStatusFeasible:
    try:
      book.offer('branch_highs', integers(solver.getSolution().col_value[: len(p.src)], 'flow'), b)
    except ValueError:
      rejected = 1
  if info.valid:
    upper = p.base_welfare - info.mip_dual_bound
    if state == permitted.kOptimal and not rejected:
      witness = book.witness(b).welfare
      # Native objective and checker sum in different orders.
      roundoff = 32 * np.finfo(float).eps * max(1.0, abs(p.base_welfare), abs(info.mip_dual_bound))
      if abs(upper - witness) <= roundoff:
        upper = witness
    book.tighten(upper, b)
  return dict(
    branch=b,
    status=solver.modelStatusToString(state),
    nodes=info.mip_node_count,
    rejected=rejected,
    **counts,
  )


def prove(
  book,
  deadline,
  slice_seconds,
  global_share,
  *,
  tighten_receive=False,
  pair_rows=None,
  group_rows=None,
  prices=None,
  device='cpu',
  max_bytes=512 * 1024**2,
):
  now = monotonic()
  root_end = deadline if book.regions.count == 1 else now + max(0.0, deadline - now) * global_share
  diagnostic = None
  if prices is not None:
    from .lagrangian import certify_regions

    diagnostic = certify_regions(
      book, prices, min(deadline, now + max(0.0, deadline - now) * 0.10), device, max_bytes
    )
    if device == 'cuda':
      import cupy

      # The resident VNS uses Numba's allocator; release unused CuPy blocks.
      cupy.get_default_memory_pool().free_all_blocks()
  root = run_highs(
    book.p,
    book,
    root_end,
    tighten_receive=tighten_receive,
    rows=(pair_rows or []) + (group_rows or []),
  )
  if diagnostic is not None:
    root['lagrangian'] = diagnostic
  visits = np.zeros(book.regions.count, dtype=np.int64)
  solved = np.zeros(book.regions.count, dtype=bool)
  tasks = []
  while not book.stop.is_set() and monotonic() < deadline:
    with book.lock:
      live = np.flatnonzero(~(book.closed | solved))
      if not len(live):
        break
      # Ties at the inherited bound rotate over previously unexplored regions.
      b = int(live[np.lexsort((live, visits[live], -book.upper[live]))[0]])
    budget = slice_seconds * (2 ** min(int(visits[b]), 10))
    visits[b] += 1
    tasks.append(
      certify_branch(
        book,
        b,
        min(deadline, monotonic() + budget),
        tighten_receive=tighten_receive,
        pair_rows=pair_rows,
        group_rows=group_rows,
      )
    )
    solved[b] = tasks[-1]['status'] == 'Optimal'
  return root, tasks
