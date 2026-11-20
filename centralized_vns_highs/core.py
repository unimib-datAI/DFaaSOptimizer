"""One HiGHS tree and one VNS population share independently checked incumbents."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
from threading import Event, Lock
from time import monotonic

import highspy
import numpy as np

from centralized_vns.core import Options
from centralized_vns_r.io import integers
from .mip import apply_rows, encode_solution, fixed_model, require_ok
from .pool import Pool


@dataclass(frozen=True)
class Record:
  source: str
  y: np.ndarray
  checked: dict

  @property
  def welfare(self):
    return self.checked['welfare']


class Exchange:
  def __init__(self, p, y, gap, start):
    self.p = p
    checked = p.check(y)
    if not math.isfinite(checked['welfare']):
      raise ValueError('initial welfare must be finite')
    self.record = Record('initial', y.astype(np.int64), checked)
    self.gap = gap
    self.start = start
    self.lock = Lock()
    self.stop = Event()
    # Relax receiving capacity and send every residual request to its best edge.
    best_reward = np.zeros_like(p.load, dtype=float)
    np.maximum.at(best_reward, (p.src, p.fun), p.reward)
    self.upper = p.base_welfare + float(np.sum((p.load - p.x) * best_reward))
    if not math.isfinite(self.upper):
      raise ValueError('initial welfare bound must be finite')
    self.trace = []
    self._observe()

  def _observe(self):
    lower = self.record.welfare
    if self.upper < lower - 1e-7 * max(1., abs(lower)):
      raise ArithmeticError('upper bound below checked feasible welfare')
    if self.upper - lower <= self.gap * max(1., abs(lower)):
      self.stop.set()
    self.trace.append(dict(elapsed_s=monotonic()-self.start, lower=lower,
                           upper=max(self.upper, lower), source=self.record.source))

  def tighten(self, upper):
    with self.lock:
      if math.isfinite(upper) and upper < self.upper:
        self.upper = upper
        self._observe()

  def offer(self, source, y):
    checked = self.p.check(y)
    if not math.isfinite(checked['welfare']):
      raise ValueError('candidate welfare must be finite')
    with self.lock:
      if checked['welfare'] > self.record.welfare:
        self.record = Record(source, y.copy(), checked)
        self._observe()

  def latest(self):
    with self.lock:
      return self.record


def run_highs(p, exchange, deadline, *, tighten_receive=False, rows=()):
  idle = dict(status='stopped before branch-and-cut', nodes=0, submissions=0, rejected=0)
  if exchange.stop.is_set() or monotonic() >= deadline:
    return idle
  solver = fixed_model(p, tighten_receive=tighten_receive)
  apply_rows(solver, rows)
  seed = exchange.latest()
  solution = highspy.HighsSolution()
  solution.col_value = encode_solution(p, seed.y, seed.checked)
  require_ok(solver.setSolution(solution))
  sent = seed
  submitted = 0
  rejected = 0
  errors = []

  def receive(values):
    nonlocal rejected
    try:
      y = integers(values[:len(p.src)], 'HiGHS flow')
      exchange.offer('highs', y)
    except ValueError:
      # Solver tolerances can admit a point that the independent checker rejects.
      rejected += 1

  def on_solution(event):
    receive(event.data_out.mip_solution)

  def on_query(event):
    nonlocal sent, submitted
    record = exchange.latest()
    if record.source == 'vns' and record is not sent:
      require_ok(event.data_in.setSolution(encode_solution(p, record.y, record.checked)))
      sent = record
      submitted += 1

  def on_interrupt(event):
    exchange.tighten(p.base_welfare - event.data_out.mip_dual_bound)
    if exchange.stop.is_set() or monotonic() >= deadline:
      event.interrupt()

  def subscribe(callback, handler):
    # Propagate Python failures after the native solve unwinds.
    def guarded(event):
      try:
        handler(event)
      except Exception as exc:
        errors.append(exc)
        exchange.stop.set()
        event.interrupt()
    callback.subscribe(guarded)

  subscribe(solver.cbMipImprovingSolution, on_solution)
  subscribe(solver.cbMipUserSolution, on_query)
  subscribe(solver.cbMipInterrupt, on_interrupt)
  remaining = deadline - monotonic()
  if remaining <= 0 or exchange.stop.is_set():
    return idle
  require_ok(solver.setOptionValue('time_limit', remaining))
  status = solver.run()
  if errors:
    raise errors[0]
  model_status = solver.getModelStatus()
  allowed = highspy.HighsModelStatus
  if status == highspy.HighsStatus.kError or model_status not in (
      allowed.kOptimal, allowed.kTimeLimit, allowed.kInterrupt, allowed.kHighsInterrupt):
    raise RuntimeError(f'HiGHS: {solver.modelStatusToString(model_status)}')
  info = solver.getInfo()
  final = solver.getSolution()
  if final.value_valid and info.primal_solution_status == highspy.SolutionStatus.kSolutionStatusFeasible:
    receive(final.col_value)
  if info.valid:
    exchange.tighten(p.base_welfare - info.mip_dual_bound)
  return dict(status=solver.modelStatusToString(model_status), nodes=info.mip_node_count,
              submissions=submitted, warm_start_source=seed.source, rejected=rejected)


def solve(p, options=None, initial_y=None, *, relative_gap=1e-4, batch_size=8):
  o = options or Options()
  o.validate()
  if not math.isfinite(relative_gap) or relative_gap < 0:
    raise ValueError('relative_gap must be finite and nonnegative')
  if type(batch_size) is not int or batch_size < 1:
    raise ValueError('batch_size must be a positive integer')
  start = monotonic()
  deadline = start + o.time_limit
  y = np.zeros(len(p.src), dtype=np.int64) if initial_y is None else np.asarray(initial_y)
  exchange = Exchange(p, y, relative_gap, start)
  initial = exchange.latest()
  seen = initial
  iterations = injections = 0
  with ThreadPoolExecutor(max_workers=1) as worker:
    task = worker.submit(run_highs, p, exchange, deadline)
    try:
      if o.iterations and monotonic() < deadline and not exchange.stop.is_set():
        pool = Pool(p, o, initial.y, initial.checked, batch_size)
      while (iterations < o.iterations and monotonic() < deadline and
             not exchange.stop.is_set() and not task.done()):
        record = exchange.latest()
        if record.source == 'highs' and record is not seen:
          pool.inject(record.y, record.checked)
          injections += 1
        seen = record
        count = min(batch_size if pool.cuda else 1, o.iterations - iterations)
        best = pool.advance(iterations, count)
        iterations += count
        if best > exchange.latest().welfare + 1e-12:
          exchange.offer('vns', pool.engine.result())
      highs = task.result()
    finally:
      exchange.stop.set()
  best = exchange.latest()
  upper = max(exchange.upper, best.welfare)
  gap = (upper - best.welfare) / max(1., abs(best.welfare))
  reason = 'highs_finished'
  if gap <= relative_gap:
    reason = 'gap'
  elif monotonic() >= deadline:
    reason = 'time_limit'
  return dict(algorithm='VNS-HiGHS', backend=o.backend, x=p.x.copy(), y=best.y,
              **p.check(best.y), upper_bound=upper, relative_gap=gap,
              termination_reason=reason, initial_welfare=initial.welfare, best_source=best.source,
              iterations_completed=iterations, candidates_evaluated=iterations*o.pool_size*o.candidates,
              highs=highs, highs_to_pool=injections, trace=exchange.trace,
              elapsed_s=monotonic()-start, highs_version=highspy.Highs().version())
