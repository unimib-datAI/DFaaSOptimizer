from concurrent.futures import ThreadPoolExecutor
from functools import partial
from time import monotonic
import math
import highspy
import numpy as np

from centralized_vns_batch_bounds.lp import Relaxation
from centralized_vns_highs.core import Exchange
from centralized_vns_highs.mip import encode_solution, require_ok
from centralized_vns_highs.pool import Pool
from centralized_vns_r.io import integers
from centralized_vns_ram_bounds.formulation import add_ram_rows


class Model:
  def __init__(self, p):
    self.p = p
    self.options, self.paths = {}, {}
    lp = Relaxation(
      p, strengthen=partial(add_ram_rows, options_by_cell=self.options, paths_by_node=self.paths)
    )
    self.upper = lp.certificate(-np.asarray(lp.root.row_dual), lp.lo, lp.hi)
    self.A, self.b = lp.A, lp.b
    self.model = lp.solver.getModel()
    self.cuts = lp.cuts

  def lift(self, record):
    p = self.p
    values = np.zeros(self.model.lp_.num_col_)
    original = encode_solution(p, record.y, record.checked)
    values[: len(original)] = original
    for (j, f), choices in self.options.items():
      if record.checked['z'][j, f] < p.load[j, f] - p.x[j, f]:
        continue
      amount = record.checked['incoming'][j, f]
      col = next(col for _, cap, col in choices if cap >= amount)
      values[col] = 1.0
    for left, layers in self.paths.values():
      for options, transitions in layers:
        cost = next((cost for cost, col in options.items() if values[col]), 0)
        left, arc = transitions[left, cost]
        values[arc] = 1.0
    if np.max(self.A @ values - self.b, initial=0) > 1e-7:
      raise ArithmeticError('invalid full RAM witness')
    return values


def trial(model, y, o, mode):
  p = model.p
  h = highspy.Highs()
  require_ok(h.passModel(model.model))
  for key, val in [
    ('output_flag', False),
    ('threads', 1),
    ('random_seed', o.seed),
    ('mip_rel_gap', 0.0),
    ('mip_abs_gap', 0.0),
  ]:
    require_ok(h.setOptionValue(key, val))
  start = monotonic()
  deadline = start + o.time_limit
  exchange = Exchange(p, y, 0.001, start)
  exchange.tighten(model.upper)
  seed = exchange.latest()
  sol = highspy.HighsSolution()
  sol.col_value = model.lift(seed)
  require_ok(h.setSolution(sol))
  sends = []
  errors = []
  rejected = 0
  last_sent = seed

  def receive(values):
    nonlocal rejected
    try:
      flow = integers(values[: len(p.src)], 'HiGHS flow')
      exchange.offer('highs', flow)
    except ValueError:
      rejected += 1

  def query(event):
    nonlocal last_sent
    record = exchange.latest()
    if record.source == 'vns' and record is not last_sent:
      require_ok(event.data_in.setSolution(model.lift(record)))
      last_sent = record
      sends.append(dict(t=monotonic() - start, value=record.welfare))

  def interrupt(event):
    exchange.tighten(p.base_welfare - event.data_out.mip_dual_bound)
    if monotonic() >= deadline or exchange.stop.is_set():
      event.interrupt()

  def subscribe(callback, fn):
    def guarded(event):
      try:
        fn(event)
      except Exception as exc:
        errors.append(exc)
        exchange.stop.set()
        event.interrupt()

    callback.subscribe(guarded)

  subscribe(h.cbMipImprovingSolution, lambda e: receive(e.data_out.mip_solution))
  subscribe(h.cbMipUserSolution, query)
  subscribe(h.cbMipInterrupt, interrupt)
  require_ok(h.setOptionValue('time_limit', max(0.0, deadline - monotonic())))
  it = highs_to_gpu = elite_to_gpu = 0
  pool = None
  with ThreadPoolExecutor(max_workers=1) as worker:
    future = worker.submit(h.run)
    try:
      if mode != 'highs':
        pool = Pool(p, o, seed.y, seed.checked, 1)
        if mode == 'coordinated':
          from .moves import install

          install(pool.engine)
      seen = seed
      while (
        pool is not None
        and monotonic() < deadline
        and not exchange.stop.is_set()
        and not future.done()
      ):
        current = exchange.latest()
        if current is not seen:
          # Recenter one stale trajectory. The rest keep exploring their own paths.
          pool.inject(current.y, current.checked)
          if current.source == 'highs':
            highs_to_gpu += 1
          else:
            elite_to_gpu += 1
          seen = current
        best = pool.advance(it, 1)
        it += 1
        if best > exchange.latest().welfare + 1e-10 and monotonic() <= deadline:
          exchange.offer('vns', pool.engine.result())
      status = future.result()
    finally:
      exchange.stop.set()
  if errors:
    raise errors[0]
  if status == highspy.HighsStatus.kError:
    raise RuntimeError('HiGHS failed')
  state = h.getModelStatus()
  if state not in (
    highspy.HighsModelStatus.kOptimal,
    highspy.HighsModelStatus.kTimeLimit,
    highspy.HighsModelStatus.kInterrupt,
    highspy.HighsModelStatus.kHighsInterrupt,
  ):
    raise RuntimeError(h.modelStatusToString(state))
  info = h.getInfo()
  if info.valid:
    if info.primal_solution_status == highspy.SolutionStatus.kSolutionStatusFeasible:
      receive(h.getSolution().col_value)
    exchange.tighten(p.base_welfare - info.mip_dual_bound)
  record = exchange.latest()
  upper = max(exchange.upper, record.welfare)
  return record, dict(
    mode=mode,
    seconds=monotonic() - start,
    start=seed.welfare,
    lower=record.welfare,
    upper=upper,
    gap=(upper - record.welfare) / max(1, abs(record.welfare)),
    trace=exchange.trace,
    solver_lower=(
      p.base_welfare - info.objective_function_value
      if info.valid
      and info.primal_solution_status == highspy.SolutionStatus.kSolutionStatusFeasible
      else None
    ),
    source=record.source,
    solver_status=h.modelStatusToString(state),
    nodes=info.mip_node_count,
    submissions=sends,
    highs_to_gpu=highs_to_gpu,
    elite_to_gpu=elite_to_gpu,
    rejected=rejected,
    coordinated_accepted=getattr(pool.engine, 'evictions', 0) if pool else 0,
    double_evictions=getattr(pool.engine, 'double_evictions', 0) if pool else 0,
    iterations=it,
    candidate_slots=it * o.pool_size * o.candidates,
    device=pool.engine.mx.device_info()['device_name'] if pool else None,
  )
