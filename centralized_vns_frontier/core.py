"""Fixed-x VNS over disjoint regions with asynchronous HiGHS certificates."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import math
from time import monotonic

import numpy as np

from centralized_vns.core import Options
from centralized_vns_highs.core import Exchange
from .pool import Pool
from .proof import Ledger, prove
from .regions import partition
from .relink import cross_regions, relink


def redistribute(pool, book):
  with book.lock:
    active = np.flatnonzero(~book.closed)
    if not len(active):
      return
    quality = [book.elite[b][0].welfare if book.elite[b] else -np.inf for b in active]
  # Guarantee coverage, give spare trajectories to the best half of the live regions.
  ranked = active[np.argsort(-np.asarray(quality), kind='stable')]
  extra = len(pool.branches) - len(active)
  target = np.bincount(
    np.r_[active, np.resize(ranked[: (len(active) + 1) // 2], extra)], minlength=book.regions.count
  )
  counts = np.bincount(pool.branches, minlength=book.regions.count)
  deficits = iter(np.repeat(np.arange(len(counts)), np.maximum(0, target - counts)))
  changed = False
  for slot in range(len(pool.branches) - 1, -1, -1):
    old = pool.branches[slot]
    if counts[old] <= target[old]:
      continue
    new = next(deficits)
    r = book.witness(new)
    pool.seed(slot, r.y, r.checked)
    pool.branches[slot] = new
    counts[old] -= 1
    counts[new] += 1
    changed = True
  if changed:
    pool.assign()


def solve(
  problem,
  options=None,
  initial_y=None,
  *,
  branches=256,
  relative_gap=1e-3,
  slice_seconds=3.0,
  exchange_every=8,
  global_share=0.20,
  final_relink_top=0,
  tighten_receive=False,
  pair_cuts=False,
  group_cuts=False,
  lagrangian=False,
):
  o = options or Options()
  o.validate()
  if type(final_relink_top) is not int or final_relink_top < 0 or final_relink_top == 1:
    raise ValueError('final_relink_top must be zero or at least two')
  if type(branches) is not int or branches < 1 or branches > 2**30:
    raise ValueError('invalid maximum number of branches')
  if (
    type(exchange_every) is not int
    or exchange_every < 1
    or not math.isfinite(slice_seconds)
    or slice_seconds <= 0
  ):
    raise ValueError('invalid exchange interval or solver time slice')
  if not math.isfinite(relative_gap) or not 0 <= relative_gap <= 1:
    raise ValueError('relative_gap must be in [0,1]')
  if not math.isfinite(global_share) or not 0 <= global_share <= 1:
    raise ValueError('global_share must be in [0,1]')
  start = monotonic()
  end = start + o.time_limit
  # Stop on the coordinator gap, removing the constant local service gain.
  p = replace(problem, alpha=np.zeros_like(problem.alpha))
  offset = float(np.sum(problem.alpha * problem.x / np.maximum(1, problem.load)))
  y = np.zeros(len(p.src), np.int64) if initial_y is None else np.asarray(initial_y)
  initial = p.check(y)['welfare']
  y = y.astype(np.int64)
  fallback = Exchange(p, y, relative_gap, start).upper
  pair_rows = [] if pair_cuts else None
  group_rows = [] if group_cuts else None
  prices = [] if lagrangian else None
  regions, lp_bound = partition(
    p,
    min(branches, o.pool_size),
    end,
    tighten_receive=tighten_receive,
    pair_rows=pair_rows,
    group_rows=group_rows,
    dual=prices,
  )
  book = Ledger(p, regions, y, min(fallback, lp_bound), relative_gap, start)
  iterations = pairs = novel = 0
  top_source = None
  prior_pairs = {}
  pool = None
  with ThreadPoolExecutor(max_workers=1) as worker:
    proof = worker.submit(
      prove,
      book,
      end,
      slice_seconds,
      global_share,
      tighten_receive=tighten_receive,
      pair_rows=pair_rows,
      group_rows=group_rows,
      prices=prices,
      device=o.backend,
      max_bytes=o.max_device_bytes,
    )
    try:
      if o.iterations and not book.stop.is_set() and monotonic() < end:
        pool = Pool(p, o, regions)
        for slot, b in enumerate(pool.branches):
          r = book.witness(int(b))
          pool.seed(slot, r.y, r.checked)
      while iterations < o.iterations and not book.stop.is_set() and monotonic() < end:
        if proof.done():
          proof.result()  # Surface worker errors while heuristic work is still in flight.
        values = pool.engine.step(iterations)
        iterations += 1
        lead = int(np.argmax(values))
        if values[lead] > book.best.welfare + 1e-10:
          book.offer('vns', pool.record(lead), int(pool.branches[lead]))
        if iterations % exchange_every:
          continue
        # Include archive leaders and rotate through the population even if the leader is stable.
        size = min(8, o.pool_size)
        cycle = (iterations // exchange_every - 1) * size
        slots = np.unique(
          np.r_[np.argsort(-values, kind='stable')[:size], (cycle + np.arange(size)) % o.pool_size]
        )
        for slot in slots:
          b = int(pool.branches[slot])
          book.offer('vns', pool.record(int(slot)), b)
        jobs = []
        with book.lock:
          for b in np.unique(pool.branches[slots]):
            archive = book.elite[b]
            if book.closed[b] or len(archive) < 2:
              continue
            a = archive[0]
            g = max(archive[1:], key=lambda r: np.count_nonzero(a.y != r.y))
            previous = prior_pairs.get(b)
            if previous is None or previous[0] is not a or previous[1] is not g:
              jobs.append((int(b), a, g))
              prior_pairs[b] = (a, g)
        if jobs:
          if o.backend == 'cuda':
            children = pool.engine.relink(
              np.stack([a.y for _, a, _ in jobs]), np.stack([g.y for _, _, g in jobs])
            )
          else:
            children = [relink(p, a.y, g.y) for _, a, g in jobs]
          for (b, a, g), child in zip(jobs, children):
            r = book.offer('path_relinking', child, b)
            pairs += 1
            if r.welfare > max(a.welfare, g.welfare) + 1e-10:
              novel += 1
              same = np.flatnonzero(pool.branches == b)
              pool.seed(int(same[np.argmin(values[same])]), child, r.checked)
        record = book.best
        if record.source in ('highs', 'branch_highs') and record is not top_source:
          b = int(regions.identify(record.y))
          same = np.flatnonzero(pool.branches == b)
          if len(same):
            pool.seed(int(same[np.argmin(values[same])]), record.y, record.checked)
          # Share the good global pattern with a different region, preserving its constraints.
          other = int(pool.branches[cycle % o.pool_size])
          if other != b and not book.closed[other]:
            projected = regions.seed(p, other, record.y)
            r = book.offer('projected_seed', projected, other)
            same = np.flatnonzero(pool.branches == other)
            weakest = int(same[np.argmin(values[same])])
            if r.welfare > values[weakest]:
              pool.seed(weakest, r.y, r.checked)
          top_source = record
        redistribute(pool, book)
      root, local = proof.result()
    finally:
      book.stop.set()
  final_fields = {}
  if final_relink_top:
    stage_start = monotonic()
    search_record = book.best
    upper_before = float(book.upper.max())
    previous_closures = len(book.closures)
    if pool is not None:
      scores = pool.engine.best.copy_to_host() if o.backend == 'cuda' else pool.engine.best
      for b in np.unique(pool.branches):
        slots = np.flatnonzero(pool.branches == b)
        slot = int(slots[np.argmax(scores[slots])])
        if not book.elite[b] or scores[slot] > book.elite[b][0].welfare + 1e-10:
          book.offer('final_pool', pool.record(slot), int(b))
    pre = book.best
    closures_before_pairs = len(book.closures)
    gpu_path = pool.engine.relink if pool is not None and o.backend == 'cuda' else None
    stage, parents = cross_regions(book, final_relink_top, gpu_path)
    stage.update(
      requested_top=final_relink_top,
      search_seconds=stage_start - start,
      seconds=monotonic() - stage_start,
      search_record=search_record.welfare,
      pre_objective=pre.welfare,
      post_objective=book.best.welfare,
      pre_gap=(max(upper_before, pre.welfare) - pre.welfare) / max(1.0, abs(pre.welfare)),
      upper_before=upper_before,
      pool_closures=closures_before_pairs - previous_closures,
      relink_closures=len(book.closures) - closures_before_pairs,
    )
    final_fields = dict(
      final_relink=stage, final_parent_y=parents, pre_final_y=pre.y, search_y=search_record.y
    )
  best = book.best
  checked = problem.check(best.y)
  upper = max(best.welfare, float(book.upper.max()))
  gap = (upper - best.welfare) / max(1.0, abs(best.welfare))
  return dict(
    algorithm='VNS-Frontier-PR',
    backend=o.backend,
    x=problem.x.copy(),
    y=best.y,
    **checked,
    upper_bound=upper + offset,
    coordinator_objective=best.welfare,
    coordinator_upper=upper,
    coordinator_gap=gap,
    welfare_offset=offset,
    initial_coordinator=initial,
    best_source=best.source,
    termination_reason=(
      'gap' if gap <= relative_gap else 'time_limit' if monotonic() >= end else 'proof_finished'
    ),
    elapsed_s=monotonic() - start,
    iterations_completed=iterations,
    candidates_evaluated=iterations * o.pool_size * o.candidates,
    branches=regions.count,
    split_edges=regions.edges.tolist(),
    split_thresholds=regions.cuts.tolist(),
    branch_upper=book.upper.tolist(),
    closures=book.closures,
    global_highs=root,
    global_share=global_share,
    tighten_receive=tighten_receive,
    pair_cuts=pair_cuts,
    root_pair_cuts=len(pair_rows or []),
    group_cuts=group_cuts,
    root_group_cuts=len(group_rows or []),
    lagrangian=lagrangian,
    coordinator_root_lp=lp_bound if math.isfinite(lp_bound) else None,
    branch_solves=local,
    pairs_relinked=pairs,
    children_better_than_both=novel,
    coordinator_trace=book.trace,
    **final_fields,
  )
