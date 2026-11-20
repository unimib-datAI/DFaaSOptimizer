"""Fixed-x hierarchical VNS with one trajectory per active region."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import math
from time import monotonic
import numpy as np

from centralized_vns.core import Options
from centralized_vns_highs.core import Exchange
from centralized_vns_frontier.proof import Ledger
from centralized_vns_frontier.regions import Regions, partition
from centralized_vns_frontier.relink import cross_regions, relink
from .frontier import Tree, prove
from .moves import TreePool


def solve(
  problem,
  options=None,
  initial_y=None,
  *,
  depth=20,
  refine_every=256,
  relative_gap=0.001,
  final_relink_top=32,
):
  o = options or Options()
  o.validate()
  if o.backend not in ('cpu', 'cuda'):
    raise ValueError('Tree supports cpu and cuda backends')
  if (
    type(depth) is not int
    or not 0 <= depth <= 30
    or type(refine_every) is not int
    or refine_every < 1
  ):
    raise ValueError('depth must be in [0,30] and refine_every must be positive')
  if not math.isfinite(relative_gap) or not 0 <= relative_gap <= 1 or final_relink_top < 0:
    raise ValueError('invalid gap or relinking limit')
  start = monotonic()
  deadline = start + o.time_limit
  p = replace(problem, alpha=np.zeros_like(problem.alpha))
  y = np.zeros(len(p.src), np.int64) if initial_y is None else np.asarray(initial_y)
  initial = p.check(y)['welfare']
  y = y.astype(np.int64)
  rows = []
  full, lp = partition(p, 1 << depth, deadline, tighten_receive=True, pair_rows=rows)
  active_depth = min(len(full.edges), o.pool_size.bit_length() - 1)
  coarse = Regions(full.edges[:active_depth], full.cuts[:active_depth])
  size = coarse.count
  upper = min(lp, Exchange(p, y, relative_gap, start).upper)
  book = Ledger(p, coarse, y, upper, relative_gap, start)
  tree = Tree(full, book)
  pool = None
  iterations = cursor = pairs = 0
  last_pairs = {}
  injected = None

  def seed(slot, key, parent):
    d, code = key
    y = tree.prefix[d].seed(p, code, parent)
    pool.seed(slot, y, p.check(y))
    pool.branches[slot], pool.depths[slot] = code, d

  with ThreadPoolExecutor(max_workers=1) as executor:
    future = executor.submit(prove, tree, deadline, rows)
    try:
      if o.iterations and not book.stop.is_set() and monotonic() < deadline:
        pool = TreePool(p, replace(o, pool_size=size), full, active_depth)
        for b, key in enumerate(tree.active):
          seed(b, key, book.witness(b).y)
        pool.assign()
      while (
        pool is not None
        and iterations < o.iterations
        and monotonic() < deadline
        and not book.stop.is_set()
      ):
        if future.done():
          future.result()
        scores = pool.step(iterations)
        iterations += 1
        lead = int(np.argmax(scores))
        if scores[lead] > book.best.welfare + 1e-10:
          book.offer('vns', pool.record(lead), lead)
        if iterations % 8 == 0:
          k = min(8, size)
          slots = np.unique(
            np.r_[
              np.argsort(-scores, kind='stable')[:k],
              ((iterations // 8 - 1) * k + np.arange(k)) % size,
            ]
          )
          jobs = []
          for b in slots:
            key = tree.active[b]
            if key is None:
              continue
            d, code = key
            record = pool.record(int(b))
            if tree.prefix[d].identify(record) != code:
              raise ArithmeticError('record crossed a prefix')
            book.offer('vns', record, int(b))
            with book.lock:
              elite = [r for r in book.elite[b] if tree.prefix[d].identify(r.y) == code]
            if len(elite) > 1:
              a = elite[0]
              g = max(elite[1:], key=lambda r: np.count_nonzero(a.y != r.y))
              old = last_pairs.get(b)
              if old is None or old[0] is not a or old[1] is not g:
                jobs.append((b, a, g))
                last_pairs[b] = (a, g)
          if jobs:
            if o.backend == 'cuda':
              children = pool.engine.relink(
                np.stack([a.y for _, a, _ in jobs]), np.stack([g.y for _, _, g in jobs])
              )
            else:
              children = [relink(p, a.y, g.y) for _, a, g in jobs]
            for (b, a, g), child in zip(jobs, children):
              d, code = tree.active[b]
              if tree.prefix[d].identify(child) != code:
                raise ArithmeticError('path crossed a prefix')
              r = book.offer('path_relinking', child, int(b))
              if r.welfare > max(scores[b], a.welfare, g.welfare) + 1e-10:
                pool.seed(b, child, r.checked)
              pairs += 1
          record = book.latest()
          if record.source == 'highs' and record is not injected:
            b = int(coarse.identify(record.y))
            key = tree.active[b]
            if key is not None:
              d, code = key
              projected = tree.prefix[d].seed(p, code, record.y)
              r = book.offer('projected_seed', projected, b)
              if r.welfare > scores[b]:
                pool.seed(b, projected, r.checked)
            injected = record
        if iterations % refine_every == 0:
          k = min(8, size)
          selected = np.unique(
            np.r_[np.argsort(-scores, kind='stable')[:k], (cursor + np.arange(k)) % size]
          )
          cursor = (cursor + k) % size
          for b in selected:
            old = tree.active[b]
            if old is None:
              continue
            record = pool.record(int(b))
            book.offer('vns', record, int(b))
            tree.advance(int(b), record)
            key = tree.active[b]
            if key != old:
              if key is None:
                pool.depths[b] = -1
              else:
                seed(
                  b,
                  key,
                  record
                  if tree.prefix[key[0]].identify(record) == key[1]
                  else book.witness(int(b)).y,
                )
          pool.assign()
      root, regional = future.result()
    finally:
      book.stop.set()
  search = monotonic() - start
  if pool is not None:
    for b in range(size):
      book.offer('final_pool', pool.record(b), b)
  rounds = []
  gpu_path = pool.engine.relink if pool is not None and o.backend == 'cuda' else None
  if final_relink_top >= 2:
    for _ in range(4):
      stage, _ = cross_regions(book, final_relink_top, gpu_path)
      rounds.append(dict(value=book.best.welfare, pairs=stage['pairs'], backend=stage['backend']))
  lower = book.best.welfare
  upper = max(lower, float(book.upper.max()))
  leaves = [
    dict(depth=d, code=c, upper=float(min(cap, book.upper[b])))
    for b, entries in enumerate(tree.bounds)
    for (d, c), cap in entries.items()
  ]
  checked = problem.check(book.best.y)
  offset = float(np.sum(problem.alpha * problem.x / np.maximum(1, problem.load)))
  gap = (upper - lower) / max(1.0, abs(lower))
  return dict(
    algorithm='VNS-Tree',
    backend=o.backend,
    x=p.x.copy(),
    y=book.best.y,
    **checked,
    upper_bound=upper + offset,
    welfare_offset=offset,
    best_source=book.best.source,
    gap_target=relative_gap,
    termination_reason='gap'
    if gap <= relative_gap
    else 'time_limit'
    if monotonic() >= deadline
    else 'proof_finished',
    initial_coordinator=initial,
    coordinator_objective=lower,
    coordinator_upper=upper,
    coordinator_gap=gap,
    coordinator_root_lp=lp if math.isfinite(lp) else None,
    potential_regions=full.count,
    active_limit=size,
    materialized_leaves=len(leaves),
    active_final=sum(k is not None for k in tree.active),
    splits=tree.splits,
    switches=tree.switches,
    max_depth=max(l['depth'] for l in leaves),
    leaves=leaves,
    split_edges=full.edges.tolist(),
    split_thresholds=full.cuts.tolist(),
    active_depth_initial=active_depth,
    global_highs=root,
    regional_proofs=regional,
    root_pair_cuts=len(rows),
    closed_leaves=sum(l['upper'] <= tree.threshold() for l in leaves),
    iterations_completed=iterations,
    candidates_evaluated=iterations * size * o.candidates,
    search_seconds=search,
    elapsed_s=monotonic() - start,
    internal_pairs=pairs,
    rounds=rounds,
  )
