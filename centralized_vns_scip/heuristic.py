"""Search the current SCIP flow box and submit fully lifted solutions."""

from time import monotonic

import numpy as np
from pyscipopt import Heur, SCIP_LPSOLSTAT, SCIP_RESULT

from centralized_vns_highs.core import Record


class NodeSearch(Heur):
  def __init__(
    self, shared, variables, seed, deadline, stats, solution_factory, allow_global=False
  ):
    self.shared, self.variables = shared, variables
    self.seed, self.deadline, self.stats = seed, deadline, stats
    self.solution_factory = solution_factory
    self.allow_global = allow_global
    self.transformed = None
    self.pool = None
    self.iteration = 0
    cols = shared.model.lp_
    self.lower = np.asarray(cols.col_lower_)[: len(seed.y)]
    self.upper = np.asarray(cols.col_upper_)[: len(seed.y)]
    self.stats.update(
      calls=0,
      batches=0,
      nonroot_batches=0,
      no_seed=0,
      aux_rejected=0,
      submissions=[],
      slots=0,
      lp_seeds=0,
      node_samples=[],
    )

  def heurexec(self, heurtiming, nodeinfeasible):
    m, stats, p = self.model, self.stats, self.shared.p
    stats["calls"] += 1
    if nodeinfeasible or monotonic() >= self.deadline or m.getLPSolstat() != SCIP_LPSOLSTAT.OPTIMAL:
      return dict(result=SCIP_RESULT.DIDNOTRUN)
    if self.transformed is None:
      self.transformed = [m.getTransformedVar(v) for v in self.variables]
    flows = self.transformed[: len(p.src)]
    lo = np.ceil(np.maximum(self.lower, [v.getLbLocal() for v in flows]) - 1e-7).astype(np.int64)
    hi = np.floor(np.minimum(self.upper, [v.getUbLocal() for v in flows]) + 1e-7).astype(np.int64)
    lp = np.array([v.getLPSol() for v in flows])
    best = m.getBestSol()
    inc = np.rint([m.getSolVal(best, v) for v in flows]).astype(np.int64)
    seeds = []
    clipped = np.clip(inc, lo, hi)
    try:
      seeds.append(("incumbent", clipped, p.check(clipped)))
    except ValueError:
      pass
    # Start at mandatory branch flows, then round the LP down within residual capacities.
    try:
      state = p.check(lo)
    except ValueError:
      state = None
    if state is not None:
      y = lo.copy()
      incoming, rejected, replicas = state["incoming"], state["z"], state["replicas"]
      free = p.ram - replicas @ p.memory
      residual = p.load - p.x
      target = np.clip(np.floor(lp + 1e-7).astype(np.int64), lo, hi)
      for edge in np.flatnonzero(target > lo)[np.argsort(-p.reward[target > lo], kind="stable")]:
        i, j, f = p.src[edge], p.dst[edge], p.fun[edge]
        if incoming[i, f] or rejected[j, f] != residual[j, f]:
          continue
        work = p.x[j, f] + incoming[j, f]
        maxrep = replicas[j, f] + free[j] // p.memory[f]
        q = min(
          target[edge] - y[edge],
          rejected[i, f],
          int(np.floor((maxrep + 1e-9) * p.capacity[j, f])) - work if maxrep else 0,
        )
        if q <= 0:
          continue
        rep = max(1, int(np.ceil((work + q) / p.capacity[j, f] - 1e-9)))
        if rep > maxrep:
          continue
        free[j] -= (rep - replicas[j, f]) * p.memory[f]
        replicas[j, f] = rep
        incoming[j, f] += q
        rejected[i, f] -= q
        y[edge] += q
      seeds.append(("lp", y, p.check(y)))
    if not seeds:
      stats["no_seed"] += 1
      return dict(result=SCIP_RESULT.DIDNOTFIND)
    e = self.engine(lo, hi)
    seeds.sort(key=lambda item: item[2]["welfare"], reverse=True)
    self.seed_slots(slice(None), *seeds[0][1:])
    if len(seeds) > 1 and seeds[1][0] == "lp":
      self.seed_slots(slice(1, None, 4), *seeds[1][1:])
    stats["lp_seeds"] += int(any(s[0] == "lp" for s in seeds[:2]))
    stats["batches"] += 1
    stats["nonroot_batches"] += int(m.getDepth() > 0)
    if len(stats["node_samples"]) < 12:
      stats["node_samples"].append(
        dict(
          node=m.getNNodes(),
          depth=m.getDepth(),
          fixed=int(np.count_nonzero(lo == hi)),
          seed=seeds[0][0],
          seed_value=seeds[0][2]["welfare"],
        )
      )
      print("GPU node", stats["node_samples"][-1], flush=True)
    until = min(self.deadline, monotonic() + (2.0 if m.getDepth() == 0 else 0.10))
    result = SCIP_RESULT.DIDNOTFIND
    attempted = m.getPrimalbound()
    while monotonic() < until:
      value = self.pool.advance(self.iteration, 1)
      self.iteration += 1
      stats["slots"] += e.o.pool_size * e.o.candidates
      if value <= attempted + 1e-7:
        continue
      y = e.result()
      assert np.all((lo <= y) & (y <= hi))
      record = Record("gpu", y, p.check(y))
      np.testing.assert_allclose(value, record.welfare, rtol=0, atol=1e-7)
      attempted = value
      vals = self.shared.lift(record)
      local = all(
        v.getLbLocal() - 1e-6 <= x <= v.getUbLocal() + 1e-6 for x, v in zip(vals, self.transformed)
      )
      if not local and not self.allow_global:
        stats["aux_rejected"] += 1
        continue
      before = m.getPrimalbound()
      stored = m.trySol(self.solution_factory(m, self.variables, vals, self), completely=True)
      after = m.getPrimalbound()
      stats["submissions"].append(
        dict(
          t=m.getSolvingTime(),
          value=value,
          before=before,
          after=after,
          stored=stored,
          depth=m.getDepth(),
          local=local,
        )
      )
      print("GPU offer", stats["submissions"][-1], flush=True)
      if stored:
        result = SCIP_RESULT.FOUNDSOL
    return dict(result=result)
