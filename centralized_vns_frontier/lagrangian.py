"""Endpoint Lagrangian upper bounds on fixed-x forwarding, in integer reward units."""

import math
from time import monotonic

import numpy as np

from centralized_vns_highs.mip import ram_receive_limit


def price_profiles(p, row_dual, cuts):
  """Two feasible price seeds from the sending and receiving sides of the root LP."""
  n = p.x.size
  dual = -np.asarray(row_dual)
  send = dual[:n].reshape(p.x.shape)[p.src, p.fun]
  receive = dual[n : 4 * n].reshape(3, *p.x.shape).sum(0)[p.dst, p.fun]
  for (edges, _), price in zip(cuts, dual[4 * n + len(p.x) :], strict=True):
    receive[edges] += price
  return [np.clip(v, 0, p.reward) for v in (p.reward - send, receive)]


class EndpointBound:
  """Evaluate the root and every disjoint split by recomputing only affected endpoints."""

  def __init__(self, p, regions, device='cpu', max_bytes=512 * 1024**2):
    self.p, self.regions = p, regions
    self.xp = np
    self.tensor = np.asarray
    self.host = np.asarray
    if device == 'cuda':
      import cupy

      self.xp = cupy
      self.tensor = cupy.asarray
      self.host = cupy.asnumpy
    n, functions = p.x.shape
    e = len(p.src)
    upper = (p.load - p.x)[p.src, p.fun].copy()
    baseline = p.check(np.zeros(e, dtype=np.int64))['replicas']
    supply = np.zeros_like(p.x)
    np.add.at(supply, (p.dst, p.fun), upper)
    cap = ram_receive_limit(p, supply.ravel(), baseline).reshape(p.x.shape)
    upper = np.minimum(upper, cap[p.dst, p.fun])
    # Left/right corrections add exactly because split edges share no endpoints.
    ends = np.column_stack((p.src[regions.edges], p.dst[regions.edges])).astype(np.int64)
    if np.unique(ends).size != ends.size:
      raise ValueError('endpoint decomposition expects node-disjoint splits')
    nodes = np.r_[np.arange(n), np.repeat(ends, 2, axis=0).ravel()]
    self.nodes = nodes
    unit = int(np.gcd.reduce(p.memory))
    self.weights = p.memory // unit
    budget = (p.ram - baseline @ p.memory) // unit
    self.width = int(budget.max()) + 1
    counts = [np.bincount(v * functions + p.fun, minlength=p.x.size) for v in (p.src, p.dst)]
    degree = max(1, *(int(c.max()) for c in counts))
    # Allow room for DP choices, input incidence, gathers and scratch tensors.
    if (
      8 * len(nodes) * (self.width * (functions + 12 + 6 * degree) + 12 * functions * degree)
      + 8 * (6 * regions.count + 20 * e)
      > max_bytes
    ):
      raise MemoryError('endpoint tables exceed the configured device memory budget')
    self.budget = self.tensor(budget[nodes])
    self.row = self.tensor(np.arange(len(nodes)))
    self.residual = self.tensor((p.load - p.x)[nodes])
    self.sides = []
    high = np.r_[upper, 0]
    for endpoint in (p.src, p.dst):
      index = np.full((n, functions, degree), e, dtype=np.int64)
      pos = np.zeros_like(p.x)
      for edge, (node, f) in enumerate(zip(endpoint, p.fun)):
        index[node, f, pos[node, f]] = edge
        pos[node, f] += 1
      index = index[nodes]
      hi = high[index]
      lo = np.zeros_like(hi)
      for k, (edge, cut) in enumerate(zip(regions.edges, regions.cuts)):
        block = slice(n + 4 * k, n + 4 * k + 2)
        hi[block] = np.where(index[block] == edge, np.minimum(hi[block], cut), hi[block])
        block = slice(n + 4 * k + 2, n + 4 * k + 4)
        lo[block] = np.where(index[block] == edge, cut + 1, 0)
      if np.any(lo > hi):
        raise ValueError('region excludes its own feasible seed')
      self.sides.append(tuple(self.tensor(v) for v in (index, lo, hi)))
    self.capacity = []
    for f, weight in enumerate(self.weights):
      # Beyond saturation, extra replicas cannot improve any receiving choice.
      with np.errstate(over='ignore'):
        required = np.maximum(
          p.x[:, f] + cap[:, f] > 0, np.ceil((p.x[:, f] + cap[:, f]) / p.capacity[:, f] - 1e-9)
        ).astype(np.int64)
      last = int(np.minimum(budget // weight, required - baseline[:, f]).max())
      replicas = baseline[:, f, None] + np.arange(last + 1)
      low = np.zeros_like(replicas)
      high = np.broadcast_to(cap[:, f, None], replicas.shape).copy()
      while np.any(low < high):
        mid = (low + high + 1) // 2
        processing = p.x[:, f, None] + mid
        with np.errstate(over='ignore'):
          fits = (
            np.maximum(processing > 0, np.ceil(processing / p.capacity[:, f, None] - 1e-9))
            <= replicas
          )
        low = np.where(fits, mid, low)
        high = np.where(fits, high, mid - 1)
      self.capacity.append(self.tensor(low[nodes]))
    # Round rewards up on a binary grid. Reserve headroom for both endpoint copies.
    self.scale = 2**24
    while True:
      scaled = p.reward * self.scale
      if np.all(np.isfinite(scaled)) and np.max(scaled, initial=0) < 2**60:
        rounded = np.ceil(scaled).astype(np.int64)
        magnitude = sum(int(a) * int(b) for a, b in zip(rounded, upper))
        if magnitude < 2**59:
          break
      if self.scale == 1:
        raise OverflowError('endpoint reward range exceeds safe integer arithmetic')
      self.scale //= 2
    self.reward = self.tensor(p.reward)
    self.rounded = self.tensor(rounded)
    self.zero = self.tensor(np.zeros(1, dtype=np.int64))
    local = (np.abs(p.alpha * p.x) + p.gamma * (p.load - p.x)) / np.maximum(1, p.load)
    self.margin = 64 * np.finfo(float).eps * (1 + local.sum() + magnitude / self.scale)

  def evaluate(self, price, deadline=math.inf):
    p, xp = self.p, self.xp
    price = self.tensor(price)
    if price.shape != self.reward.shape or not bool(xp.isfinite(price).all()):
      raise ValueError('expected one finite price per edge')
    price = xp.round(xp.minimum(price.clip(0, None), self.reward) * self.scale)
    price = price.astype(np.int64)
    n, functions = p.x.shape
    side_prices = [xp.concatenate((v, self.zero)) for v in (self.rounded - price, price)]
    dp = self.tensor(np.zeros((len(self.nodes), self.width), dtype=np.int64))
    decisions, sorted_edges = [], []
    impossible = -(2**61)
    for f in range(functions):
      if monotonic() >= deadline:
        return None
      offers = []
      for side, ((index, lo, hi), weights) in enumerate(zip(self.sides, side_prices)):
        edges = index[:, f, :]
        order = xp.argsort(-weights[edges], 1)
        ix = (self.row[:, None], order)
        edges, lo_f, hi_f = edges[ix], lo[:, f, :][ix], hi[:, f, :][ix]
        gains = weights[edges]
        # Skip zero-price optional flow to select a sparse dual subgradient.
        room = (hi_f - lo_f) * (gains > 0)
        prefix = room.cumsum(1) - room
        mandatory = lo_f.sum(1)
        limits = self.residual[:, f, None] if side == 0 else self.capacity[f]
        remaining = limits - mandatory[:, None]
        take = xp.minimum(
          room[:, None, :], (remaining[:, :, None] - prefix[:, None, :]).clip(0, None)
        )
        value = (lo_f * gains).sum(1)[:, None] + (take * gains[:, None, :]).sum(2)
        offers.append((value, mandatory, remaining))
        sorted_edges.append((edges[:n], lo_f[:n], room[:n], prefix[:n]))
      send = xp.where(
        (offers[1][1] == 0) & (offers[0][2][:, 0] >= 0), offers[0][0][:, 0], impossible
      )
      recv = xp.where((offers[0][1][:, None] == 0) & (offers[1][2] >= 0), offers[1][0], impossible)
      idle = xp.maximum(send, recv[:, 0])
      # Unreachable RAM budgets remain unreachable across every stage.
      nxt = xp.where(
        (dp == impossible) | (idle[:, None] == impossible), impossible, dp + idle[:, None]
      )
      choice = xp.zeros_like(dp[:n])
      for k in range(1, recv.shape[1]):
        shift = int(k * self.weights[f])
        prev, value = dp[:, :-shift], recv[:, k, None]
        candidate = xp.where((prev == impossible) | (value == impossible), impossible, prev + value)
        choice[:, shift:] = xp.where(candidate[:n] > nxt[:n, shift:], k, choice[:, shift:])
        nxt[:, shift:] = xp.maximum(nxt[:, shift:], candidate)
      decisions.append((choice, (send >= recv[:, 0])[:n]))
      dp = nxt
    score = dp[self.row, self.budget]
    if bool((score == impossible).any()):
      raise ArithmeticError('endpoint DP lost a feasible region')
    base = score[:n].sum()
    edges = self.regions.edges
    delta = (
      score[n:].reshape(-1, 2, 2).sum(2)
      - (score[self.tensor(p.src[edges])] + score[self.tensor(p.dst[edges])])[:, None]
    )
    totals = self.tensor(np.zeros(self.regions.count, dtype=np.int64)) + base
    ids = self.tensor(np.arange(self.regions.count))
    for k, bit in enumerate(self.regions.bits):
      totals += delta[k, (ids // int(bit)) % 2]
    # Subgradients need only the decisions of the unrestricted (first n) nodes.
    available = self.budget[:n]
    copies = [xp.zeros_like(v) for v in side_prices]
    for f in reversed(range(functions)):
      choice, prefer_send = decisions[f]
      k = choice[self.row[:n], available]
      sending = (k == 0) & prefer_send
      available = available - k * int(self.weights[f])
      for side in (0, 1):
        edge, lo, room, prefix = sorted_edges[2 * f + side]
        limit = self.residual[:n, f] if side == 0 else self.capacity[f][self.row[:n], k]
        take = lo + xp.minimum(room, (limit[:, None] - lo.sum(1)[:, None] - prefix).clip(0, None))
        flow = xp.where((sending if side == 0 else ~sending)[:, None], take, 0)
        # Each edge occurs once on each side. Repeated padding always writes zero.
        copies[side][edge.ravel()] = flow.ravel()
    # The integer DP is exact; cover roundoff in the independent float64 welfare.
    bound = np.nextafter(p.base_welfare + self.host(totals) / self.scale + self.margin, np.inf)
    return (
      bound,
      p.base_welfare + int(base) / self.scale + self.margin,
      (copies[1] - copies[0])[:-1],
    )


def certify_regions(book, profiles, deadline, device, max_bytes=512 * 1024**2):
  start = monotonic()
  report = dict(evaluations=0, device='cuda' if device == 'cuda' else 'cpu')
  if not profiles or start >= deadline:
    return dict(**report, seconds=0, reason='no root prices or time')
  allocation_errors = (MemoryError, OverflowError)
  if device == 'cuda':
    import cupy

    allocation_errors += (cupy.cuda.memory.OutOfMemoryError,)
    max_bytes = min(max_bytes, cupy.cuda.runtime.memGetInfo()[0])
  try:
    oracle = EndpointBound(book.p, book.regions, device, max_bytes)
  except allocation_errors as error:
    return dict(**report, seconds=monotonic() - start, reason=str(error))
  best = np.full(book.regions.count, np.inf)
  queue = [oracle.tensor(v) for v in (*profiles, (profiles[0] + profiles[1]) * 0.5)]
  best_root = np.inf
  best_price = best_gradient = None
  evaluated = []
  for i in range(12):
    if monotonic() >= deadline or book.stop.is_set():
      break
    if i < len(queue):
      price = queue[i]
    else:
      g = best_gradient.astype(float)
      norm = float(g @ g)
      if norm == 0:
        break
      step = max(0, best_root - book.latest().welfare) / (norm * (1 + i - len(queue)))
      price = oracle.xp.minimum((best_price - step * g).clip(0, None), oracle.reward)
    key = oracle.xp.round(price * oracle.scale)
    if any(bool((key == prior).all()) for prior in evaluated):
      continue
    evaluated.append(key)
    try:
      result = oracle.evaluate(price, deadline)
    except allocation_errors as error:
      report['reason'] = str(error)
      break
    if result is None:
      break
    bounds, root, g = result
    best = np.minimum(best, bounds)
    if root < best_root:
      best_root, best_price, best_gradient = root, price, g
    report['evaluations'] += 1
    # Apply each proven bound within its own region only.
    with book.lock:
      for b in np.flatnonzero(bounds < book.upper):
        book.tighten(bounds[b], int(b))
  if report['evaluations']:
    report.update(
      covering_upper=float(best.max()), regional_upper=best.tolist(), root_upper=best_root
    )
  return dict(**report, seconds=monotonic() - start)
