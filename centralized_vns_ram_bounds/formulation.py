"""Per-receiver traffic envelopes and compatible integer RAM choices."""

from itertools import combinations
import numpy as np
from scipy.sparse import csr_matrix
from centralized_vns_highs.mip import require_ok, ram_receive_limit


def levels(x, capacity, baseline, upper):
  """Undominated (extra replicas, integer traffic) pairs using checker arithmetic."""
  if upper > 65536:
    pairs = []
    first = 1
    while first <= upper:
      r = int(max(1, np.ceil((int(x) + first) / capacity - 1e-9)))
      left, right = first, upper
      while left < right:
        mid = (left + right + 1) // 2
        if np.ceil((int(x) + mid) / capacity - 1e-9) <= r:
          left = mid
        else:
          right = mid - 1
      pairs.append((int(r - baseline), left))
      first = left + 1
    return pairs
  demand = np.arange(upper + 1, dtype=np.int64)
  replicas = np.maximum(x + demand > 0, np.ceil((x + demand) / capacity - 1e-9)).astype(np.int64)
  last = np.r_[np.flatnonzero(np.diff(replicas)), upper]
  return [(int(replicas[i] - baseline), int(i)) for i in last if i > 0]


def add_ram_rows(h, p, *, options_by_cell=None, paths_by_node=None):
  n, fcount = p.x.shape
  ne = len(p.src)
  base = p.check(np.zeros(ne, dtype=np.int64))['replicas']
  free = p.ram - base @ p.memory
  supply = (p.load - p.x)[p.src, p.fun]
  total = np.zeros_like(p.x)
  np.add.at(total, (p.dst, p.fun), supply)
  capacity = ram_receive_limit(p, total.ravel(), base).reshape(p.x.shape)
  cap_edge = np.minimum(supply, capacity[p.dst, p.fun])
  incoming = [[] for _ in range(p.x.size)]
  for e in np.flatnonzero(cap_edge):
    incoming[p.dst[e] * fcount + p.fun[e]].append(int(e))
  first = h.getNumCol()
  next_col = first
  cols, values, ptr, rhs = [], [], [0], []

  def column():
    nonlocal next_col
    idx = next_col
    next_col += 1
    return idx

  def row(entries, limit):
    for c, v in entries:
      if v:
        cols.append(c)
        values.append(v)
    ptr.append(len(cols))
    rhs.append(limit)

  for e in range(ne):
    s = ne + p.x.size + p.src[e] * fcount + p.fun[e]
    if cap_edge[e] < supply[e]:
      row([(e, 1), (s, -int(cap_edge[e]))], 0)

  arcs = contended = 0
  for j in range(n):
    layers = []
    for f in range(fcount):
      cell = j * fcount + f
      edges = incoming[cell]
      if not edges:
        continue
      # Bound by the sum of individually deliverable flows before building levels.
      upper = min(int(capacity[j, f]), sum(int(cap_edge[e]) for e in edges))
      choices = [
        (k, q, column()) for k, q in levels(p.x[j, f], p.capacity[j, f], base[j, f], upper)
      ]
      if options_by_cell is not None:
        if choices[0][0] > 0:
          choices.insert(0, (0, 0, column()))
        options_by_cell[j, f] = choices
      role = ne + p.x.size + cell
      row([(role, 1)] + [(z, 1) for _, _, z in choices], 1)
      if options_by_cell is not None:
        row([(role, -1)] + [(z, -1) for _, _, z in choices], -1)
      row([(ne + cell, -1)] + [(z, k) for k, _, z in choices], -int(base[j, f]))
      # Singletons and the full set stay valid at high degree; avoid exponential growth.
      sizes = range(1, len(edges) + 1) if len(edges) <= 6 else (1, len(edges))
      for size in sizes:
        for subset in combinations(edges, size):
          supply_subset = sum(int(cap_edge[e]) for e in subset)
          row([(e, 1) for e in subset] + [(z, -min(q, supply_subset)) for _, q, z in choices], 0)
      if choices[-1][0]:
        options = {k * int(p.memory[f]): z for k, _, z in choices if k}
        layers.append(options)
    # Merge states with enough RAM left to buy every remaining option.
    layers.sort(key=lambda opts: min(opts), reverse=True)
    future = sum(max(opts) for opts in layers)
    if future <= free[j]:
      continue
    contended += 1
    states = {int(free[j]): []}
    if paths_by_node is not None:
      paths_by_node[j] = (int(free[j]), [])
    for i, opts in enumerate(layers):
      future -= max(opts)
      next_states = {}
      use = {cost: [] for cost in opts}
      if paths_by_node is not None:
        transitions = {}
        paths_by_node[j][1].append((opts, transitions))
      for left, arrivals in states.items():
        exits = []
        for cost in (0, *opts):
          if cost > left:
            continue
          a = column()
          arcs += 1
          exits.append((a, 1))
          destination = min(future, left - cost)
          next_states.setdefault(destination, []).append(a)
          if paths_by_node is not None:
            transitions[left, cost] = (destination, a)
          if cost:
            use[cost].append(a)
        # A subunit flow suffices: idle includes every unused fraction.
        row(exits + [(a, -1) for a in arrivals], 1 if i == 0 else 0)
      for cost, z in opts.items():
        row([(z, 1)] + [(a, -1) for a in use[cost]], 0)
      states = next_states
  count = next_col - first
  require_ok(h.addVars(count, np.zeros(count), np.ones(count)))
  matrix = csr_matrix(
    (
      np.asarray(values, dtype=float),
      np.asarray(cols, dtype=np.int32),
      np.asarray(ptr, dtype=np.int32),
    ),
    shape=(len(rhs), next_col),
  )
  require_ok(
    h.addRows(
      len(rhs),
      np.full(len(rhs), -np.inf),
      np.array(rhs, dtype=float),
      matrix.nnz,
      matrix.indptr,
      matrix.indices,
      matrix.data,
    )
  )
  return dict(
    additional_rows=len(rhs),
    additional_columns=count,
    ram_flow_arcs=arcs,
    contended_nodes=contended,
  )


def propagate(p, lower, upper, options_by_cell):
  """Use only consequences of mandatory integer flows inside each leaf."""
  ne = len(p.src)
  base = p.check(np.zeros(ne, dtype=np.int64))['replicas']
  for lo, hi in zip(lower, upper):
    required = lo[:ne].astype(np.int64)
    checked = p.check(required)
    incoming = checked['incoming']
    outgoing = p.load - p.x - checked['z']
    copies = checked['replicas']
    remaining = p.ram - copies @ p.memory
    sender_lower = lo[ne + p.x.size : ne + 2 * p.x.size].reshape(p.x.shape)
    sender_upper = hi[ne + p.x.size : ne + 2 * p.x.size].reshape(p.x.shape)
    sender_lower[outgoing > 0] = 1
    sender_upper[incoming > 0] = 0
    lo[ne : ne + p.x.size] = np.maximum(lo[ne : ne + p.x.size], copies.ravel())
    for (j, f), options in options_by_cell.items():
      limit = copies[j, f] - base[j, f] + remaining[j] // p.memory[f]
      for k, q, z in options:
        if q < incoming[j, f] or k > limit:
          hi[z] = 0
    hi[:ne] = np.minimum(hi[:ne], checked['z'][p.src, p.fun] + required)
    hi[:ne][(incoming[p.src, p.fun] > 0) | (outgoing[p.dst, p.fun] > 0)] = 0
    if np.any(lo > hi):
      raise ArithmeticError('forced flow contradicts propagated bounds')
