"""Sparse fixed-x MIP. Column order is edge flow, replica count, sender flag."""
import highspy
import numpy as np
from scipy.sparse import bmat, csr_matrix, diags


def require_ok(status):
  if status != highspy.HighsStatus.kOk:
    raise RuntimeError(f'HiGHS returned {status}')


def ram_receive_limit(p, supply, baseline_replicas):
  """Largest integer inflow compatible with fixed x, mandatory replicas and supply."""
  free = p.ram - baseline_replicas @ p.memory
  max_replicas = baseline_replicas + free[:, None] // p.memory
  low = np.zeros_like(p.x)
  high = supply.reshape(p.x.shape).astype(np.int64)
  # Use the checker's division and rounding even at fractional replica boundaries.
  while np.any(low < high):
    mid = (low + high + 1) // 2
    processing = p.x + mid
    with np.errstate(over='ignore'):
      required = np.maximum(processing > 0, np.ceil(processing / p.capacity - 1e-9))
    fits = required <= max_replicas
    low = np.where(fits, mid, low)
    high = np.where(fits, high, mid - 1)
  return low.ravel()


def fixed_model(p, *, tighten_receive=False):
  solver = highspy.Highs()
  require_ok(solver.setOptionValue('output_flag', False))
  require_ok(solver.setOptionValue('mip_rel_gap', 0.0))
  require_ok(solver.setOptionValue('mip_abs_gap', 0.0))
  nodes, functions = p.load.shape
  cells = p.x.size
  edges = np.arange(len(p.src))
  source = p.src * functions + p.fun
  destination = p.dst * functions + p.fun
  residual = (p.load - p.x).ravel()
  limits = residual[source]

  def incidence(endpoint):
    return csr_matrix((np.ones(len(edges)), (endpoint, edges)), shape=(cells, len(edges)))

  outgoing, incoming = incidence(source), incidence(destination)
  receive_limit = np.asarray(incoming @ limits).ravel()
  baseline_replicas = p.check(np.zeros(len(edges)))['replicas']
  role_limit = ram_receive_limit(p, receive_limit, baseline_replicas) if tighten_receive else receive_limit
  x = p.x.ravel()
  ram = csr_matrix((np.tile(p.memory, nodes), (np.repeat(np.arange(nodes), functions),
                   np.arange(cells))), shape=(nodes, cells))
  # A capacity above all possible processing fits everything in one replica.
  capacity = np.minimum(p.capacity.ravel(), x + receive_limit)
  matrix = bmat([
    [outgoing, None, -diags(residual)],
    [incoming, None, diags(role_limit)],
    [incoming, -diags(capacity), None],
    [incoming, -diags(x + receive_limit), None],
    [None, ram, None],
  ], format='csr')
  rhs = np.r_[np.zeros(cells), role_limit, capacity * 1e-9 - x, -x, p.ram]
  minimum = np.r_[np.zeros(len(edges)), baseline_replicas.ravel(),
                  np.zeros(cells)]
  maximum = np.r_[limits, (p.ram[:, None] // p.memory).ravel(), np.ones(cells)]
  columns = np.arange(len(maximum), dtype=np.int32)
  require_ok(solver.addVars(len(columns), minimum, maximum))
  require_ok(solver.changeColsIntegrality(len(columns), columns, np.ones(len(columns), np.uint8)))
  require_ok(solver.changeColsCost(len(edges), edges.astype(np.int32), -p.reward))
  require_ok(solver.addRows(len(rhs), np.full(len(rhs), -np.inf), rhs, matrix.nnz,
                            matrix.indptr, matrix.indices, matrix.data))
  return solver


def apply_rows(solver, rows):
  for edges, limit in rows:
    require_ok(solver.addRow(-np.inf, limit, len(edges), edges, np.ones(len(edges))))


def encode_solution(p, y, checked):
  sender = checked['z'] < p.load - p.x
  return np.r_[y, checked['replicas'].ravel(), sender.ravel()].astype(float)
