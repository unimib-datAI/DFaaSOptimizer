"""PG proposals must not require unrelated nodes' parameters or weights."""
import numpy as np
import pytest

from decentralized_potentialgame import propose_node_move
from models.sp import LSP_pg, LSP_pg_fixedr


@pytest.mark.parametrize('model', [LSP_pg(), LSP_pg_fixedr()])
def test_proposal_uses_own_parameters_and_inbound_commitments(model):
  class RemoteWeights(dict):
    def __deepcopy__(self, memo):
      raise AssertionError('proposal copied unrelated network weights')

  # Node 2's parameters are deliberately unavailable. Its committed unit still
  # requires capacity at node 1: ceil((4 + 1) / (0.8 / 0.5)) == 4 replicas.
  data = {None: {
    'Nn': {None: 2}, 'Nf': {None: 2},
    'incoming_load': {(1, 1): 4, (1, 2): 0},
    'demand': {(1, 1): .5, (1, 2): .5},
    'memory_requirement': {1: 1, 2: 2}, 'memory_capacity': {1: 4},
    'alpha': {(1, 1): 1., (1, 2): 1.},
    'delta': {(1, 1): 0., (1, 2): 0.},
    'gamma': {(1, 1): 1., (1, 2): 1.},
    'r_bar': {(1, 1): 4, (1, 2): 0}, 'beta': RemoteWeights(),
  }}
  y = np.zeros((2, 2, 2))
  y[1, 0, 0] = 1.
  x, r, omega, _ = propose_node_move(
    0, np.zeros(2), y, data, model, 'missing_solver', {'use_dp': True}, 0,
  )
  np.testing.assert_array_equal(x, [4, 0])
  np.testing.assert_array_equal(r, [4, 0])
  np.testing.assert_array_equal(omega, [0, 0])


def test_compact_proposals_preserve_full_native_allocations():
  from benchmark_madea_pg_compact import full_propose
  rng = np.random.default_rng(2026)
  nn, nf = 4, 3
  for _ in range(30):
    data = {None: {
      'Nn': {None: nn}, 'Nf': {None: nf},
      'memory_capacity': {i: 12 for i in range(1, nn + 1)},
      'memory_requirement': {f: f for f in range(1, nf + 1)},
      'max_utilization': {f: .8 for f in range(1, nf + 1)},
      'incoming_load': {(i, f): int(rng.integers(0, 9))
                        for i in range(1, nn + 1) for f in range(1, nf + 1)},
      'demand': {(i, f): float(rng.choice([.2, .4, .7]))
                 for i in range(1, nn + 1) for f in range(1, nf + 1)},
      'r_bar': {(i, f): 1 for i in range(1, nn + 1) for f in range(1, nf + 1)},
    }}
    for parameter in ('alpha', 'delta', 'gamma'):
      data[None][parameter] = {(i, f): float(rng.integers(0, 6))
                              for i in range(1, nn + 1) for f in range(1, nf + 1)}
    y = rng.choice([0., .05, .125], size=(nn, nn, nf))
    for i in range(nn):
      y[i, i] = 0.
    cap = rng.uniform(0, 8, size=nf)
    for model in (LSP_pg(), LSP_pg_fixedr()):
      for node in range(nn):
        args = (node, cap, y, data, model, 'missing_solver', {'use_dp': True}, 0)
        full = full_propose(*args)
        compact = propose_node_move(*args)
        for old, new in zip(full[:3], compact[:3]):
          np.testing.assert_array_equal(old, new)


def test_missing_required_parameter_retains_pyomo_diagnostic():
  data = {None: {
    'Nn': {None: 1}, 'Nf': {None: 1},
    'incoming_load': {(1, 1): 1}, 'demand': {},
    'memory_requirement': {1: 1}, 'memory_capacity': {1: 4},
  }}
  with pytest.raises(ValueError, match='demand'):
    propose_node_move(0, np.zeros(1), np.zeros((1, 1, 1)), data, LSP_pg(), 'missing_solver', {'use_dp': True}, 0)
