"""Zero load contributes no utility; positive loads keep their original scale."""

import numpy as np
import pytest

from decentralized_potentialgame import compute_node_utility
from utils.faasmacro import compute_centralized_objective


@pytest.mark.parametrize("loads,local,rejected,forwarded,expected", [
  ([0.0, 0.0], [0.0, 0.0], [0.0, 0.0], 0.0, 0.0),
  ([0.0, 2.0], [0.0, 1.0], [0.0, 1.0], 0.0, 0.3),
  ([0.5, 2.0], [0.25, 1.0], [0.25, 1.0], 0.0, 0.6),
  ([2.0, 0.0], [0.0, 0.0], [0.0, 0.0], 2.0, 0.9),
], ids=["idle-system", "idle-source", "positive-fractional", "idle-source-receives"])
def test_centralized_scoring_preserves_per_source_normalization(
    loads, local, rejected, forwarded, expected,
):
  data = {None: {
    "Nn": {None: 2}, "Nf": {None: 1},
    "incoming_load": {(n + 1, 1): load for n, load in enumerate(loads)},
    "alpha": {(1, 1): 1.0, (2, 1): 1.0},
    "gamma": {(1, 1): 0.4, (2, 1): 0.4},
    "beta": {(i, j, 1): 0.9 for i in (1, 2) for j in (1, 2)},
  }}
  x = np.array(local).reshape(2, 1)
  z = np.array(rejected).reshape(2, 1)
  y = np.zeros((2, 2, 1))
  y[0, 1, 0] = forwarded

  with np.errstate(divide="raise", invalid="raise"):
    objective = compute_centralized_objective(data, x, y, z)
    node_sum = sum(compute_node_utility(n, x, y, z, data) for n in range(2))

  assert objective == pytest.approx(expected)
  assert node_sum == pytest.approx(expected)
