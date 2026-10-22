import copy

import networkx as nx
import numpy as np
import pytest

from generators import generate_data as generator
from remote_experiments.definitions.paper import build_e0
from remote_experiments.instances import generation_spec, instance_id


@pytest.mark.parametrize('kind', ['homogeneous', 'heterogeneous'])
def test_generated_delta_is_the_all_node_mean_with_zero_diagonal(kind):
  limits = {'weights': {
    'type': kind, 'alpha': {'min': 10., 'max': 10.},
    'beta_multiplier': {'min': .6, 'max': .6},
    'gamma': {'min': 1., 'max': 1.},
    'delta_multiplier': {'min': .1, 'max': .2},
  }}
  _, beta, _, delta = generator.generate_weights(
    3, 2, limits, np.random.default_rng(12), nx.path_graph(3))
  assert np.all(beta[np.arange(3), np.arange(3), :] == 0.)
  # Includes the non-neighbor and the zero self-entry; it is neither a
  # neighbors-only mean nor beta times the old delta_multiplier.
  np.testing.assert_allclose(delta, np.full((3, 2), 4.))
  np.testing.assert_allclose(delta, beta.mean(axis=1))


def test_recompute_mean_after_transforming_beta():
  beta = np.full((3, 3, 1), 6.)
  first = generator.offload_gain_mean(beta)
  np.testing.assert_allclose(first, 4.)
  beta[0, :, :] = 3.
  revised = generator.offload_gain_mean(beta)
  np.testing.assert_allclose(revised[:, 0], [2., 4., 4.])
  assert beta[0, 0, 0] == 0.


def test_generation_identity_marks_the_new_weight_rule():
  experiment = build_e0(seeds=(42,), algorithms=('centralized',))[0]
  spec = generation_spec(experiment)
  assert spec['weight_generation'] == 'paper-delta-mean-v1'
  import hashlib
  import json
  legacy = copy.deepcopy(spec)
  legacy.pop('weight_generation')
  legacy_id = 'instance-' + hashlib.sha256(json.dumps(
    legacy, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:16]
  assert instance_id(experiment) != legacy_id
