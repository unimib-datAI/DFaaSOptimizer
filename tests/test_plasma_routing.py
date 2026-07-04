import numpy as np
import pytest

from plasma.core.types import LOCAL, REJ, PlasmaOptions
from plasma.core.routing import choose_target, target_weights, update_conductance


def test_local_gate_closes_local_column():
  D_f = np.array([10.0, 0.1, 5.0])  # LOCAL, REJ, one neighbor
  w = target_weights(D_f, local_open=False, nbr_spare=np.array([1.0]),
                     eps_explore=0.01)
  assert w[LOCAL] == 0.0
  assert w[2] == 5.0


def test_neighbor_without_spare_gets_exploration_floor():
  D_f = np.array([1.0, 0.1, 4.0])
  w = target_weights(D_f, local_open=True, nbr_spare=np.array([0.0]),
                     eps_explore=0.01)
  assert w[2] == pytest.approx(4.0 * 0.01)


def test_choose_target_unsplittable_is_argmax():
  rng = np.random.default_rng(0)
  w = np.array([1.0, 0.5, 7.0])
  assert choose_target(rng, w, unsplittable=True) == 2


def test_choose_target_sampled_follows_weights():
  rng = np.random.default_rng(0)
  w = np.array([0.0, 0.0, 1.0])
  assert choose_target(rng, w, unsplittable=False) == 2


def test_conductance_reinforces_accepted_traffic():
  opts = PlasmaOptions()
  D = np.full((1, 3), 1.0)
  phi = np.array([[10.0, 0.0, 0.0]])
  rewards = np.array([[2.0, 0.0, 0.0]])
  D2 = update_conductance(D, phi, rewards, opts)
  assert D2[0, LOCAL] == pytest.approx(0.9 * 1.0 + 0.1 * 10.0 * 2.0)


def test_dead_neighbor_conductance_decays_to_floor():
  opts = PlasmaOptions()
  D = np.full((1, 3), 100.0)
  phi = np.zeros((1, 3))
  rewards = np.zeros((1, 3))
  for _ in range(200):
    D = update_conductance(D, phi, rewards, opts)
  # pure evaporation: (1-mu)^200 * 100 << D_min -> clipped at floor
  assert D[0, 2] == pytest.approx(opts.D_min)


def test_conductance_clipped_above():
  opts = PlasmaOptions()
  D = np.full((1, 3), 1.0)
  phi = np.full((1, 3), 1e9)
  rewards = np.full((1, 3), 1e9)
  D2 = update_conductance(D, phi, rewards, opts)
  assert (D2 <= opts.D_max).all()
