from collections import deque

import numpy as np
import pytest

from run_faasmacro import (
  check_stopping_criteria,
  compute_deviation,
  merge_agents_solutions,
  prepare_master_data,
  update_neighborhood,
  update_prices,
)


def _base_data():
  return {
    None: {
      "Nn": {None: 2},
      "Nf": {None: 2},
      "memory_capacity": {1: 10, 2: 12},
      "memory_requirement": {1: 2, 2: 3},
      "demand": {(1, 1): 2.0, (2, 1): 2.0, (1, 2): 4.0, (2, 2): 4.0},
      "max_utilization": {1: 0.5, 2: 0.5},
    }
  }


def test_check_stopping_criteria_max_iteration():
  stop, why, new_psi = check_stopping_criteria(
    it = 4,
    max_iterations = 5,
    sp_omega = np.array([[0.0]]),
    rmp_omega = np.array([[1.0]]),
    pi_queue = deque(maxlen = 2),
    dev_queue = deque([np.array([1.0])], maxlen = 2),
    sw_queue = deque(maxlen = 2),
    current_sw_queue = deque(maxlen = 2),
    odev_queue = deque(maxlen = 2),
    psi = 2.0,
    tolerance = 1e-3,
    total_runtime = 1.0,
    time_limit = 100.0,
  )
  assert stop is True
  assert why == "max iterations reached"
  assert new_psi == 2.0


def test_check_stopping_criteria_discount_psi_when_stagnating():
  current_sw_queue = deque([1.0, 1.1], maxlen = 2)
  stop, why, new_psi = check_stopping_criteria(
    it = 0,
    max_iterations = 10,
    sp_omega = np.array([[1.0, 2.0]]),
    rmp_omega = np.array([[0.0, 0.0]]),
    pi_queue = deque([{1: 0.1, 2: 0.2}], maxlen = 3),
    dev_queue = deque([np.array([1.0, 1.0])], maxlen = 3),
    sw_queue = deque([1.0], maxlen = 3),
    current_sw_queue = current_sw_queue,
    odev_queue = deque([1.0], maxlen = 3),
    psi = 2.0,
    tolerance = 1e-3,
    total_runtime = 1.0,
    time_limit = 100.0,
  )
  assert stop is False
  assert why is None
  assert new_psi == 1.0
  assert len(current_sw_queue) == 0


def test_compute_deviation_and_detailed_deviation():
  rmp_data = _base_data()
  sp_x = np.array([[1.0, 0.0], [1.0, 0.0]])
  sp_omega = np.array([[0.2, 0.1], [0.3, 0.2]])
  rmp_r = np.array([[4.0, 2.0], [4.0, 2.0]])
  rmp_omega = np.array([[0.1, 0.0], [0.2, 0.1]])

  dev, nf_thr, detailed = compute_deviation(rmp_data, sp_x, sp_omega, rmp_r, rmp_omega)
  assert dev.shape == (2,)
  assert nf_thr == [0.0, 0.5]
  assert detailed.shape == (2, 2)
  assert detailed[0, 0] == pytest.approx(0.1)


def test_update_neighborhood_removes_incoming_edges_to_saturated_nodes():
  neighborhood = {
    (1, 1): 0, (1, 2): 1,
    (2, 1): 1, (2, 2): 0,
  }
  out = update_neighborhood(
    neighborhood,
    sp_rho = np.array([0.0, 4.0]),
    sp_omega = np.zeros((2, 1)),
  )
  assert out[(2, 1)] == 0
  assert out[(1, 2)] == 1


def test_update_prices_updates_functional_and_detailed_components():
  old_pi = {1: 1.0, 2: 2.0}
  old_detailed_pi = np.array([[1.0, 1.0], [0.5, 0.5]])
  dev = np.array([0.0, 2.0])
  detailed_dev = np.array([[0.0, 1.0], [0.0, -1.0]])

  pi, detailed = update_prices(
    dev = dev,
    detailed_dev = detailed_dev,
    psi = 1.0,
    delta_w = 2.0,
    old_pi = old_pi,
    old_detailed_pi = old_detailed_pi,
  )
  assert pi[1] == 1.0
  assert pi[2] == pytest.approx(3.0)
  assert detailed[0, 1] == pytest.approx(2.0)
  assert detailed[1, 1] == pytest.approx(0.0)


def test_prepare_master_data_clips_negative_values():
  base_data = _base_data()
  sp_solution = (
    np.array([[1.0, -1.0], [2.0, -2.0]]),
    np.array([
      [[0.0, -1.0], [1.0, 0.0]],
      [[-3.0, 0.0], [0.0, 2.0]],
    ]),
    np.array([[0.0, -4.0], [1.0, -1.0]]),
    np.array([[0.2, -0.3], [0.4, 0.5]]),
    np.array([[1.0, -2.0], [2.0, -3.0]]),
    np.array([1.0, 2.0]),
  )
  out = prepare_master_data(base_data, sp_solution)
  assert out[None]["x_bar"][(1, 2)] == 0
  assert out[None]["d_bar"][(2, 1, 1)] == 0
  assert out[None]["z_bar"][(1, 2)] == 0
  assert out[None]["r_bar"][(2, 2)] == 0
  assert out[None]["omega_bar"][(1, 2)] == 0


def test_merge_agents_solutions_aggregates_obj_tc_runtime():
  data = _base_data()
  agents_sol = {
    0: {
      "x": [1.0, 2.0],
      "y": [0.0, 0.0, 0.0, 0.0],
      "z": [0.0, 1.0],
      "omega": [0.0, 0.0],
      "r": [1.0, 1.0],
      "obj": 5.0,
      "termination_condition": "ok0",
      "runtime": 2.0,
    },
    1: {
      "x": [3.0, 4.0],
      "y": [0.0, 0.0, 0.0, 0.0],
      "z": [0.0, 2.0],
      "omega": [0.0, 0.0],
      "r": [2.0, 1.0],
      "obj": 7.0,
      "termination_condition": "ok1",
      "runtime": 4.0,
    },
  }
  x, y, z, omega, r, rho, obj, tc, runtime = merge_agents_solutions(data, agents_sol)
  assert x.shape == (2, 2)
  assert y.shape == (2, 2, 2)
  assert z.shape == (2, 2)
  assert omega.shape == (2, 2)
  assert r.shape == (2, 2)
  assert rho.shape == (2,)
  assert obj["tot"] == pytest.approx(12.0)
  assert tc["tot"] == "ok0-ok1"
  assert runtime["tot"] == pytest.approx(3.0)
