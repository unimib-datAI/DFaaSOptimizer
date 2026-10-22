"""Selective requests carry only MADEA's (i,j,f), with bounded local retries."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from utils import step_time
from utils.selective_memory import SelectiveMemory


def test_sender_selects_one_neighbor_and_retries_without_extra_fields():
  selector = SelectiveMemory()
  bids = pd.DataFrame({'i': [0, 0], 'j': [1, 2], 'f': [0, 0]})
  data = {None: {'beta': {(1, 2, 1): 8., (1, 3, 1): 5.}}}  # only sender's beta
  omega = np.array([[4.], [0.], [0.]])
  prices = np.zeros((3, 1))
  latency = np.zeros((3, 3))
  for expected, pending in [(1, True), (2, False), (None, False)]:
    got = selector.select(bids, omega, prices, data, latency, {'latency_weight': 0.})
    assert list(got.columns) == ['i', 'j', 'f']
    assert got.j.tolist() == ([] if expected is None else [expected])
    assert selector.pending == pending
    assert selector.retry(0, 2, 0., 1.) == pending
    assert not selector.retry(1, 2, 0., 1.)
    assert not selector.retry(0, 2, 1., 1.)
  # Successful progress permits another request; no global state leaks between steps.
  omega[0, 0] = 2
  assert selector.select(bids, omega, prices, data, latency, {}).j.tolist() == [1]
  assert SelectiveMemory().select(bids, omega, prices, data, latency, {}).j.tolist() == [1]
  # Prices and latency, already visible to the sender, can change its preference.
  prices[1, 0] = 4
  assert SelectiveMemory().select(bids, omega, prices, data, latency, {}).j.tolist() == [2]
  omega[0, 0] = 0
  assert selector.select(bids, omega, prices, data, latency, {}).empty
  assert not selector.pending


def test_satisfied_buyer_can_retry_after_its_accepted_flow_is_displaced():
  selector = SelectiveMemory()
  bids = pd.DataFrame({'i': [0], 'j': [1], 'f': [0]})
  data = {None: {'beta': {(1, 2, 1): 1.}}}
  omega, prices, latency = np.array([[4.], [0.]]), np.zeros((2, 1)), np.zeros((2, 2))
  assert selector.select(bids, omega, prices, data, latency, {}).j.tolist() == [1]
  # Confirmation removes this buyer's whole group once its demand is satisfied.
  omega[0, 0] = 0
  assert selector.select(bids.iloc[:0], omega, prices, data, latency, {}).empty
  omega[0, 0] = 4
  assert selector.select(bids, omega, prices, data, latency, {}).j.tolist() == [1]


@pytest.mark.parametrize('one_shot', [False, True])
@pytest.mark.parametrize('iterations,served', [(1, 0), (5, 4)])
def test_failed_memory_request_tries_next_neighbor_but_respects_budget(one_shot, iterations, served, tmp_path):
  import run_faasmadea as madea
  import decentralized_auction as auction
  module = auction if one_shot else madea
  config = json.loads((Path(__file__).parents[1]/'experiments/madea_pg_compact/catalog_config_overnight.json').read_text())
  config['max_iterations'] = iterations
  config['solver_options']['general'].update(use_dp=True, algorithm_time_limit=60)
  data = {None: {
    'Nn': {None: 3}, 'Nf': {None: 1},
    'incoming_load': {(n, 1): 4 if n == 1 else 0 for n in (1, 2, 3)},
    'demand': {(n, 1): 1. for n in (1, 2, 3)},
    'alpha': {(n, 1): 2. for n in (1, 2, 3)},
    'gamma': {(n, 1): .1 for n in (1, 2, 3)},
    'delta': {(n, 1): .5 for n in (1, 2, 3)},
    'beta': {(n, j, 1): (.9 if j == 2 else .8) if n != j else 0.
             for n in (1, 2, 3) for j in (1, 2, 3)},
    'neighborhood': {(n, j): int(n != j) for n in (1, 2, 3) for j in (1, 2, 3)},
    'memory_capacity': {1: 0, 2: 1, 3: 4},
    'memory_requirement': {1: 1}, 'max_utilization': {1: 1.},
  }}
  ctx = SimpleNamespace(Nn=3, Nf=1, tolerance=1e-6, config=config,
    solver_options=config['solver_options'], general_options=config['solver_options']['general'],
    max_iterations=iterations, time_limit=60, reference=None, step_data=lambda d,t:deepcopy(d),
    agents=list(range(3)), solver_name='missing_solver', parallelism=0,
    neighborhood=np.ones((3, 3))-np.eye(3), latency=np.zeros((3, 3)))
  seen = []
  def allocate(bids, r, d, rho):
    assert list(bids.columns) == ['i', 'j', 'f']
    assert bids.groupby(['i', 'f']).size().le(1).all()
    seen.extend(bids.j.tolist())
    if len(seen) == 1:  # first neighbor cannot allocate this round
      return np.zeros_like(r), rho.copy()
    return madea.start_additional_replicas(bids, r, d, rho)
  rules = {name: getattr(module, name) for name in madea.RULE_NAMES}
  rules['start_additional_replicas'] = allocate
  clock = step_time.start_step()
  try:
    from utils import communication
    name = 'MADEA-one-shot-selective' if one_shot else 'MADEA-selective'
    with communication.session(True, name, tmp_path):
      result = module._solve_step(ctx, 0, data, rules=rules, refinement=None,
                                 detailed_init=True, selective_memory=True)
      communication.finish()
  finally:
    step_time.end_step(clock)
  assert seen == ([1] if iterations == 1 else [1, 2])
  messages = pd.read_csv(tmp_path/'communication.csv')
  assert messages.loc[messages.kind == 'memory_request', 'items'].sum() == len(seen)
  central = result.solutions['LSPc']
  assert central['y'].sum() == served
  assert np.all(central['x'] == 0)
  from utils.centralized import validate_centralized_solution
  validate_centralized_solution(central['x'], central['y'], central['z'], central['r'], data)
