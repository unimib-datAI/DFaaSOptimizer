"""One-shot may reuse an optimum only within a fixed local problem."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import decentralized_auction as auction
from hierarchical_auction import runner as hierarchy
from utils import step_time


@pytest.mark.parametrize('variant', ['standard', 'detailed', 'selective', 'hierarchical', 'hierarchical_moves'])
@pytest.mark.parametrize('condition', ['optimal', 'maxTimeLimit', 'optimal-maxTimeLimit'])
def test_unchanged_rounds_reuse_only_optima(variant, condition, monkeypatch):
  is_hierarchical = variant.startswith('hierarchical')
  module = hierarchy if is_hierarchical else auction
  config = json.loads((Path(__file__).parents[1] /
                      'experiments/madea_pg_compact/catalog_config_overnight.json').read_text())
  config['max_iterations'] = 4
  config['solver_options']['general'].update(use_dp=True, algorithm_time_limit=60)
  data = {None: {
    'Nn': {None: 2}, 'Nf': {None: 1},
    'incoming_load': {(1, 1): 4, (2, 1): 0},
    'demand': {(n, 1): 1. for n in (1, 2)},
    'alpha': {(n, 1): 2. for n in (1, 2)},
    'gamma': {(n, 1): .1 for n in (1, 2)},
    'delta': {(n, 1): .5 for n in (1, 2)},
    'beta': {(n, j, 1): float(n != j) for n in (1, 2) for j in (1, 2)},
    'neighborhood': {(n, j): int(n != j) for n in (1, 2) for j in (1, 2)},
    'memory_capacity': {1: 0, 2: 8},
    'memory_requirement': {1: 1}, 'max_utilization': {1: 1.},
  }}
  ctx = SimpleNamespace(Nn=2, Nf=1, tolerance=1e-6, config=config,
    solver_options=config['solver_options'], general_options=config['solver_options']['general'],
    max_iterations=4, time_limit=60, agents=[0, 1], solver_name='missing_solver', parallelism=0,
    neighborhood=np.ones((2, 2))-np.eye(2), latency=np.zeros((2, 2)), dense_latency=np.zeros((2, 2)))
  calls, rounds = [], []
  real_welfare = module.compute_social_welfare

  def welfare(*args):
    calls.append(args[5].copy())
    solution, objective, _, runtime = real_welfare(*args)
    return solution, objective, condition, runtime

  def define(*args, **kwargs):
    return (pd.DataFrame({'i': [0], 'j': [1], 'f': [0], 'd': [1], 'b': [1.]}),
            pd.DataFrame(columns=['i', 'j', 'f']), 1)

  def evaluate(*args, **kwargs):
    y = np.zeros((2, 2, 1))
    y[0, 1, 0] = int(len(rounds) % 2 == 0)
    rounds.append(1)
    return y, kwargs['p'], np.zeros((2, 1)), 1

  def allocate(bids, r, data, rho):
    # Tentative replicas must not mutate the cached restricted solution.
    extra = np.zeros_like(r)
    extra[1, 0] = 1
    remaining = rho.copy()
    remaining[1] -= 1
    return extra, remaining

  def stop(*args, **kwargs):
    return (kwargs['it'] >= 3, 'test rounds completed')

  class IdleHierarchy:
    def __init__(self, **kwargs):
      pass

    def run_higher_levels(self, **kwargs):
      y = kwargs['y'].copy()
      accepted = variant == 'hierarchical_moves' and len(rounds) == 1
      if accepted:
        y[0, 1, 0] += 1
      return SimpleNamespace(y=y, accepted_allocations=[1] if accepted else [])

  monkeypatch.setattr(module, 'compute_social_welfare', welfare)
  monkeypatch.setattr(module, 'check_stopping_criteria', stop)
  rules = dict(define_bids=define, evaluate_bids=evaluate, start_additional_replicas=allocate)
  if is_hierarchical:
    for name, rule in rules.items():
      monkeypatch.setattr(module, name, rule)
    options = dict(engine_class=IdleHierarchy, auction_options=config['solver_options']['auction'])
  else:
    options = dict(rules=rules, detailed_init=variant != 'standard', selective_memory=variant == 'selective')

  def run(step):
    rounds.clear()
    clock = step_time.start_step()
    try:
      return module._solve_step(ctx, step, deepcopy(data), refinement=None, **options)
    finally:
      step_time.end_step(clock)

  actual = run(0)
  assert len(rounds) == 4
  assert len(calls) == (2 if condition == 'optimal' else 4) + int(variant == 'hierarchical_moves')
  # A fresh step must solve again, even for the same flows.
  count = len(calls)
  run(1)
  assert len(calls) == 2 * count
  # The uncached execution must produce exactly the same solution.
  monkeypatch.setattr(module, 'cache_fixed_welfare', lambda solve: solve)
  expected = run(0)
  for key, array in actual.solutions['LSPc'].items():
    np.testing.assert_array_equal(array, expected.solutions['LSPc'][key])


def test_cache_copies_inputs_outputs_and_invalidates_on_demand_changes():
  from run_faasmacro import cache_fixed_welfare
  calls = []

  def solve(y, omega):
    calls.append(1)
    return [y.copy(), omega.copy()], 7., 'optimal-optimal', 12.

  cached = cache_fixed_welfare(solve)
  y, omega = np.zeros((2, 2, 1)), np.zeros((2, 1))
  first = cached(y, omega)
  first[0][0][:] = 99
  second = cached(y, omega)
  assert len(calls) == 1 and second[3] == 0
  assert not second[0][0].any()
  second[0][1][:] = 99
  assert not cached(y, omega)[0][1].any()
  omega[0, 0] = 1
  cached(y, omega)
  y[0, 1, 0] = 1
  cached(y, omega)
  assert len(calls) == 3
