"""Regression cases for window units, heartbeat delivery and RL welfare."""
from types import SimpleNamespace
import subprocess
import sys

import numpy as np
import pytest

from plasma.core.node import NodeParams, PlasmaNode
from plasma.core.types import PlasmaOptions
from plasma.engine import PlasmaEngine
from plasma.runner import _solve_step
from plasma.welfare import WelfareEngine, WelfareNode, WelfareOptions
from utils import step_time
from utils.centralized import validate_centralized_solution
from test_zero_load_rl_rewards import environments as environments, _receiver


@pytest.mark.parametrize('welfare', [False, True])
@pytest.mark.parametrize('W,load,served,rejected,objective', [
  (2., 1., 1., 0., 1.),
  (.4, 1., 0., 1., -1.),
  (1., .6, 0., .6, -1.),
  (2., .75, .5, .25, 1 / 3),
])
def test_plasma_exports_rates_and_preserves_fractional_load(W, load, served, rejected, objective, welfare):
  data = {None: {
    'Nn': {None: 1}, 'Nf': {None: 1}, 'neighborhood': {(1, 1): 0},
    'alpha': {(1, 1): 1}, 'beta': {(1, 1, 1): 0}, 'gamma': {(1, 1): 1},
    'max_utilization': {1: 1}, 'demand': {(1, 1): 1},
    'memory_capacity': {1: 1}, 'memory_requirement': {1: 1},
    'incoming_load': {(1, 1): load},
  }}
  ctx = SimpleNamespace(Nn=1, Nf=1, base_data=data, config={'seed': 0}, parallelism=0)
  opts = (WelfareOptions(W=W, rounds_per_step=1) if welfare
          else PlasmaOptions(W=W, rounds_per_step=1, k_sb=0))
  result = _solve_step(ctx, 0, data, opts=opts, welfare=welfare,
                       state={'engine': None, 'messages': [], 'trades': []})
  solution = result.solutions['LSPc']
  np.testing.assert_allclose(solution['x'], [[served]])
  np.testing.assert_allclose(solution['z'], [[rejected]])
  assert result.objective == pytest.approx(objective)
  validate_centralized_solution(*(solution[k] for k in ('x', 'y', 'z', 'r')), data)
  assert result.data[None]['incoming_load'] == {(1, 1): load}


def test_zero_delay_heartbeats_allow_next_round_negotiation():
  opts = WelfareOptions(hb_latency_rounds=0, staleness_rounds=0)
  nodes = [WelfareNode(NodeParams(
    i, (1-i,), np.array([1.]), np.array([1.]), np.array([[2.]]),
    np.array([10.]), float(i), np.array([1.]),
  ), opts, np.random.default_rng(i)) for i in range(2)]
  engine = WelfareEngine(nodes, opts, np.random.default_rng(0))
  result = engine.run_rounds(4, np.array([[10], [0]]))
  assert result.y[0, 1, 0] == 10
  assert result.z.sum() == 0


@pytest.mark.parametrize('welfare', [False, True])
def test_plasma_replica_export_is_observer_time(monkeypatch, welfare):
  opts = WelfareOptions() if welfare else PlasmaOptions(k_sb=0)
  cls = WelfareNode if welfare else PlasmaNode
  node = cls(NodeParams(0, (), np.ones(1), np.ones(1), np.zeros((0, 1)),
                        np.ones(1), 1., np.ones(1)), opts, np.random.default_rng(0))
  node.init_replicas()
  engine = (WelfareEngine if welfare else PlasmaEngine)([node], opts, np.random.default_rng(0))
  now = [0.]
  real_array = np.array

  def slow_replica_export(values, *args, **kwargs):
    if isinstance(values, list) and len(values) == 1 and values[0] is node.r:
      now[0] += 5.
    return real_array(values, *args, **kwargs)

  monkeypatch.setattr(step_time, 'perf_counter', lambda: now[0])
  monkeypatch.setattr(np, 'array', slow_replica_export)
  clock = step_time.start_step()
  try:
    engine.run_rounds(1, real_array([[1]]))
  finally:
    timing = step_time.end_step(clock)
  assert timing['wall_s'] == 5.
  assert timing['algorithm_s'] == 0.


@pytest.mark.parametrize('per_node', [False, True])
def test_rl_rewards_subtract_rejection_for_every_function(environments, per_node):
  env = _receiver([1., 1.], per_node)
  for agent in env.agents:
    metrics = env.info[agent]
    if per_node:
      metrics.update(loc=[0, 0], total_fwd=[0, 0], rej=metrics['input_rate'],
                     fwd=[[0, 0]], cpu_utilization=[0, 0])
      metrics[f'fwd_to_{3-int(agent)}'] = [0, 0]
    else:
      metrics.update(loc=0, total_fwd=0, rej=metrics['input_rate'], fwd=[0], cpu_utilization=0)
      n, f = map(int, agent.split('_'))
      metrics[f'fwd_to_{3-n}_{f}'] = 0
  rewards = environments[int(per_node)].compute_reward(env)
  assert sum(rewards.values()) == -8.
  assert env.info['__common__']['cobj'] == -8.


def test_node_rl_forwarding_reaches_reward_and_objective(environments):
  cls = environments[1]
  env = _receiver([1., 1.], True)
  env.instance_data['demand'] = {(n, f): .1 for n in (1, 2) for f in (1, 2)}
  cls.init_agent_metrics(env)
  env.info['1']['input_rate'] = [1, 1]
  actions = {str(n): {'xyz_1': np.array([0., 1., 0.]), 'xyz_2': np.array([0., 1., 0.]),
                      'r': np.array([.5, .5])} for n in (1, 2)}
  cls.simulate_action(env, actions)
  rewards = cls.compute_reward(env)
  assert env.info['1']['fwd_to_2'] == [1., 1.]
  assert rewards == {'1': 6., '2': 0.}
  assert env.info['__common__']['cobj'] == 6.


def test_training_cli_help_needs_no_rl_framework():
  result = subprocess.run([sys.executable, 'train_rl_agent.py', '--help'],
                          capture_output=True, text=True)
  assert result.returncode == 0, result.stderr
  assert '--config' in result.stdout


def test_scenario_scores_fractional_arrivals_against_original_load(tmp_path, monkeypatch):
  from plasma.eval import scenario
  data = {None: {
    'Nn': {None: 1}, 'Nf': {None: 1}, 'neighborhood': {(1, 1): 0},
    'alpha': {(1, 1): 1}, 'beta': {(1, 1, 1): 0}, 'gamma': {(1, 1): 1},
    'max_utilization': {1: 1}, 'demand': {(1, 1): 1},
    'memory_capacity': {1: 1}, 'memory_requirement': {1: 1},
    'incoming_load': {(1, 1): .6},
  }}
  monkeypatch.setattr(scenario, 'init_problem', lambda *args: (data, {0: {0: [.6]}}, [0], None))
  # Baseline solvers are unrelated to PLASMA scoring and need not launch here.
  solution = (np.array([[.6]]), np.zeros((1, 1, 1)), np.zeros((1, 1)), np.ones((1, 1)))
  monkeypatch.setattr(scenario, 'solve_snapshot', lambda *args: (*solution, 1.))
  monkeypatch.setattr(scenario, 'greedy_solve', lambda *args: solution)
  config = {'seed': 0, 'limits': {'load': {}}, 'max_steps': 1,
            'base_solution_folder': str(tmp_path),
            'solver_options': {'plasma': {'W': 1., 'rounds_per_step': 1, 'k_sb': 0}}}
  result = scenario.run_scenario(config)
  assert result['plasma'].tolist() == [-1.]
