"""Parallel local plans must preserve the sequential neighbor transactions."""
from copy import deepcopy
from pathlib import Path
import json
import multiprocessing as mp

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import run_faasmacro as macro
from plasma.core.node import NodeParams
from plasma.welfare import WelfareEngine, WelfareNode, WelfareOptions, run


@pytest.mark.parametrize('seed,loss,W', [(7, 0., 1.), (42, .5, 2.), (4850, 1., 1.)])
def test_parallel_rounds_match_serial_with_changing_load_and_node_failures(seed, loss, W, monkeypatch):
  # Adjacent integer labels must include independent receivers, not only a chain.
  order = [0, 3, 1, 4, 2, 5, 6, 9, 7, 10, 8, 11]
  graph = nx.relabel_nodes(nx.circular_ladder_graph(6),
                          {old: new for new, old in enumerate(order)})
  assert nx.check_planarity(graph)[0]
  rng = np.random.default_rng(seed)
  opts = WelfareOptions(W=W, hb_loss=loss, rounds_per_step=5)
  nodes = []
  for i in range(12):
    neighbors = tuple(sorted(graph[i]))
    params = NodeParams(i, neighbors, rng.uniform(0, 5, 3), rng.uniform(0, 2, 3),
      rng.uniform(0, 8, (3, 3)), rng.integers(2, 6, 3), 4., np.ones(3))
    nodes.append(WelfareNode(params, opts, np.random.default_rng(i)))
  serial = WelfareEngine(deepcopy(nodes), opts, np.random.default_rng(seed))
  parallel = WelfareEngine(deepcopy(nodes), opts, np.random.default_rng(seed), parallelism=2)
  parallel_batches = []
  real_pool = mp.Pool
  def create_pool(*args, **kwargs):
    pool = real_pool(*args, **kwargs)
    real_map = pool.map
    def solve(function, jobs, **kwargs):
      if any(reservations for _, reservations in jobs):
        parallel_batches.append(len(jobs))
      return real_map(function, jobs, **kwargs)
    pool.map = solve
    return pool
  monkeypatch.setattr(macro.mpp, 'Pool', create_pool)
  children = {child.pid for child in mp.active_children()}
  with macro.parallel_solver_session():
    workers = None
    for epoch in range(3):
      arrivals = rng.integers(0, 20, (12, 3))
      serial.set_alive(0, epoch != 1)
      parallel.set_alive(0, epoch != 1)
      expected = serial.run_rounds(5, arrivals)
      actual = parallel.run_rounds(5, arrivals)
      for field in ('x', 'y', 'z', 'r', 'xi'):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
      assert parallel.accepted_trades == serial.accepted_trades
      assert (parallel.msg_count, parallel.hb_count) == (serial.msg_count, serial.hb_count)
      assert all(node.reservation is None for node in parallel.nodes)
      current = {child.pid for child in mp.active_children()} - children
      assert len(current) == 2
      if workers is not None:
        assert current == workers
      workers = current
  assert {child.pid for child in mp.active_children()} == children
  if loss < 1:
    assert parallel_batches and min(parallel_batches) >= 2


def test_planar_runner_reuses_auto_pool_and_exports_identical_solutions(tmp_path, monkeypatch):
  from remote_experiments.batch import Experiment
  from remote_experiments.instances import materialize_instance, load_materialized_instance
  from postprocessing import load_solution
  from run_centralized_model import encode_solution, get_current_load, update_data
  from utils.centralized import validate_centralized_solution
  config = json.loads((Path(__file__).resolve().parents[1] /
                       'experiments/madea_pg_compact/config.json').read_text())
  config.update(seed=7, max_steps=2, max_run_time=2, run_time_step=1,
                base_solution_folder=str(tmp_path / 'runs'))
  config['limits']['Nn'] = {'min': 10, 'max': 10}
  config['limits']['Nf'] = {'min': 3, 'max': 3}
  config['solver_options']['plasma_welfare'] = {'rounds_per_step': 5}
  source = tmp_path / 'instance'
  materialize_instance(Experiment('plasma-pool', 'plasma-pool', 'plasma-welfare',
                                 7, {}, {}, config), source)
  data, traces, agents, graph = load_materialized_instance(source)
  assert nx.check_planarity(graph)[0] and nx.is_connected(graph)
  config['limits'].update(instance_type='materialized', path=str(source))
  pools = []
  real_pool = mp.Pool
  def create_pool(*args, **kwargs):
    pool = real_pool(*args, **kwargs)
    pools.append(pool)
    return pool
  monkeypatch.setattr(macro.os, 'sched_getaffinity', lambda _: {0, 1}, raising=False)
  monkeypatch.setattr(macro.mpp, 'Pool', create_pool)
  sequential = Path(run(deepcopy(config), 0, log_on_file=True, disable_plotting=True))
  assert not pools
  parallel = Path(run(deepcopy(config), -1, log_on_file=True, disable_plotting=True))
  assert len(pools) == 1 and len(pools[0]._pool) == 2
  assert all(not child.is_alive() for child in pools[0]._pool)
  for filename in ('obj.csv', 'plasma_messages.csv', 'plasma_welfare.csv'):
    pd.testing.assert_frame_equal(pd.read_csv(sequential / filename),
                                  pd.read_csv(parallel / filename))
  serial = load_solution(str(sequential), 'LSPc')
  actual = load_solution(str(parallel), 'LSPc')
  for step in range(2):
    expected = encode_solution(10, 3, serial[0], serial[2], serial[1], step)
    arrays = encode_solution(10, 3, actual[0], actual[2], actual[1], step)
    current = update_data(data, {'incoming_load': get_current_load(traces, list(agents), step)})
    validate_centralized_solution(*arrays[:4], current)
    for result, oracle in zip(arrays, expected):
      np.testing.assert_array_equal(result, oracle)


def test_worker_failure_closes_pool():
  params = NodeParams(0, (), np.ones(1), np.ones(1), np.zeros((0, 1)),
                      np.ones(1), 2., np.array([.5]))
  opts = WelfareOptions()
  engine = WelfareEngine([WelfareNode(params, opts, np.random.default_rng(0))],
                          opts, np.random.default_rng(0), parallelism=2)
  children = {child.pid for child in mp.active_children()}
  with pytest.raises(ValueError, match='positive integer'):
    engine.run_rounds(2, np.array([[4]]))
  assert {child.pid for child in mp.active_children()} == children


def test_failed_parallel_negotiation_releases_offers_without_moving_traffic():
  opts = WelfareOptions()
  nodes = []
  # Two independent receivers compete on a connected planar path: 0-2-3-1.
  for i, neighbors in enumerate(((2,), (3,), (0, 3), (1, 2))):
    params = NodeParams(i, neighbors, np.zeros(1), np.ones(1),
      np.full((len(neighbors), 1), 2.), np.full(1, 10.), 1. if i < 2 else 0., np.ones(1))
    node = WelfareNode(params, opts, np.random.default_rng(i))
    node.start_window(np.array([0 if i < 2 else 10]), 0)
    nodes.append(node)
  for node in nodes[:2]:
    source = nodes[node.params.nbrs[0]]
    node.on_heartbeat(source.make_heartbeat(), 0)
    # A genuine worker-side DP validation failure, after offers have been locked.
    node.params.ram_req[:] = .5
  engine = WelfareEngine(nodes, opts, np.random.default_rng(0), parallelism=2)
  children = {child.pid for child in mp.active_children()}
  with pytest.raises(ValueError, match='positive integer'):
    with macro.parallel_solver_session() as state:
      state['pool'] = mp.Pool(processes=2)
      state['workers'] = 2
      engine._parallel_round(0, state['pool'])
  assert {child.pid for child in mp.active_children()} == children
  assert all(node.reservation is None for node in nodes)
  assert [node.z.tolist() for node in nodes] == [[0], [0], [10], [10]]
  assert all(not node.y.any() and not node.incoming.any() for node in nodes)
