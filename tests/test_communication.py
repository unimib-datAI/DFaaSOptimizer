"""Logical protocol accounting: batches, records and request quantities differ."""
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils import communication, step_time


def test_batched_messages_include_rejections_and_displaced_buyers(tmp_path):
  graph = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
  bids = pd.DataFrame({'i': [0, 0, 0, 2], 'j': [1, 1, 1, 1],
                       'f': [0, 0, 1, 0], 'd': [2, 3, 1, 4]})
  delta = np.zeros((3, 3, 2))
  delta[2, 0, 0] = -1  # displaced incumbent, even without a new bid
  with communication.session(True, 'MADEA-selective', tmp_path):
    communication.begin_step(0)
    communication.begin_round(0)
    communication.announce(graph, 2)
    communication.requests('capacity_request', bids)
    communication.capacity_replies(bids, delta)
    communication.requests('memory_request', [(0, 1, 0), (0, 1, 1)])
    communication.finish()
  rows = {r['kind']: r for r in csv.DictReader((tmp_path/'communication.csv').open())}
  assert [(rows[k]['messages'], rows[k]['items'], float(rows[k]['request_units'])) for k in rows] == [
    ('4', '12', 0.), ('2', '3', 10.), ('3', '4', 0.), ('1', '2', 0.)]
  meta = json.loads((tmp_path/'communication.json').read_text())
  assert meta['complete'] and meta['messages'] == 10 and meta['items'] == 21
  assert meta['model'] == 'dfaas-logical-batched-v1'


def test_unit_bids_are_compact_but_distinct_prices_and_rounds_remain(tmp_path):
  bids = pd.DataFrame({'i': [0]*1001, 'j': [1]*1001, 'f': [0]*1001,
                       'd': [1]*1001, 'b': [2.]*1000+[3.], 'utility': [4.]*1001})
  with communication.session(True, 'MADEA-one-shot-detailed', tmp_path):
    for round_index in range(2):
      communication.begin_round(round_index)
      communication.requests('capacity_request', bids)
    communication.finish()
  rows = list(csv.DictReader((tmp_path/'communication.csv').open()))
  assert len(rows) == 2
  assert all((r['messages'], r['items'], float(r['request_units'])) == ('1', '2', 1001.)
             for r in rows)
  meta = json.loads((tmp_path/'communication.json').read_text())
  assert meta['messages'] == 2 and meta['items'] == 4 and meta['request_units'] == 2002.


@pytest.mark.parametrize('module', ['run_faasmadea', 'decentralized_auction'])
@pytest.mark.parametrize('variant', ['detailed_init', 'selective_memory'])
@pytest.mark.parametrize('extra', [{'refine_welfare': True}, {'rules': {}}])
def test_metrics_reject_unaccounted_runner_combinations(module, variant, extra):
  from importlib import import_module
  with pytest.raises(ValueError, match='communication metrics unsupported'):
    import_module(module).run({'communication_metrics': True}, 0,
                             **{variant: True}, **extra)


def test_pg_counts_turn_queries_and_only_committed_updates(tmp_path):
  with communication.session(True, 'FaaS-MAPG-R-detailed', tmp_path):
    communication.begin_step(3)
    communication.begin_round(2)
    communication.pg_visit(0, {1, 2}, 2)
    old = np.zeros((3, 2))
    new = old.copy(); new[1] = [2, 1]
    communication.pg_commit(0, old, new)
    communication.pg_commit(0, new, new)
    communication.finish()
  rows = {r['kind']: r for r in csv.DictReader((tmp_path/'communication.csv').open())}
  assert {k: int(v['messages']) for k,v in rows.items()} == {
    'pg_turn': 1, 'pg_state_query': 2, 'pg_state_reply': 2, 'pg_commit': 1}
  assert rows['pg_commit']['items'] == '2'
  assert rows['pg_commit']['request_units'] == '3.0'
  assert all(r['step']=='3' and r['round']=='2' for r in rows.values())
  meta = json.loads((tmp_path/'communication.json').read_text())
  assert meta['data_messages'] == 5 and meta['control_messages'] == 1


def test_observer_time_is_excluded_and_failure_does_not_leak(tmp_path, monkeypatch):
  now = [0.]
  monkeypatch.setattr(step_time, 'perf_counter', lambda: now[0])
  def slow_rows():
    now[0] += 10
    yield (0, 1, 0, 2)
  with pytest.raises(RuntimeError, match='failed'):
    with communication.session(True, 'Greedy-RR', tmp_path):
      communication.begin_step(0)
      clock = step_time.start_step()
      communication.requests('capacity_request', slow_rows())
      assert step_time.end_step(clock)['algorithm_s'] == 0
      raise RuntimeError('failed')
  assert not json.loads((tmp_path/'communication.json').read_text())['complete']
  # Disabled/no active session must not even consume the payload.
  def forbidden():
    raise AssertionError('observer executed while disabled')
    yield
  communication.requests('capacity_request', forbidden())
  with communication.session(False, 'unsupported', tmp_path/'disabled'):
    communication.requests('capacity_request', forbidden())
  assert not (tmp_path/'disabled').exists()
  with pytest.raises(ValueError, match='unsupported'):
    with communication.session(True, 'unsupported', tmp_path/'unsupported'):
      pass


@pytest.mark.parametrize('name', sorted(communication.SUPPORTED))
def test_metrics_preserve_solutions_and_are_independent_of_workers(name, tmp_path):
  from copy import deepcopy
  from experiments.harmonization.capture_baseline import config_for, DATA
  from methods import HARMONIZED, SEQUENTIAL_METHODS
  source = next(p for p in sorted(DATA.iterdir()) if (p/'metadata.json').exists())
  run = {**HARMONIZED, **SEQUENTIAL_METHODS}[name]
  config = config_for(source, tmp_path, steps=2)
  config['max_iterations'] = 3
  config['solver_options']['general']['algorithm_time_limit'] = 10000
  folders = []
  for enabled, workers in [(False, 0), (True, 0), (True, 2)]:
    config['communication_metrics'] = enabled
    original = deepcopy(config)
    folder = Path(run(config, workers, log_on_file=True, disable_plotting=True))
    assert config == original
    folders.append(folder)
    if enabled:
      meta = json.loads((folder/'communication.json').read_text())
      rows = pd.read_csv(folder/'communication.csv')
      assert meta['complete'] and meta['messages'] == rows.messages.sum() > 0
      assert set(rows.step) == {0, 1}
      assert rows['round'].between(0, 2).all()
  assert not (folders[0]/'communication.csv').exists()
  pd.testing.assert_frame_equal(pd.read_csv(folders[1]/'communication.csv'),
                                pd.read_csv(folders[2]/'communication.csv'))
  for filename in ['obj.csv', 'LSPc_solution.csv', 'LSPc_replicas.csv', 'LSPc_detailed_fwd_solution.csv']:
    baseline = pd.read_csv(folders[0]/filename)
    for folder in folders[1:]:
      pd.testing.assert_frame_equal(baseline, pd.read_csv(folder/filename))
