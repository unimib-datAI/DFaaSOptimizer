"""Hierarchy protocol: snapshots, shared ledgers, rejected bids and commits."""
import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from utils import communication, step_time
from hierarchical_auction.types import TokenRequest, AcceptedAllocation


def request(root, buyer, seller, function=0):
  return TokenRequest(2, root, buyer, seller, function, 2, 1., 2.)


def test_hierarchy_counts_overlaps_rejections_and_shared_demand(tmp_path):
  structures = {
    0: SimpleNamespace(member_nodes={0, 1}, adjacent_structures={1}),
    1: SimpleNamespace(member_nodes={1, 2}, adjacent_structures={0}),
  }
  offers = [request(0, 0, 2), request(1, 0, 2, 1), request(1, 0, 1)]
  accepted = [AcceptedAllocation(2, 1, 0, 1, 0, 1, 1., 1.)]
  with communication.session(True, 'hierarchical-madea-detailed', tmp_path):
    communication.hierarchy_state(structures, 3, 2)
    communication.hierarchy_requests(offers, 3, 2)
    communication.hierarchy_results(offers, accepted)
    communication.finish()
  rows = {r['kind']: (int(r['messages']), int(r['items']), float(r['request_units']))
          for r in csv.DictReader((tmp_path / 'communication.csv').open())}
  assert rows == {
    'hierarchy_state': (4, 8, 0.),
    'hierarchy_role_handoff': (1, 6, 0.),
    'hierarchy_request': (2, 2, 4.),  # root 1 -> seller 1 is local
    'hierarchy_demand_state': (1, 2, 0.),  # buyer 0 -> first active seller 1
    'hierarchy_demand_handoff': (1, 6, 0.),  # seller 1 -> seller 2
    'hierarchy_reply': (2, 2, 0.),  # both rejected remote offers still get replies
    'hierarchy_commit': (1, 1, 1.),
  }
  meta = json.loads((tmp_path / 'communication.json').read_text())
  assert meta['complete'] and meta['messages'] == 12


def test_hierarchy_observers_are_untimed_and_disabled_is_noop(tmp_path, monkeypatch):
  class Forbidden:
    def __iter__(self):
      raise AssertionError('consumed disabled observer input')
  communication.hierarchy_requests(Forbidden(), 3, 2)
  now = [0.]
  monkeypatch.setattr(step_time, 'perf_counter', lambda: now[0])
  class SlowOffers:
    def __iter__(self):
      now[0] += 10
      yield request(0, 0, 1)
  with communication.session(True, 'hierarchical-madea-detailed', tmp_path):
    clock = step_time.start_step()
    communication.hierarchy_requests(SlowOffers(), 3, 2)
    assert step_time.end_step(clock)['algorithm_s'] == 0


def test_iterative_engine_counts_real_repeated_rounds_without_changing_flows(tmp_path):
  from test_hierarchical_level_iterations import _engine, _inputs
  engine = _engine(depth=10)
  expected = engine.run_higher_levels(**_inputs())
  with communication.session(True, 'hierarchical-madea-detailed', tmp_path):
    actual = engine.run_higher_levels(**_inputs())
    communication.finish()
  np.testing.assert_array_equal(actual.y, expected.y)
  np.testing.assert_array_equal(actual.omega, expected.omega)
  assert actual.accepted_allocations == expected.accepted_allocations
  rows = {r['kind']: int(r['messages']) for r in csv.DictReader((tmp_path / 'communication.csv').open())}
  # Four executed rounds, including the final unproductive rounds (2,2,2,3).
  assert rows['hierarchy_role_handoff'] == 4 * 8
  assert rows['hierarchy_state'] > 0 and rows['hierarchy_request'] > 0
  assert rows['hierarchy_reply'] > 0 and rows['hierarchy_commit'] > 0


def test_no_higher_level_exchange_when_there_is_no_residual_demand(tmp_path):
  from test_hierarchical_level_iterations import _engine, _inputs
  with communication.session(True, 'hierarchical-madea-detailed', tmp_path):
    _engine().run_higher_levels(**_inputs((0., 0., 0., 0., 0.)))
    communication.finish()
  assert json.loads((tmp_path / 'communication.json').read_text())['messages'] == 0
