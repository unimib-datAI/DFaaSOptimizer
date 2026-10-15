"""A budget stops between complete rounds, using the common modelled clock."""
import numpy as np
import pytest

from utils import step_time
from test_plasma_clock import _line_engine
from test_plasma_welfare import _node
from plasma.welfare import WelfareEngine, WelfareOptions


def engine_for(welfare):
  if not welfare:
    return _line_engine()[0]
  return WelfareEngine([_node(0, (1,)), _node(1, (0,))],
                       WelfareOptions(), np.random.default_rng(7))


@pytest.fixture
def clock(monkeypatch):
  time = [0.]
  monkeypatch.setattr(step_time, 'perf_counter', lambda: time[0])
  step_time._clock.set(None)
  yield time
  step_time._clock.set(None)


@pytest.mark.parametrize('welfare', [False, True])
@pytest.mark.parametrize('cost,limit,expected,reached', [
  (.1, .25, 3, True), (.125, .25, 2, True),
  (.125, float('inf'), 5, False), (.125, 0., 1, True),
  (.125, .625, 5, True),
])
def test_round_budget(clock, monkeypatch, welfare, cost, limit, expected, reached):
  engine = engine_for(welfare)
  original = engine._send_heartbeats
  def send(round_):
    original(round_)
    clock[0] += cost
  monkeypatch.setattr(engine, '_send_heartbeats', send)
  active = step_time.start_step()
  res = engine.run_rounds(5, np.full((2, 2 if welfare else 1), 8), time_limit=limit)
  assert res.completed_rounds == engine.clock.round == expected
  assert res.time_limit_reached is reached
  np.testing.assert_array_equal(res.x + res.z + res.y.sum(axis=1), 8)
  assert step_time.end_step(active)['algorithm_s'] == pytest.approx(cost * expected)


def test_welfare_initial_plans_consume_budget(clock, monkeypatch):
  import plasma.welfare as welfare
  original = welfare._solve_plan
  def costly(job):
    clock[0] += .5
    return original(job)
  monkeypatch.setattr(welfare, '_solve_plan', costly)
  engine = engine_for(True)
  active = step_time.start_step()
  res = engine.run_rounds(5, np.full((2, 2), 8), time_limit=.25)
  assert res.completed_rounds == 1 and res.time_limit_reached
  assert step_time.end_step(active)['algorithm_s'] == pytest.approx(.5)


@pytest.mark.parametrize('welfare', [False, True])
def test_budget_is_checked_after_parallel_phase(clock, monkeypatch, welfare):
  engine = engine_for(welfare)
  def send(round_):
    with step_time.parallel_phase():
      with step_time.task(0):
        clock[0] += .125
      with step_time.task(1):
        clock[0] += .0625
  monkeypatch.setattr(engine, '_send_heartbeats', send)
  active = step_time.start_step()
  res = engine.run_rounds(5, np.full((2, 2 if welfare else 1), 8), time_limit=.25)
  assert res.completed_rounds == 2
  assert step_time.end_step(active)['algorithm_s'] == .25


@pytest.mark.parametrize('welfare', [False, True])
def test_nonbinding_budget_preserves_state_and_resume(clock, welfare):
  a, b = engine_for(welfare), engine_for(welfare)
  arrivals = np.full((2, 2 if welfare else 1), 8)
  for rounds in (2, 3):
    ra = a.run_rounds(rounds, arrivals)
    rb = b.run_rounds(rounds, arrivals, time_limit=1.)  # no active StepClock
    assert not rb.time_limit_reached
    for field in ('x', 'y', 'z', 'r', 'xi'):
      np.testing.assert_array_equal(getattr(ra, field), getattr(rb, field))
    assert a.msg_count == b.msg_count and a.hb_count == b.hb_count
    assert a.rng.bit_generator.state == b.rng.bit_generator.state
    assert [(r, seq) for r, seq, _ in a.clock._queue] == [(r, seq) for r, seq, _ in b.clock._queue]
    for na, nb in zip(a.nodes, b.nodes):
      assert na.rng.bit_generator.state == nb.rng.bit_generator.state


@pytest.mark.parametrize('welfare', [False, True])
def test_interrupted_engine_resumes_without_skipping_events(clock, monkeypatch, welfare):
  engine = engine_for(welfare)
  deliveries = []
  engine.clock.schedule(2, lambda: deliveries.append(2))
  arrivals = np.full((2, 2 if welfare else 1), 8)
  active = step_time.start_step()
  assert engine.run_rounds(5, arrivals, time_limit=0).completed_rounds == 1
  assert deliveries == []
  step_time.end_step(active)
  active = step_time.start_step()
  assert engine.run_rounds(3, arrivals, time_limit=float('inf')).completed_rounds == 3
  assert deliveries == [2] and engine.clock.round == 4
  step_time.end_step(active)


@pytest.mark.parametrize('welfare', [False, True])
def test_runner_budget_metrics_and_parser(tmp_path, welfare):
  import pandas as pd
  from pathlib import Path
  from test_plasma_e2e import _config
  from plasma.runner import run
  from run import load_termination_condition
  config = _config(tmp_path, max_steps=2)
  config['solver_options']['general'] = {'algorithm_time_limit': 0.}
  if welfare:
    config['solver_options']['plasma_welfare'] = {'W': .5, 'rounds_per_step': 5}
  else:
    config['solver_options']['plasma']['W'] = .5
  folder = Path(run(config, 0, disable_plotting=True, welfare=welfare))
  term = pd.read_csv(folder / 'termination_condition.csv')
  values = term.iloc[:, -1].tolist()
  assert len(values) == 2
  assert all(s.startswith('reached time limit:') and '(it: 1;' in s for s in values)
  parsed = load_termination_condition(str(folder))
  assert parsed is not None
  msgs = pd.read_csv(folder / 'plasma_messages.csv')
  # All three alive nodes send each of the four directed tree heartbeats once.
  np.testing.assert_allclose(msgs.hb_per_node_s, 4 / (3 * .5))
  if welfare:
    assert pd.read_csv(folder / 'plasma_welfare.csv')['rounds'].tolist() == [1, 1]
  assert pd.read_csv(folder / 'runtime.csv').shape == (2, 5)
