import pandas as pd
import pytest

from utils import step_time


class FakeClock:
  def __init__(self):
    self.t = 0.0

  def __call__(self):
    return self.t


@pytest.fixture(autouse=True)
def no_clock():
  step_time._clock.set(None)  # context variables survive between tests in one thread
  yield
  step_time._clock.set(None)


@pytest.fixture
def now(monkeypatch):
  fake = FakeClock()
  monkeypatch.setattr(step_time, "perf_counter", fake)
  return fake


def test_sequential_time_is_wall_time(now):
  clock = step_time.start_step()
  now.t = 5.0
  assert clock.modelled() == 5.0 and clock.wall() == 5.0


def test_phase_counts_its_longest_task(now):
  clock = step_time.start_step()
  now.t = 1.0                       # 1 s sequential
  with step_time.parallel_phase():
    with step_time.task(0):
      now.t = 3.0                   # node 0: 2 s
    with step_time.task(1):
      now.t = 4.0                   # node 1: 1 s
    now.t = 4.5                     # 0.5 s of phase overhead (not a task)
  now.t = 5.0                       # 0.5 s sequential
  assert clock.modelled() == pytest.approx(1.0 + 2.0 + 0.5)
  assert step_time.end_step(clock, 0) == {"algorithm_s": pytest.approx(3.5), "wall_s": 5.0,
                                          "parallel_phases": 1, "workers": 0}


def test_tasks_of_one_node_add_up_and_pool_records_match_in_process(now):
  clock = step_time.start_step()
  with step_time.parallel_phase() as phase:
    phase.add(7, 1.0)
    step_time.record(7, 2.0)        # same node twice in the phase: 3 s
    step_time.record(8, 2.5)
    now.t = 4.0                     # pool wall time, transfers included
  assert clock.modelled() == pytest.approx(3.0)


def test_uniform_phase_divides_wall_by_nodes(now):
  clock = step_time.start_step()
  with step_time.parallel_phase() as phase:
    phase.uniform(4)
    now.t = 8.0
  assert clock.modelled() == pytest.approx(2.0)


def test_untimed_is_removed_outside_phases(now):
  clock = step_time.start_step()
  now.t = 1.0
  with step_time.untimed():
    now.t = 3.0
  assert clock.modelled() == pytest.approx(1.0) and clock.wall() == 3.0


def test_nested_phase_raises():
  step_time.start_step()
  with step_time.parallel_phase():
    with pytest.raises(RuntimeError, match="nested"):
      with step_time.parallel_phase():
        pass


def test_decorator_and_no_clock_are_safe(now):
  @step_time.parallel
  def work():
    with step_time.task(0):
      now.t += 2.0
    return "done"

  assert work() == "done"           # no clock: runs, records nothing
  clock = step_time.start_step()
  work()
  assert clock.modelled() == pytest.approx(2.0)


def test_budget_workers_and_csv(tmp_path):
  assert step_time.algorithm_time_limit({}) == float("inf")
  assert step_time.algorithm_time_limit({"solver_options": {"general": {"algorithm_time_limit": 300}}}) == 300.0
  assert step_time.workers(3) == 3 and step_time.workers(-1) >= 1
  step_time.write_runtime(str(tmp_path), [dict(step=0, algorithm_s=1.0, wall_s=2.0, parallel_phases=3, workers=2)])
  frame = pd.read_csv(tmp_path / "runtime.csv")
  assert tuple(frame.columns) == step_time.COLUMNS and frame.iloc[0].algorithm_s == 1.0
