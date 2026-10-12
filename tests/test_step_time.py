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
    now.t = 4.5                     # 0.5 s of phase set-up/wrap-up outside tasks: sequential
  now.t = 5.0                       # 0.5 s sequential
  assert clock.modelled() == pytest.approx(1.0 + 2.0 + 0.5 + 0.5)
  assert step_time.end_step(clock) == {"algorithm_s": pytest.approx(4.0), "wall_s": 5.0,
                                       "parallel_phases": 1, "workers": 0}


def test_tasks_of_one_node_add_up_and_pool_records_match_in_process(now):
  clock = step_time.start_step()
  with step_time.parallel_phase() as phase:
    with step_time.offloaded(workers=3):
      phase.add(7, 1.0)
      step_time.record(7, 2.0)      # same node twice in the phase: 3 s
      step_time.record(8, 2.5)
      now.t = 4.0                   # pool wall time, transfers included
    now.t = 4.5                     # merging the results: sequential
  assert clock.modelled() == pytest.approx(3.0 + 0.5)
  assert step_time.end_step(clock)["workers"] == 3   # the pool size actually used


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


def test_solve_subproblem_is_one_phase_with_one_task_per_agent(monkeypatch):
  import run_faasmacro

  durations = {0: 0.02, 1: 0.05, 2: 0.01}

  def fake_solve(model, data, options, solver):
    import time
    agent = data[None]["whoami"][None] - 1
    time.sleep(durations[agent])
    return {"agent": agent}

  monkeypatch.setattr(run_faasmacro, "solve_agent_problem", fake_solve)
  monkeypatch.setattr(run_faasmacro, "merge_agents_solutions",
                      lambda *a, **k: (None,) * 6 + ({"tot": 0.0}, None, None))
  monkeypatch.setattr(run_faasmacro, "compute_utilization", lambda *a, **k: None)
  data = {None: {"Nn": {None: 3}, "Nf": {None: 1}}}
  clock = step_time.start_step()
  run_faasmacro.solve_subproblem(data, [0, 1, 2], None, "missing_solver", {}, 0)
  row = step_time.end_step(clock)
  assert row["parallel_phases"] == 1
  assert row["algorithm_s"] == pytest.approx(0.05, abs=0.02) and row["wall_s"] >= 0.08


def test_timed_loop_records_each_iteration_to_its_node_even_on_break(now):
  clock = step_time.start_step()
  with step_time.parallel_phase():
    for i, f in step_time.timed([(0, 0), (0, 1), (1, 0), (2, 0)], key=lambda item: item[0]):
      now.t += {0: 1.0, 1: 2.5, 2: 0.5}[i]
      if i == 2:
        break                         # last iteration still recorded
  assert clock.modelled() == pytest.approx(2.5)   # node 0: 2 s, node 1: 2.5 s, node 2: 0.5 s


def test_current_modelled_without_clock_is_zero():
  assert step_time.current_modelled() == 0.0


def test_plasma_arrival_order_is_seeded_and_worker_independent(tmp_path):
  import sys
  sys.path.insert(0, "experiments/harmonization")
  from capture_baseline import DATA, welfare
  from plasma import welfare as plasma_welfare
  instance = sorted(p for p in DATA.iterdir() if (p / "metadata.json").exists())[0]
  first = welfare(plasma_welfare.run, instance, 0)
  assert welfare(plasma_welfare.run, instance, 0) == first   # same seed, same result
  assert welfare(plasma_welfare.run, instance, 2) == first   # pool or not, same result


def test_plasma_arrival_order_is_a_seeded_permutation_per_round():
  from types import SimpleNamespace
  import numpy as np
  from plasma.engine import PlasmaEngine
  nodes = [SimpleNamespace(params=SimpleNamespace(nbrs=[])) for _ in range(6)]
  engine = PlasmaEngine(nodes, SimpleNamespace(), np.random.default_rng(5))
  expected = np.random.default_rng(5)
  assert engine._arrival_order() == [int(i) for i in expected.permutation(6)]
  assert engine._arrival_order() == [int(i) for i in expected.permutation(6)]


def test_plasma_batches_keep_arrival_order_between_competitors():
  from types import SimpleNamespace
  import numpy as np
  from plasma.engine import PlasmaEngine
  nbrs = {0: [1], 1: [0], 2: [3], 3: [2]}          # two disjoint pairs
  nodes = [SimpleNamespace(params=SimpleNamespace(nbrs=nbrs[i])) for i in range(4)]
  engine = PlasmaEngine(nodes, SimpleNamespace(), np.random.default_rng(0))
  assert engine._negotiation_batches([2, 0, 3, 1]) == [[2, 0], [3, 1]]


def test_phase_without_tasks_counts_its_full_wall_time(now):
  # A helper that forgets to time its per-node loop must err high, never vanish.
  clock = step_time.start_step()
  with step_time.parallel_phase():
    now.t = 3.0
  assert clock.modelled() == pytest.approx(3.0)


def test_hierarchy_seller_resolution_is_sequential_time(now):
  # Sellers resolve one after another against shared buyer demand and tokens:
  # their resolution time adds up, it is not one parallel phase.
  import numpy as np
  from types import SimpleNamespace
  from hierarchical_auction.engine import HierarchicalAuctionEngine
  engine = HierarchicalAuctionEngine(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]), 1, np.ones(1))
  engine._aggregate_residual_demand = lambda *a, **k: None
  engine._populate_indicative_tokens = lambda *a, **k: None
  engine._generate_level_requests = lambda *a, **k: []

  def resolve(k, f, remaining):
    now.t += 1.0
    return []

  tokens = SimpleNamespace(resolve_node_function=resolve, commit=lambda accepted: None)
  clock = step_time.start_step()
  engine._run_level_iteration({}, 2, tokens, np.zeros((3, 3, 1)), np.zeros((3, 1)),
                              np.zeros((3, 1)), np.zeros((3, 3)), np.zeros((3, 1)))
  assert clock.modelled() == pytest.approx(3.0)


def test_untimed_inside_a_phase_is_removed_from_tasks_and_overhead(now):
  clock = step_time.start_step()
  with step_time.parallel_phase():
    with step_time.task(0):
      now.t = 1.0                   # node 0: 1 s of work
      with step_time.untimed():
        now.t = 5.0                 # logging inside the task
    with step_time.untimed():
      now.t = 7.0                   # logging between tasks
  assert clock.modelled() == pytest.approx(1.0) and clock.wall() == 7.0


def test_timed_iteration_left_by_an_exception_stays_in_its_phase(now):
  step_time.start_step()
  loop = step_time.timed([0])
  with pytest.raises(ValueError):
    with step_time.parallel_phase():
      for _ in loop:
        now.t += 2.0
        raise ValueError
  with step_time.parallel_phase() as later:
    loop.close()                    # the generator finishes late, e.g. at garbage collection
  assert later.tasks == {}


def test_nested_untimed_counts_union(now):
  clock = step_time.start_step()
  now.t = 1.0                       # 1 s of work
  with step_time.untimed():         # outer observer: 5 s ...
    now.t = 2.0
    with step_time.untimed():       # ... containing an inner 2 s observer
      now.t = 4.0
    now.t = 6.0
  assert step_time.end_step(clock) == {"algorithm_s": pytest.approx(1.0), "wall_s": 6.0,
                                       "parallel_phases": 0, "workers": 0}


def test_nested_untimed_in_task(now):
  clock = step_time.start_step()
  with step_time.parallel_phase():
    with step_time.task(0):
      now.t = 1.0                   # node 0: 1 s of work
      with step_time.untimed():
        now.t = 2.0
        with step_time.untimed():
          now.t = 4.0
        now.t = 6.0
  row = step_time.end_step(clock)
  assert row["algorithm_s"] == pytest.approx(1.0) and row["wall_s"] == 6.0 and row["parallel_phases"] == 1


def test_untimed_spanning_a_phase_is_removed_once(now):
  # An observer may itself run per-node work (e.g. a parallel check): it all counts zero,
  # the phase's savings are not subtracted again on top of the observer interval.
  clock = step_time.start_step()
  now.t = 1.0
  with step_time.untimed():
    with step_time.parallel_phase():
      for node in (0, 1):
        with step_time.task(node):
          now.t += 2.0
          with step_time.untimed():
            now.t += 1.0
  now.t += 0.5
  row = step_time.end_step(clock)
  assert row["algorithm_s"] == pytest.approx(1.5) and row["wall_s"] == pytest.approx(7.5)


def test_untimed_exception_resets_state(now):
  step_time.start_step()
  with pytest.raises(ValueError):
    with step_time.parallel_phase():
      with step_time.task(0):
        with step_time.untimed():
          with step_time.untimed():
            raise ValueError
  clock = step_time.start_step()
  now.t += 3.0
  with step_time.untimed():         # counted again: no observer depth left behind
    now.t += 2.0
  assert step_time.end_step(clock)["algorithm_s"] == pytest.approx(3.0)


def test_cancelled_step_does_not_leak(now):
  clock = step_time.start_step()
  step_time.cancel_step(clock)
  assert not step_time.active() and step_time.current_modelled() == 0.0
  other = step_time.start_step()
  step_time.cancel_step(clock)      # stale handle: the current clock stays
  assert step_time.active()
  now.t = 3.0
  assert step_time.end_step(other)["algorithm_s"] == pytest.approx(3.0)
  assert not step_time.active()


def test_now_reads_the_module_clock(now):
  now.t = 42.0
  assert step_time.now() == 42.0


def test_run_timestamp_keeps_the_folder_format(monkeypatch):
  from datetime import datetime
  from utils import common

  class Fixed(datetime):
    @classmethod
    def now(cls, tz=None):
      return cls(2026, 1, 2, 3, 4, 5, 6)

  monkeypatch.setattr(common, "datetime", Fixed)
  assert common.run_timestamp() == "2026-01-02_03-04-05.000006"
