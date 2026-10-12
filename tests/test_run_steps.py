"""Simulated steps and checkpoints, shared by every runner (utils/common.py)."""
from utils.common import checkpoint_due, run_steps


def test_steps_are_min_to_max_run_time_excluded():
  assert list(run_steps({"max_steps": 100, "min_run_time": 0, "max_run_time": 10})) == list(range(10))


def test_equal_min_and_max_run_time_run_that_single_step():
  assert list(run_steps({"max_steps": 100, "min_run_time": 5, "max_run_time": 5})) == [5]


def test_defaults_run_every_step_one_by_one():
  assert list(run_steps({"max_steps": 3})) == [0, 1, 2]


def test_run_time_step_strides():
  assert list(run_steps({"max_steps": 10, "max_run_time": 7, "run_time_step": 3})) == [0, 3, 6]


def test_last_simulated_step_is_always_checkpointed():
  # 10 steps, interval 4: 0, 4, 8 by the interval, 9 because it is the last one
  steps = run_steps({"max_steps": 100, "max_run_time": 10})
  assert [t for t in steps if checkpoint_due(t, steps, 4)] == [0, 4, 8, 9]
