from pathlib import Path

import pytest

from remote_experiments.results_reader import (
  read_run_objective, read_run_runtime, run_failed,
)


def _make_run(tmp_path: Path, obj_rows, runtime_rows=None) -> Path:
  run = tmp_path / "exp-id" / "2026-07-06_00-00-00.000000"
  run.mkdir(parents=True)
  if obj_rows is not None:
    (run / "obj.csv").write_text("Model\n" + "\n".join(str(v) for v in obj_rows) + "\n")
  if runtime_rows is not None:
    (run / "runtime.csv").write_text("tot\n" + "\n".join(str(v) for v in runtime_rows) + "\n")
  return tmp_path / "exp-id"


def test_read_objective_is_column_mean(tmp_path):
  run = _make_run(tmp_path, [10.0, 20.0, 30.0])
  assert read_run_objective(run) == 20.0


def test_read_runtime_is_column_mean(tmp_path):
  run = _make_run(tmp_path, [1.0], runtime_rows=[0.2, 0.4])
  assert read_run_runtime(run) == pytest.approx(0.3)


def test_missing_obj_is_failure(tmp_path):
  run = _make_run(tmp_path, None)
  assert read_run_objective(run) is None
  assert run_failed(run) is True


def test_present_obj_is_not_failure(tmp_path):
  run = _make_run(tmp_path, [5.0])
  assert run_failed(run) is False


def test_read_run_runtime_uses_algorithm_time_of_the_new_format(tmp_path):
  # runtime.csv of harmonized methods: step, algorithm_s, wall_s, parallel_phases, workers
  run = tmp_path / "run"
  run.mkdir()
  (run / "runtime.csv").write_text(
    "step,algorithm_s,wall_s,parallel_phases,workers\n0,1.0,9.0,3,4\n1,3.0,9.0,3,4\n")
  assert read_run_runtime(run) == pytest.approx(2.0)


@pytest.mark.parametrize("text, expected", [
  ("step,algorithm_s,wall_s,parallel_phases,workers\n0,1.0,9.0,0,1\n1,3.0,9.0,0,1\n", 2.0),  # not step
  ("tot\n1.0\n3.0\n", 2.0),
  ("LoadManagementModel\n1.0\n3.0\n", 2.0),
  ("TightLoadManagementModel\n1.0\n3.0\n", 2.0),
  ("SelfishLoadManagementModel\n1.0\n3.0\n", 2.0),
  ("", None),
])
def test_read_run_runtime_every_format(tmp_path, text, expected):
  run = tmp_path / "run"
  run.mkdir()
  (run / "runtime.csv").write_text(text)
  assert read_run_runtime(run) == (pytest.approx(expected) if expected is not None else None)


def test_read_run_runtime_missing_file(tmp_path):
  (tmp_path / "run").mkdir()
  assert read_run_runtime(tmp_path / "run") is None
