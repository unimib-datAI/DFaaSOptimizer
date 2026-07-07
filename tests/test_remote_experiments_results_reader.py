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
