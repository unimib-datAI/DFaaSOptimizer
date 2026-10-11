"""Contract of the harmonized methods (registry, runtime.csv, budget, regression)."""
import inspect
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments" / "harmonization"), str(ROOT)]

import methods  # noqa: E402
from utils import step_time  # noqa: E402

BASELINE = json.loads((ROOT / "tests/data/harmonization/baseline.json").read_text())


def test_registry_covers_the_baseline_and_excludes_sequential_methods():
  assert set(methods.HARMONIZED) == set(BASELINE)
  assert {"FaaS-MAPG-S", "FaaS-MAPG-R", "faas-br-s", "MADEA-PG", "MAFW-PG"} <= set(methods.EXCLUDED)
  assert not set(methods.HARMONIZED) & set(methods.EXCLUDED)
  for name, runner in methods.HARMONIZED.items():
    params = list(inspect.signature(runner).parameters)
    assert params[:2] == ["config", "parallelism"], name


def test_benchmark_summary_reads_the_new_format(tmp_path):
  import benchmark_summary
  step_time.write_runtime(str(tmp_path), [dict(step=0, algorithm_s=1.5, wall_s=9.0, parallel_phases=2, workers=4),
                                          dict(step=1, algorithm_s=2.5, wall_s=9.0, parallel_phases=2, workers=4)])
  assert benchmark_summary._single_column_sum(str(tmp_path / "runtime.csv")) == pytest.approx(4.0)


from capture_baseline import DATA, config_for, welfare  # noqa: E402

INSTANCES = sorted(p for p in DATA.iterdir() if (p / "metadata.json").exists())
NAMES = sorted(methods.HARMONIZED)


def run_folder(name, instance, parallelism, tmp, **general):
  config = config_for(instance, Path(tmp))
  config["solver_options"]["general"].update(general)
  return Path(methods.HARMONIZED[name](config, parallelism, log_on_file=True, disable_plotting=True))


@pytest.mark.parametrize("name", NAMES)
def test_runtime_contract_and_worker_independence(name, tmp_path):
  instance = INSTANCES[0]
  for parallelism in (0, 2):
    folder = run_folder(name, instance, parallelism, tmp_path / str(parallelism))
    frame = pd.read_csv(folder / "runtime.csv")
    assert tuple(frame.columns) == step_time.COLUMNS and len(frame) == 2
    assert (frame.algorithm_s > 0).all() and (frame.algorithm_s <= frame.wall_s + 1e-3).all()
    assert (frame.parallel_phases >= 1).all() and (frame.workers == parallelism).all()
    got = [float(v) for v in pd.read_csv(folder / "obj.csv").iloc[:, -1]]
    assert got == pytest.approx(BASELINE[name][instance.name], abs=1e-6), (name, parallelism)


@pytest.mark.parametrize("name", NAMES)
def test_regression_on_every_instance(name):
  for instance in INSTANCES[1:]:
    assert welfare(methods.HARMONIZED[name], instance, 2) == pytest.approx(
      BASELINE[name][instance.name], abs=1e-6)


ITERATIVE = [n for n in NAMES if not n.startswith("PLASMA") and "one-shot" not in n]


@pytest.mark.parametrize("name", ITERATIVE)
def test_algorithm_time_limit_stops_every_iterative_method(name, tmp_path):
  folder = run_folder(name, INSTANCES[0], 0, tmp_path, algorithm_time_limit=1e-9, TimeLimit=1e9)
  reasons = (folder / "termination_condition.csv").read_text().lower()
  assert "time limit" in reasons, name
  # a tiny budget allows at most the first round: welfare is defined and not above the unbounded run
  got = [float(v) for v in pd.read_csv(folder / "obj.csv").iloc[:, -1]]
  assert all(g <= b + 1e-6 for g, b in zip(got, BASELINE[name][INSTANCES[0].name]))


def test_observer_time_is_excluded(tmp_path, monkeypatch):
  import time
  import run_faasmadea
  original = run_faasmadea.check_feasibility

  def slow(*args, **kwargs):
    time.sleep(0.2)
    return original(*args, **kwargs)

  monkeypatch.setattr(run_faasmadea, "check_feasibility", slow)
  folder = run_folder("MADEA", INSTANCES[0], 0, tmp_path)
  frame = pd.read_csv(folder / "runtime.csv")
  assert (frame.wall_s - frame.algorithm_s >= 0.2).all()
