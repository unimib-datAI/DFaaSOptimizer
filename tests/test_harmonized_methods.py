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
    pool = 0 if name == "PLASMA" else parallelism   # PLASMA negotiates in-process, no pool
    assert (frame.parallel_phases >= 1).all() and (frame.workers == pool).all()
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


def test_run_py_runtime_comparison_reads_the_new_format():
  import run
  frame = pd.DataFrame({"step": [0, 1], "algorithm_s": [1.0, 2.0], "wall_s": [5.0, 5.0]})
  assert run.runtime_series(frame, "FaaS-MADeA").tolist() == [1.0, 2.0]
  assert run.runtime_series(pd.DataFrame({"tot": [3.0]}), "FaaS-MAPG").tolist() == [3.0]  # excluded methods
  assert run.runtime_series(pd.DataFrame({"LoadManagementModel": [4.0]}), "LoadManagementModel").tolist() == [4.0]
  five = pd.DataFrame({"step": [0, 1], "algorithm_s": [4.0, 5.0], "wall_s": [9.0, 9.0],
                       "parallel_phases": [0, 0], "workers": [1, 1]})
  assert run.runtime_series(five, "LoadManagementModel").tolist() == [4.0, 5.0]   # new centralized file


def test_families_harness_sets_the_modelled_budget(tmp_path, monkeypatch):
  sys.path.insert(0, str(ROOT / "experiments" / "madea_pg_compact"))
  import families_extended as families
  from datetime import datetime, timedelta, timezone
  seen = {}

  def runner(cfg, workers, **kwargs):
    seen.update(general=cfg["solver_options"]["general"], workers=workers)
    raise RuntimeError("stop after capturing the configuration")

  instance = sorted(p for p in (ROOT / "tests/data/harmonization").iterdir() if (p / "metadata.json").exists())[0]
  case = dict(id="c", phase="static", nodes=300, functions=5, seed=1, repeat=0, profile=[1.0], steps=1)
  monkeypatch.setattr(families, "cases", lambda: [case])
  monkeypatch.setattr(families, "instance", lambda config, case, output: (instance, None, None, None))
  monkeypatch.setattr(families, "METHODS", [("M", runner)])
  config = json.loads((ROOT / "experiments/madea_pg_compact/catalog_config_overnight.json").read_text())
  families.run(tmp_path, datetime.now(timezone.utc) + timedelta(hours=1), -1, 1, config=config)
  assert seen["general"]["algorithm_time_limit"] == 150.0   # max(30, 0.5 * 300) modelled seconds
  assert seen["workers"] == -1


@pytest.mark.parametrize("text, expected", [
  ("step,algorithm_s,wall_s,parallel_phases,workers\n0,1.5,9.0,0,1\n1,2.5,9.0,0,1\n", 4.0),  # new centralized
  ("tot\n1.0\n2.0\n", 3.0),
  ("LoadManagementModel\n1.0\n2.0\n", 3.0),
  ("TightLoadManagementModel\n1.0\n2.0\n", 3.0),
  ("SelfishLoadManagementModel\n1.0\n2.0\n", 3.0),
])
def test_benchmark_summary_sums_every_runtime_format(tmp_path, text, expected):
  import benchmark_summary
  (tmp_path / "runtime.csv").write_text(text)
  assert benchmark_summary._single_column_sum(str(tmp_path / "runtime.csv")) == pytest.approx(expected)


def test_benchmark_summary_missing_and_empty_runtime(tmp_path):
  import benchmark_summary
  import math
  assert math.isnan(benchmark_summary._single_column_sum(str(tmp_path / "runtime.csv")))
  (tmp_path / "runtime.csv").write_text("step,algorithm_s,wall_s,parallel_phases,workers\n")
  assert math.isnan(benchmark_summary._single_column_sum(str(tmp_path / "runtime.csv")))


@pytest.mark.parametrize("name", sorted(methods.EXCLUDED) + ["centralized"])
def test_runtime_contract_of_excluded_and_centralized_runners(name, tmp_path):
  from capture_baseline import baseline_runners, capture_config
  config = capture_config(INSTANCES[0], tmp_path, name)
  config["verbose"] = 0
  frame = pd.read_csv(Path(baseline_runners()[name](config, 2, log_on_file=True, disable_plotting=True))
                      / "runtime.csv")
  assert tuple(frame.columns) == step_time.COLUMNS and frame["step"].tolist() == [0, 1]
  assert (frame.algorithm_s >= 0).all() and (frame.algorithm_s <= frame.wall_s + 1e-3).all()
  if name == "centralized":   # one solver call: no phases, the solver's own threads
    assert (frame.parallel_phases == 0).all() and (frame.workers == 1).all()
  else:                       # sequential sweeps may add no phase; a pool is not required
    assert (frame.parallel_phases >= 0).all() and frame.workers.isin([0, 2]).all()
