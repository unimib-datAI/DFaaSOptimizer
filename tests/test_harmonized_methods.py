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
