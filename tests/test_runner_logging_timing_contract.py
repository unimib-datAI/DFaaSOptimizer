"""Contract of the runners migrated to the shared frame (utils/run_frame.py).

Each case: exact obj.csv label, five-column runtime.csv, artifacts equal to the captured baseline
(experiments/harmonization/capture_baseline.py --artifacts; RUNNER_BASELINE, default
/tmp/runner-logging-timing/before, compared when present), no step clock or log handler left
after a return or a failure, the frame and the logger instead of print/log_stream, run() <= 80 lines.
"""
import ast
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "harmonization"))

import capture_baseline  # noqa: E402
from compare_artifacts import compare_run  # noqa: E402
from utils import step_time  # noqa: E402

BASELINE = Path(os.environ.get("RUNNER_BASELINE", "/tmp/runner-logging-timing/before"))
# registry name, module, obj.csv label, a module-level callable of the step (failure injection)
CASES = [
  ("faas-diffuse", "decentralized_diffusion", "FaaS-MADiG", "solve_subproblem"),
  ("faas-powd", "decentralized_powerd", "FaaS-MAPoD", "sample_assignments"),
  ("faas-dual", "decentralized_dual", "FaaS-MALD", "solve_subproblem"),
  ("FaaS-MAFW", "decentralized_frankwolfe", "FaaS-MAFW", "solve_subproblem"),
  ("faas-gcaa", "decentralized_gcaa", "FaaS-MAGCAA", "solve_subproblem"),
  ("faas-br-s", "decentralized_bestresponse", "FaaS-MABR-S", "solve_subproblem"),
  ("faas-br-r", "decentralized_bestresponse", "FaaS-MABR-R", "solve_subproblem"),
  ("faas-br-o", "decentralized_bestresponse", "FaaS-MABR-O", "solve_subproblem"),
]
MODULES = sorted({case[1] for case in CASES})


def _run(name, folder):
  instance = capture_baseline.instances()[0]
  config = capture_baseline.capture_config(instance, folder, name)
  return Path(capture_baseline.baseline_runners()[name](config, 0, log_on_file=True, disable_plotting=True))


@pytest.mark.parametrize("name, module, label, _", CASES)
def test_runner_outputs_match_the_baseline(tmp_path, name, module, label, _):
  folder = _run(name, tmp_path)
  assert list(pd.read_csv(folder / "obj.csv").columns) == [label]
  assert tuple(pd.read_csv(folder / "runtime.csv").columns) == step_time.COLUMNS
  assert {"config.json", "out.log", "LSP_solution.csv", "LSPc_solution.csv"} <= {p.name for p in folder.iterdir()}
  assert not step_time.active()
  runs = BASELINE / name / capture_baseline.instances()[0].name
  baseline = next(runs.iterdir(), None) if runs.is_dir() else None
  if baseline is not None:
    assert compare_run(baseline, folder, name) == []


@pytest.mark.parametrize("name, module, label, step_call", CASES)
def test_runner_failure_leaves_no_clock_or_log(tmp_path, monkeypatch, name, module, label, step_call):
  import importlib
  root = logging.getLogger("dfaas")
  handlers = list(root.handlers)

  def failing(*args, **kwargs):
    raise RuntimeError("injected failure")

  monkeypatch.setattr(importlib.import_module(module), step_call, failing)
  with pytest.raises(RuntimeError, match="injected failure"):
    _run(name, tmp_path)
  assert not step_time.active() and root.handlers == handlers


@pytest.mark.parametrize("module", MODULES)
def test_runner_uses_the_frame_and_the_logger(module):
  source = (ROOT / f"{module}.py").read_text()
  tree = ast.parse(source)
  prints = [n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print"]
  assert not prints and "log_stream" not in source, prints
  entries = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith(("run", "_run"))]
  assert entries and all(n.end_lineno - n.lineno + 1 <= 80 for n in entries)
  assert any("run_frame.run(" in ast.unparse(n) for n in entries)
