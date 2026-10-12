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
  ("FaaS-MAPG-S", "decentralized_potentialgame", "FaaS-MAPG-S", "solve_subproblem"),
  ("FaaS-MAPG-R", "decentralized_potentialgame", "FaaS-MAPG-R", "solve_subproblem"),
  ("MADEA", "run_faasmadea", "FaaS-MADeA", "solve_subproblem"),
  ("MADEA2", "run_faasmadea", "FaaS-MADeA", "solve_subproblem"),
  ("MADEA2-b", "run_faasmadea", "FaaS-MADeA", "solve_subproblem"),
  ("MADEA-PG", "run_faasmadea", "FaaS-MADeA-PG", "solve_subproblem"),
  ("MADEA-one-shot", "decentralized_auction", "FaaS-MADeA(1s)", "solve_subproblem"),
  ("MADEA2-one-shot", "decentralized_auction", "FaaS-MADeA(1s)", "solve_subproblem"),
  ("MADEA2-b-one-shot", "decentralized_auction", "FaaS-MADeA(1s)", "solve_subproblem"),
  ("one-shot-pg", "decentralized_auction", "One-shot-PG", "solve_subproblem"),
  ("one-shot2-pg", "decentralized_auction", "One-shot-PG", "solve_subproblem"),
  ("one-shot2-b-pg", "decentralized_auction", "One-shot-PG", "solve_subproblem"),
  ("faas-macro", "run_faasmacro", "FaaS-MACrO", "solve_subproblem"),
  ("faas-macro-v0", "run_faasmacro", "FaaS-MACrO", "solve_subproblem"),
  ("hierarchical-MAFW", "hierarchical_frankwolfe", "Hierarchical-MAFW", "solve_subproblem"),
  ("MAFW-PG", "hierarchical_frankwolfe", "MAFW-PG", "solve_subproblem"),
  ("MAFW-b-PG", "hierarchical_frankwolfe", "MAFW-PG", "solve_subproblem"),
  ("hierarchical-MAFW-PG", "hierarchical_frankwolfe", "Hierarchical-MAFW-PG", "solve_subproblem"),
  ("FaaS-MAFW-b", "decentralized_frankwolfe", "FaaS-MAFW", "solve_subproblem"),
  ("hierarchical", "hierarchical_auction.runner", "HierarchicalAuction", "solve_subproblem"),
  ("hierarchical-one-shot", "hierarchical_auction.runner", "HierarchicalAuction", "solve_subproblem"),
  ("hierarchical-one-shot-pg", "hierarchical_auction.runner", "HierarchicalOneShotPG", "solve_subproblem"),
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
  assert {"config.json", "out.log", "LSPc_solution.csv"} <= {p.name for p in folder.iterdir()}
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
  source = (ROOT / f"{module.replace('.', '/')}.py").read_text()
  tree = ast.parse(source)
  prints = [n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print"]
  assert not prints and "log_stream" not in source, prints
  entries = [n for n in tree.body if isinstance(n, ast.FunctionDef)   # runner signature
             and [a.arg for a in n.args.args[:2]] == ["config", "parallelism"]]
  assert entries and all(n.end_lineno - n.lineno + 1 <= 80 for n in entries)
  assert any("run_frame.run(" in ast.unparse(n) for n in entries)


def test_macro_draws_both_histories(tmp_path):
  import pyomo.environ as pyo
  import run_faasmacro
  from test_review_distributed_regressions import _materialized_config, _two_node_data
  if not pyo.SolverFactory("glpk").available(exception_flag=False):
    pytest.skip("GLPK required")
  config = _materialized_config(tmp_path, _two_node_data(), steps=2)
  config.update(patience=2, solver_options={"general": {}})
  folder = Path(run_faasmacro.run(config, 0, log_on_file=True, disable_plotting=False))
  assert (folder / "sp.png").exists() and (folder / "rmp.png").exists()
  assert {"LRMP_solution.csv", "LRMPc_solution.csv"} <= {p.name for p in folder.iterdir()}


def test_hierarchical_plots_at_any_size(tmp_path):
  instance = capture_baseline.instances()[0]          # 30 nodes: above the default plot limit
  config = capture_baseline.capture_config(instance, tmp_path, "hierarchical")
  config.update(max_steps=1, max_run_time=1, verbose=0)
  folder = Path(capture_baseline.baseline_runners()["hierarchical"](config, 0, log_on_file=True,
                                                                     disable_plotting=False))
  assert (folder / "sp.png").exists()
