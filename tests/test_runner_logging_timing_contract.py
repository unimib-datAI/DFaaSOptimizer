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
  ("hierarchical-madea", "hierarchical_auction.madea_cycles_runner", "HierarchicalMADeALevelCycles", "solve_subproblem"),
  ("hierarchical-madea-cycles", "hierarchical_auction.madea_cycles_runner", "HierarchicalMADeACycles", "solve_subproblem"),
  ("hierarchical-madea-pg", "hierarchical_auction.madea_cycles_runner", "HierarchicalMADeALevelCyclesPG", "solve_subproblem"),
  ("hierarchical-madea-level-cycles-pg", "hierarchical_auction.madea_cycles_runner", "HierarchicalMADeALevelCyclesPG", "solve_subproblem"),
  ("hierarchical-madea-legacy", "hierarchical_auction.madea_runner", "HierarchicalMADeA", "solve_subproblem"),
  ("PLASMA", "plasma.runner", "Plasma", "compute_centralized_objective"),
  ("PLASMA-Welfare", "plasma.runner", "Plasma-Welfare", "compute_centralized_objective"),
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



# The modules this contract covers: runners, their algorithm helpers that own clocks, the frame.
# Tests, RL, campaign/report tooling and archived snapshots are out of scope.
CORE_FILES = [
  "run_centralized_model.py", "run_faasmacro.py", "run_faasmadea.py", "decentralized_auction.py",
  "decentralized_diffusion.py", "decentralized_powerd.py", "decentralized_dual.py",
  "decentralized_frankwolfe.py", "decentralized_gcaa.py", "decentralized_bestresponse.py",
  "decentralized_potentialgame.py", "madea2.py", "madea_pg.py", "one_shot_pg.py",
  "hierarchical_frankwolfe.py", "hierarchical_auction/runner.py", "hierarchical_auction/madea_runner.py",
  "hierarchical_auction/madea_cycles_runner.py", "hierarchical_auction/madea_level_cycles_runner.py",
  "plasma/runner.py", "plasma/welfare.py", "models/model.py", "models/dp_solver.py",
  "utils/run_frame.py", "utils/run_log.py", "utils/step_time.py", "utils/common.py",
]
CLOCKS = {"datetime.datetime.now", "time.time", "time.monotonic", "time.perf_counter",
          "pandas.Timestamp.now"}
LOG_CALLS = {"debug", "info", "warning", "error", "exception", "critical"}


def _qualified_calls(tree):
  """(qualified name, call node, enclosing function) of every call, import aliases resolved."""
  aliases = {}
  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      for a in node.names:
        aliases[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
    elif isinstance(node, ast.ImportFrom) and node.module:
      for a in node.names:
        aliases[a.asname or a.name] = f"{node.module}.{a.name}"

  def name(expr):
    if isinstance(expr, ast.Name):
      return aliases.get(expr.id, expr.id)
    if isinstance(expr, ast.Attribute):
      base = name(expr.value)
      return base and f"{base}.{expr.attr}"
    return None

  def visit(node, function):
    for child in ast.iter_child_nodes(node):
      inner = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else function
      if isinstance(child, ast.Call):
        yield name(child.func), child, inner
      yield from visit(child, inner)

  yield from visit(tree, None)


@pytest.mark.parametrize("path", CORE_FILES)
def test_core_reads_time_only_through_step_time(path):
  if path == "utils/step_time.py":   # the clock itself
    return
  tree = ast.parse((ROOT / path).read_text())
  direct = [f"{path}:{call.lineno} {qualified}" for qualified, call, function in _qualified_calls(tree)
            if qualified in CLOCKS and (path, function) != ("utils/common.py", "run_timestamp")]
  assert not direct, direct


@pytest.mark.parametrize("path", CORE_FILES)
def test_core_has_no_print_and_no_log_stream(path):
  source = (ROOT / path).read_text()
  tree = ast.parse(source)
  prints = [call.lineno for qualified, call, _ in _qualified_calls(tree) if qualified == "print"]
  assert not prints and "log_stream" not in source, prints


@pytest.mark.parametrize("path", [p for p in CORE_FILES if not p.startswith("utils/")])
def test_logging_call_sites_in_steps_are_untimed(path):
  # every logger call of the step code sits inside `with step_time.untimed():` (its level guard
  # and argument preparation with it); the frame logs between steps, outside any clock
  tree = ast.parse((ROOT / path).read_text())
  covered = set()
  for node in ast.walk(tree):
    if isinstance(node, ast.With) and any("untimed" in ast.unparse(i.context_expr) for i in node.items):
      covered.update(id(child) for child in ast.walk(node))
  bare = [f"{path}:{call.lineno}" for qualified, call, _ in _qualified_calls(tree)
          if qualified and qualified.split(".")[-1] in LOG_CALLS and qualified.split(".")[0] == "log"
          and id(call) not in covered]
  assert not bare, bare


@pytest.mark.parametrize("name", ["MADEA", "faas-macro"])
def test_verbosity_and_slow_logging_do_not_change_decisions(tmp_path, monkeypatch, name):
  import time
  from compare_artifacts import normalize_termination
  from utils import run_log
  delay, records = 0.002, []

  def slow(emit):
    def wrapped(self, record):
      records.append(record)
      time.sleep(delay)                # a slow observer: disk, terminal, network
      emit(self, record)
    return wrapped

  instance = capture_baseline.instances()[0]
  folders = {}
  for verbose in (0, 2):
    if verbose == 2:
      monkeypatch.setattr(run_log._FileHandler, "emit", slow(run_log._FileHandler.emit))
    config = capture_baseline.capture_config(instance, tmp_path / str(verbose), name)
    config["verbose"] = verbose
    folders[verbose] = Path(capture_baseline.baseline_runners()[name](config, 0, log_on_file=True,
                                                                      disable_plotting=True))
  quiet, loud = folders[0], folders[2]
  for f in ("obj.csv", "LSPc_solution.csv", "LSP_solution.csv"):
    assert (quiet / f).read_bytes() == (loud / f).read_bytes(), f
  assert (normalize_termination((quiet / "termination_condition.csv").read_text())
          == normalize_termination((loud / "termination_condition.csv").read_text()))
  runtime = pd.read_csv(loud / "runtime.csv")
  in_steps = len(records) - 2 * len(runtime) - 1   # outside the clock: t = and TOTAL per step, saved
  assert in_steps > 0
  assert (runtime.wall_s - runtime.algorithm_s).sum() >= 0.9 * delay * in_steps
