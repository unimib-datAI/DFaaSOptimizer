import json
from pathlib import Path

import pandas as pd

import fwd_analysis
import what_if_analysis
from logs_postprocessing import (
  parse_log_file,
  parse_faasmacro_log_file,
  get_faasmacro_runtime,
)


def _write_lines(path: Path, lines: list[str]):
  path.write_text("\n".join(lines + [""]), encoding="utf-8")


def _write_model_outputs(folder: Path, model: str):
  solution = pd.DataFrame({
    "n0_f0_loc": [3.0, 0.0],
    "n1_f0_loc": [4.0, 5.0],
    "n0_f0_fwd": [2.0, 1.0],
    "n1_f0_fwd": [1.0, 2.0],
    "n0_f0": [0.0, 1.0],
    "n1_f0": [2.0, 0.0],
  })
  replicas = pd.DataFrame({"n0_f0": [1.0, 1.0], "n1_f0": [2.0, 2.0]})
  detailed = pd.DataFrame({
    "n0_f0_n1_tot": [2.0, 1.0],
    "n1_f0_n0_tot": [1.0, 2.0],
    "n0_f0_n1_accepted": [1.5, 0.5],
    "n1_f0_n0_accepted": [0.5, 1.5],
  })
  utilization = pd.DataFrame({"n0_f0": [0.5, 0.2], "n1_f0": [0.7, 0.8]})
  solution.to_csv(folder / f"{model}_solution.csv", index=False)
  replicas.to_csv(folder / f"{model}_replicas.csv", index=False)
  detailed.to_csv(folder / f"{model}_detailed_fwd_solution.csv", index=False)
  utilization.to_csv(folder / f"{model}_utilization.csv", index=False)


# --- fwd_analysis ---

def test_count_requests_two_timesteps(tmp_path):
  _write_model_outputs(tmp_path, "ModelX")
  (tmp_path / "base_instance_data.json").write_text(
    json.dumps({"None": {"Nn": {"None": 2}, "Nf": {"None": 1}}}),
    encoding="utf-8",
  )

  local, sentrecv, rejected = fwd_analysis.count_requests(str(tmp_path), "ModelX")

  assert len(local["t"].unique()) == 2
  assert "f0" in local.columns
  assert "f0_sent" in sentrecv.columns
  assert "f0_recv" in sentrecv.columns


def test_filter_traces_classifies_only_local():
  sentrecv = pd.DataFrame({"t": [0, 1], "sent": [0.0, 0.0]})
  rejected = pd.DataFrame({"t": [0, 1], "all": [0.0, 0.0]})

  only_local, with_offloading, with_reject = fwd_analysis.filter_traces(sentrecv, rejected)

  assert only_local == [0, 1]
  assert with_offloading == []
  assert with_reject == []


def test_filter_traces_classifies_with_offloading():
  sentrecv = pd.DataFrame({"t": [0], "sent": [5.0]})
  rejected = pd.DataFrame({"t": [0], "all": [0.0]})

  only_local, with_offloading, with_reject = fwd_analysis.filter_traces(sentrecv, rejected)

  assert only_local == []
  assert with_offloading == [0]
  assert with_reject == []


# --- what_if_analysis ---

def test_add_time_merges_best_solution_with_logs():
  best = pd.DataFrame({
    "exp": ["a", "a"],
    "Nn": [2, 2],
    "time": [0, 1],
    "best_solution_it": [1, 2],
    "obj": [10.0, 12.0],
  })
  logs = pd.DataFrame({
    "exp": ["a", "a"],
    "Nn": [2, 2],
    "time": [0, 1],
    "iteration": [1, 2],
    "measured_total_time": [3.0, 5.0],
  })
  result = what_if_analysis.add_time(best, logs)

  assert "measured_total_time" in result.columns
  assert result.loc[0, "obj"] == 10.0


def test_find_best_iterations_faas_macro(tmp_path):
  (tmp_path / "base_instance_data.json").write_text(
    json.dumps({"None": {"Nn": {"None": 2}}}),
    encoding="utf-8",
  )
  _write_model_outputs(tmp_path, "LSP")
  _write_lines(
    tmp_path / "out.log",
    [
      "t = 0",
      "    it = 0 (psi = 1.0)",
      "        compute_social_welfare: DONE (ok; current: 10.0; sw: 12.0)",
      "        rmp: DONE (optimal; obj = 5.0)",
      "        sp: DONE  (ok; obj = 7.0; runtime = 0.3)",
      "        best solution updated; obj = 12.0",
      "        best centralized solution updated; obj = 5.0",
      "    TOTAL RUNTIME [s] = 2.0 (wallclock: 3.0)",
      "All solutions saved",
    ],
  )

  sw, cobj = what_if_analysis.find_best_iterations(str(tmp_path))

  assert len(sw) > 0
  assert len(cobj) > 0
  assert sw.loc[0, "method"] == "faas-macro"


def test_get_faasmacro_runtime_handles_missing_columns():
  logs = pd.DataFrame({
    "exp": ["exp"],
    "time": [0],
    "iteration": [0],
    "social_welfare_runtime": [1.0],
  })
  out = get_faasmacro_runtime(logs, "/tmp")
  assert "tot" in out.columns


def test_what_if_compute_minmaxavg_in_milestone_all_stats():
  tvals = pd.DataFrame({
    "Nn": [2, 3, 4],
    "obj": [10.0, 12.0, 14.0],
    "measured_total_time": [3.0, 5.0, 7.0],
    "dev": [0.0, 20.0, 30.0],
    "centralized_dev": [1.0, 2.0, 3.0],
  })
  metrics = what_if_analysis.compute_minmaxavg_in_milestone(tvals, 30)

  assert set(metrics["which"].unique()) == {"min", "max", "avg"}
  assert len(metrics) == 3
  assert metrics.loc[metrics["which"] == "avg", "obj"].iloc[0] == 12.0
