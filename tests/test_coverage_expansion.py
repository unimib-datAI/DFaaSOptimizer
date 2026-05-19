import json
from pathlib import Path

import numpy as np
import pandas as pd

import compare_results
import fwd_analysis
import postprocessing
import run
import what_if_analysis
from logs_postprocessing import (
  get_faasmacro_runtime,
  parse_faasmacro_log_file,
  parse_faasmadea0_log_file,
  parse_logs,
)


def _write_lines(path: Path, lines: list[str]):
  path.write_text("\n".join(lines + [""]), encoding = "utf-8")


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
  solution.to_csv(folder / f"{model}_solution.csv", index = False)
  replicas.to_csv(folder / f"{model}_replicas.csv", index = False)
  detailed.to_csv(folder / f"{model}_detailed_fwd_solution.csv", index = False)
  utilization.to_csv(folder / f"{model}_utilization.csv", index = False)
  pd.DataFrame({"SP/coord": [10.0, 11.0]}).to_csv(folder / "obj.csv", index = False)


def test_parse_faasmacro_log_file_extracts_iterations_and_best_solutions(tmp_path):
  _write_lines(
    tmp_path / "out.log",
    [
      "t = 0",
      "    it = 0 (psi = 1.0)",
      "        compute_social_welfare: DONE (ok; current: 10.0; sw: 12.0; runtime = 0.1)",
      "        rmp: DONE (optimal; obj = 5.0; runtime = 0.2)",
      "        sp: DONE  (ok; obj = 7.0; runtime = 0.3)",
      "        best solution updated; obj = 12.0",
      "        best centralized solution updated; obj = 5.0",
      "    it = 1 (psi = 0.5)",
      "        compute_social_welfare: DONE (ok; current: 11.0; sw: 13.0)",
      "        rmp: DONE (optimal; obj = 4.0)",
      "        sp: DONE  (ok; obj = 6.0; runtime = 0.4)",
      "    TOTAL RUNTIME [s] = 2.0 (wallclock: 3.0)",
      "All solutions saved",
    ],
  )

  logs_df, best = parse_faasmacro_log_file(str(tmp_path), "exp", pd.DataFrame(), {}, 2)

  assert list(logs_df["iteration"]) == [0, 1]
  assert logs_df.loc[0, "social_welfare_runtime"] == 0.1
  assert pd.isna(logs_df.loc[1, "coord_runtime"])
  assert list(logs_df["measured_total_time"]) == [2.0, 2.0]
  assert best["social_welfare"].loc[0, "obj"] == 12.0
  assert best["centralized"].loc[0, "obj"] == 5.0


def test_parse_faasmadea0_log_file_reuses_step_sp_runtime(tmp_path):
  _write_lines(
    tmp_path / "out.log",
    [
      "t = 0",
      "    sp: DONE  (ok; obj = 12.0; runtime = 2.5)",
      "    it = 0",
      "        define_bids: DONE; runtime = 0.5)",
      "        evaluate_bids: DONE; runtime = 0.25)",
      "        best solution updated; obj = 8.0",
      "        best centralized solution updated; obj = 7.0",
      "    TOTAL RUNTIME [s] = 3.25 (wallclock: 4.0)",
      "All solutions saved",
    ],
  )

  logs_df, best = parse_faasmadea0_log_file(str(tmp_path), "exp0", pd.DataFrame(), {}, 3)

  assert logs_df.loc[0, "sp_runtime"] == 2.5
  assert logs_df.loc[0, "define_bids_runtime"] == 0.5
  assert best["social_welfare"].loc[0, "best_solution_it"] == 0


def test_parse_logs_dispatches_macro_and_madea_runs(tmp_path):
  macro = tmp_path / "macro"
  madea = tmp_path / "madea"
  macro.mkdir()
  madea.mkdir()
  for folder, extra_file in [(macro, "LRMP_solution.csv"), (madea, None)]:
    (folder / "LSP_solution.csv").write_text("", encoding = "utf-8")
    if extra_file:
      (folder / extra_file).write_text("", encoding = "utf-8")
    (folder / "base_instance_data.json").write_text(
      json.dumps({"None": {"Nn": {"None": 2}}}),
      encoding = "utf-8",
    )
  _write_lines(
    macro / "out.log",
    [
      "t = 0",
      "    it = 0 (psi = 1.0)",
      "        compute_social_welfare: DONE (ok; current: 10.0; sw: 12.0)",
      "        rmp: DONE (optimal; obj = 5.0)",
      "        sp: DONE  (ok; obj = 7.0; runtime = 0.3)",
      "    TOTAL RUNTIME [s] = 2.0 (wallclock: 3.0)",
      "All solutions saved",
    ],
  )
  _write_lines(
    madea / "out.log",
    [
      "t = 0",
      "    it = 0",
      "        sp: DONE  (ok; obj = 12.0; runtime = 2.5)",
      "        define_bids: DONE; runtime = 0.5)",
      "        evaluate_bids: DONE; runtime = 0.25)",
      "    TOTAL RUNTIME [s] = 3.25 (wallclock: 4.0)",
      "All solutions saved",
    ],
  )

  logs_df, _ = parse_logs(str(tmp_path))

  assert set(logs_df["exp"]) == {"macro", "madea"}
  assert list(logs_df["Nn"].unique()) == [2]


def test_get_faasmacro_runtime_computes_iteration_totals(tmp_path):
  logs = pd.DataFrame({
    "exp": ["exp", "exp"],
    "time": [0, 0],
    "iteration": [0, 1],
    "social_welfare_runtime": [1.0, 2.0],
    "coord_runtime": [3.0, 4.0],
    "sp_runtime": [5.0, 6.0],
  })

  out = get_faasmacro_runtime(logs, str(tmp_path))

  assert out.loc[0, "tot"] == 21.0
  assert out.loc[0, "min"] == 9.0
  assert out.loc[0, "max"] == 12.0


def test_postprocessing_loads_counts_and_detects_ping_pong(tmp_path):
  _write_model_outputs(tmp_path, "ModelA")

  solution, replicas, detailed, utilization, obj = postprocessing.load_solution(
    str(tmp_path), "ModelA"
  )
  assert "FaaS-MACrO" in obj.columns
  assert list(utilization.columns) == ["n0_f0", "n1_f0"]

  local, fwd, rejected, replica_counts, ping_pong = postprocessing.load_models_results(
    str(tmp_path),
    ["ModelA"],
    ["Readable"],
  )

  assert local["by_node"]["tot"].loc["n0", "Readable"] == 3.0
  assert fwd["by_function"]["tot"].loc["f0", "Readable"] == 6.0
  assert rejected["by_node"]["tot"].loc["n1", "Readable"] == 2.0
  assert replica_counts["by_node"]["tot"].loc["n1", "Readable"] == 4.0
  assert ping_pong["Readable"]
  assert detailed.loc[0, "n0_f0_n1_tot"] == 2.0
  assert solution.loc[1, "n0_f0"] == 1.0


def test_postprocessing_plot_helpers_write_outputs(tmp_path):
  df = pd.DataFrame({
    "n0_f0": [1.0, 2.0],
    "n1_f0": [3.0, 4.0],
    "node": ["n0", "n1"],
    "function": ["f0", "f0"],
  })

  tot = postprocessing.plot_count(df, "function", str(tmp_path), plot_all = False)
  postprocessing.plot_global_count(
    {"by_function": {"tot": tot[["tot"]]}},
    "local",
    plot_folder = str(tmp_path),
  )

  assert (tmp_path / "function_tot.png").exists()
  assert (tmp_path / "local-by_function-tot.png").exists()


def test_fwd_analysis_counts_and_filters_requests(tmp_path):
  _write_model_outputs(tmp_path, "ModelA")
  (tmp_path / "base_instance_data.json").write_text(
    json.dumps({"None": {"Nn": {"None": 2}, "Nf": {"None": 1}}}),
    encoding = "utf-8",
  )

  local, sentrecv, rejected = fwd_analysis.count_requests(str(tmp_path), "ModelA")
  only_local, with_offloading, with_reject = fwd_analysis.filter_traces(sentrecv, rejected)

  assert local.groupby("t")["all"].sum().loc[0] == 7.0
  assert sentrecv.groupby("t")["sent"].sum().loc[0] == 3.0
  assert rejected.groupby("t")["all"].sum().loc[1] == 1.0
  assert only_local == []
  assert with_offloading == []
  assert with_reject == [0, 1]


def test_what_if_small_dataframe_helpers(tmp_path):
  best = pd.DataFrame({
    "exp": ["a"],
    "Nn": [2],
    "time": [0],
    "best_solution_it": [1],
    "obj": [10.0],
  })
  logs = pd.DataFrame({
    "exp": ["a"],
    "Nn": [2],
    "time": [0],
    "iteration": [1],
    "measured_total_time": [3.0],
  })
  with_time = what_if_analysis.add_time(best, logs)
  metrics = what_if_analysis.compute_minmaxavg_in_milestone(
    pd.DataFrame({
      "Nn": [2, 3],
      "obj": [10.0, 12.0],
      "measured_total_time": [3.0, 5.0],
      "dev": [0.0, 20.0],
      "centralized_dev": [1.0, 2.0],
    }),
    30,
  )

  assert with_time.loc[0, "measured_total_time"] == 3.0
  assert set(metrics["which"]) == {"min", "max", "avg"}


def test_compare_results_plotters_and_csv_loader(tmp_path):
  models = ["LoadManagementModel", "FaaS-MACrO"]
  obj = pd.DataFrame({
    "Nn": [2, 2, 3, 3],
    "dev": [0.0, 1.0, 2.0, 3.0],
    "LoadManagementModel": [10.0, 11.0, 12.0, 13.0],
    "FaaS-MACrO": [9.0, 10.0, 11.0, 12.0],
  })
  runtime = pd.DataFrame({
    "Nn": [2, 2, 3, 3],
    "dev": [1.0, 1.1, 1.2, 1.3],
    "iteration": [1, 2, 3, 4],
    "best_iteration": [1, 1, 2, 2],
    "LoadManagementModel": [5.0, 6.0, 7.0, 8.0],
    "FaaS-MACrO": [6.0, 7.0, 8.0, 9.0],
  })
  rej = pd.DataFrame({
    "Nn": [2, 2, 3, 3],
    "dev": [0.0, -1.0, 1.0, 2.0],
    "LoadManagementModel": [1.0, 2.0, 3.0, 4.0],
    "FaaS-MACrO": [1.5, 2.5, 3.5, 4.5],
  })
  obj.to_csv(tmp_path / "obj.csv", index = False)
  runtime.to_csv(tmp_path / "runtime.csv", index = False)
  rej.to_csv(tmp_path / "rejections.csv", index = False)

  loaded_obj, loaded_rej, loaded_runtime = compare_results.compare_results(
    str(tmp_path),
    "Nn",
    "Nodes",
    models,
  )

  assert loaded_obj.shape == obj.shape
  assert loaded_rej.shape == rej.shape
  assert loaded_runtime.shape == runtime.shape
  assert (tmp_path / "box.png").exists()
  assert (tmp_path / "box_detailed.png").exists()
  assert (tmp_path / "bars.png").exists()
  assert (tmp_path / "violin_detailed.png").exists()


def test_run_results_postprocessing_aggregates_synthetic_experiment(tmp_path):
  centralized = tmp_path / "centralized_exp"
  macro = tmp_path / "macro_exp"
  centralized.mkdir()
  macro.mkdir()
  _write_model_outputs(centralized, "LoadManagementModel")
  _write_model_outputs(macro, "LSP")
  pd.DataFrame({"LoadManagementModel": [1.0, 1.5]}).to_csv(
    centralized / "runtime.csv",
    index = False,
  )
  pd.DataFrame({"tot": [2.0, 2.5]}).to_csv(macro / "runtime.csv", index = False)
  pd.DataFrame({
    "0": [
      "steady (it: 2; obj. deviation: 0.0; best it: 1)",
      "dev below tol (it: 3; obj. deviation: 0.1)",
    ]
  }).to_csv(macro / "termination_condition.csv")

  run.base_solution_folder = str(tmp_path)
  solution_folders = {
    "experiments_list": [[2, 123]],
    "centralized": [str(centralized)],
    "faas-macro": [str(macro)],
  }

  run.results_postprocessing(
    solution_folders,
    str(tmp_path),
    loop_over = "Nn",
    methods = ["centralized", "faas-macro"],
  )

  post_folder = tmp_path / "postprocessing"
  assert (post_folder / "obj.csv").exists()
  assert (post_folder / "runtime.csv").exists()
  assert (post_folder / "rejections.csv").exists()
  assert (post_folder / "ping_pong_problems.txt").exists()


def test_postprocessing_history_and_boxplot_helpers(tmp_path):
  input_requests = {
    0: {
      0: np.array([2.0, 3.0]),
      1: np.array([1.0, 2.0]),
    }
  }
  solution = pd.DataFrame({
    "n0_f0_loc": [1.0, 2.0],
    "n0_f0_fwd": [0.5, 0.5],
    "n0_f0": [0.0, 0.0],
    "n1_f0_loc": [1.0, 1.0],
    "n1_f0_fwd": [0.0, 0.5],
    "n1_f0": [0.0, 0.0],
  })
  utilization = pd.DataFrame({"n0_f0": [0.5, 0.6], "n1_f0": [0.3, 0.4]})
  replicas = pd.DataFrame({"n0_f0": [1.0, 1.0], "n1_f0": [1.0, 2.0]})
  offloaded = pd.DataFrame({
    "n0_f0_accepted": [0.0, 0.5],
    "n1_f0_accepted": [0.5, 0.0],
  })

  postprocessing.plot_history(
    input_requests,
    min_run_time = 0,
    max_run_time = 2,
    run_time_step = 1,
    solution = solution,
    utilization = utilization,
    replicas = replicas,
    offloaded = offloaded,
    obj_values = [10.0, 9.0],
    plot_filename = str(tmp_path / "history.png"),
  )
  postprocessing.runtime_obj_boxplot(
    pd.DataFrame({
      "method": ["A", "A", "B", "B"],
      "Nn": [2, 3, 2, 3],
      "runtime": [1.0, 2.0, 1.5, 2.5],
    }),
    "runtime",
    str(tmp_path),
    "runtime_box",
  )

  assert (tmp_path / "history.png").exists()
  assert (tmp_path / "runtime_box.png").exists()


def test_what_if_analyze_final_results_writes_deviation_outputs(tmp_path):
  last_by_sw = pd.DataFrame({
    "best_solution_it": [1, 2],
    "obj": [10.0, 12.0],
  })
  last_by_cobj = pd.DataFrame({
    "best_solution_it": [2, 4],
    "obj": [11.0, 15.0],
  })

  what_if_analysis.analyze_final_results(last_by_sw, last_by_cobj, str(tmp_path))

  assert (tmp_path / "best_solution_obj.png").exists()
  assert (tmp_path / "best_solution_obj_dev.png").exists()
  out = pd.read_csv(tmp_path / "best_solution_obj.csv")
  assert "obj_dev" in out.columns
