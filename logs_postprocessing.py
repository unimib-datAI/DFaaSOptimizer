from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
from typing import Tuple
import pandas as pd
import parse
import json
import os


def get_spcoord_runtime(
    logs_df: pd.DataFrame, plot_folder: str
  ) -> pd.DataFrame:
  all_total_runtime = pd.DataFrame()
  for exp, data in logs_df.groupby("exp"):
    all_time_data = pd.DataFrame()
    for _, t_data in data.groupby("time"):
      time_data = t_data.loc[:,t_data.columns.str.endswith("runtime")].copy(
        deep = True
      )
      time_data["tot_runtime"] = time_data.sum(axis = "columns")
      time_data.columns = [c.split("_")[0] for c in time_data.columns]
      time_data["iteration"] = t_data["iteration"]
      time_data["time"] = t_data["time"]
      all_time_data = pd.concat([all_time_data, time_data], ignore_index = True)
    # plot
    avg_t = all_time_data.groupby("time").mean()
    min_t = all_time_data.groupby("time").min()
    max_t = all_time_data.groupby("time").max()
    tot_t = all_time_data.groupby("time").sum()
    total_runtime = pd.DataFrame({
      "avg": avg_t["tot"],
      "min": min_t["tot"],
      "max": max_t["tot"],
      "tot": tot_t["tot"]
    })
    _, axs = plt.subplots(
      nrows = 4, ncols = 1, sharex = True, figsize = (8,20)
    )
    # -- min
    min_t.drop(["tot", "iteration"], axis = "columns").plot.bar(
      grid = True, 
      ax = axs[0],
      stacked = True
    )
    # -- max
    max_t.drop(["tot", "iteration"], axis = "columns").plot.bar(
      grid = True, 
      ax = axs[1],
      stacked = True
    )
    # -- avg
    avg_t.drop(["tot", "iteration"], axis = "columns").plot.bar(
      grid = True, 
      ax = axs[2],
      stacked = True
    )
    # -- tot
    tot_t.drop(["tot", "iteration"], axis = "columns").plot.bar(
      grid = True, 
      ax = axs[3],
      stacked = True
    )
    total_runtime["tot"].plot(
      grid = True, 
      ax = axs[3],
      linewidth = 3,
      color = "k"
    )
    axs[3].axhline(
      y = total_runtime["tot"].mean(),
      linestyle = "dashed",
      linewidth = 3,
      color = mcolors.TABLEAU_COLORS["tab:red"]
    )
    axs[0].set_ylabel("Min 1-iter runtime [s]")
    axs[1].set_ylabel("Max 1-iter runtime [s]")
    axs[2].set_ylabel("Avg 1-iter runtime [s]")
    axs[-1].set_ylabel("Total runtime [s]")
    axs[-1].set_xlabel("Control time period $t$")
    plt.savefig(
      os.path.join(plot_folder, "runtime.png"),
      dpi = 300,
      format = "png",
      bbox_inches = "tight"
    )
    plt.close()
    # merge
    total_runtime["exp"] = exp
    all_total_runtime = pd.concat([all_total_runtime, total_runtime])
  return all_total_runtime


def parse_log_file(
    complete_path: str, 
    exp: str, 
    logs_df: pd.DataFrame, 
    best_sol_df: pd.DataFrame, 
    Nn: int
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
  # open log file
  lines = []
  with open(os.path.join(complete_path, "out.log"), "r") as istream:
    lines = istream.readlines()
  # loop over lines
  row_idx = 0
  best_solution_df = {
    "social_welfare": {
      "exp": [],
      "time": [],
      "best_solution_it": [],
      "obj": [],
      "Nn": []
    },
    "centralized": {
      "exp": [],
      "time": [],
      "best_solution_it": [],
      "obj": [],
      "Nn": []
    }
  }
  while row_idx < len(lines):
    # look for next time step
    if lines[row_idx].startswith("t = "):
      t = int(parse.parse("t = {}\n", lines[row_idx])[0])
      # loop over iterations
      t_row_idx = row_idx + 1
      df = {
        "exp": [],
        "time": [],
        "iteration": [],
        "social_welfare": [],
        "best_social_welfare": [],
        "social_welfare_runtime": [],
        "coord_tc": [],
        "coord_runtime": [],
        "sp_runtime": []
      }
      n_iterations = 0
      while t_row_idx < len(lines) and not (
          lines[t_row_idx].startswith("t = ") or
            lines[t_row_idx].startswith("All solutions saved")
        ):
        if lines[t_row_idx].startswith("    it = "):
          n_iterations += 1
          it, psi = parse.parse(
            "    it = {} (psi = {})\n", lines[t_row_idx]
          )
          it = int(it)
          psi = float(psi)
          # loop over iteration info
          it_row_idx = t_row_idx + 1
          while it_row_idx < len(lines) and not (
              lines[it_row_idx].startswith("    it = ") or 
                lines[it_row_idx].startswith("t = ") or
                  lines[it_row_idx].startswith("    TOTAL RUNTIME")
            ):
            if "compute_social_welfare" in lines[it_row_idx]:
              runtime = None
              if "runtime" in lines[it_row_idx]:
                _, c_val, val, runtime = parse.parse(
                  "        compute_social_welfare: DONE ({}; current: {}; sw: {}; runtime = {})\n",
                  lines[it_row_idx]
                )
              else:
                _, c_val, val = parse.parse(
                  "        compute_social_welfare: DONE ({}; current: {}; sw: {})\n",
                  lines[it_row_idx]
                )
              df["social_welfare"].append(float(c_val))
              df["best_social_welfare"].append(float(val))
              df["social_welfare_runtime"].append(float(runtime) if runtime else None)
            elif "rmp" in lines[it_row_idx]:
              runtime = None
              if "runtime" in lines[it_row_idx]:
                tc, _, runtime = parse.parse(
                  "        rmp: DONE ({}; obj = {}; runtime = {})\n", 
                  lines[it_row_idx]
                )
              else:
                tc, _ = parse.parse(
                  "        rmp: DONE ({}; obj = {})\n", 
                  lines[it_row_idx]
                )
              df["coord_tc"].append(tc)
              df["coord_runtime"].append(float(runtime) if runtime else None)
            elif "sp" in lines[it_row_idx]:
              runtime = None
              if "runtime" in lines[it_row_idx]:
                _, _, runtime = parse.parse(
                  "        sp: DONE  ({}; obj = {}; runtime = {})\n", 
                  lines[it_row_idx]
                )
              df["sp_runtime"].append(float(runtime) if runtime else None)
            elif (
                "check_stopping_criteria" in lines[it_row_idx] and 
                  "wallclock" in lines[it_row_idx]
              ):
              _, trt, wct, _ = parse.parse(
                "        check_stopping_criteria: DONE (runtime = {}; total runtime = {}; wallclock: {}) --> stop? {}\n",
                lines[it_row_idx]
              )
              if "measured_total_time" not in df:
                df["measured_total_time"] = []
              if "wallclock_time" not in df:
                df["wallclock_time"] = []
              df["measured_total_time"].append(float(trt))
              df["wallclock_time"].append(float(wct))
              n_iterations -= 1
            elif "best solution updated" in lines[it_row_idx]:
              best_solution_df["social_welfare"]["exp"].append(exp)
              best_solution_df["social_welfare"]["time"].append(t)
              best_solution_df["social_welfare"]["best_solution_it"].append(it)
              best_solution_df["social_welfare"]["obj"].append(
                float(parse.parse(
                  "        best solution updated; obj = {}\n",
                  lines[it_row_idx]
                )[0])
              )
              best_solution_df["social_welfare"]["Nn"].append(Nn)
            elif "best centralized solution updated" in lines[it_row_idx]:
              best_solution_df["centralized"]["exp"].append(exp)
              best_solution_df["centralized"]["time"].append(t)
              best_solution_df["centralized"]["best_solution_it"].append(it)
              best_solution_df["centralized"]["obj"].append(
                float(parse.parse(
                  "        best centralized solution updated; obj = {}\n",
                  lines[it_row_idx]
                )[0])
              )
              best_solution_df["centralized"]["Nn"].append(Nn)
            it_row_idx += 1
          # save iteration info
          df["iteration"].append(it)
          df["time"].append(t)
          df["exp"].append(exp)
          # move to the next iteration
          t_row_idx = it_row_idx
          # if the iterations are finished, save info on total runtime and 
          # wallclock time
          if (
              it_row_idx < len(lines) and 
                lines[it_row_idx].startswith("    TOTAL RUNTIME")
            ):
            trt, wct = parse.parse(
              "    TOTAL RUNTIME [s] = {} (wallclock: {})\n", lines[it_row_idx]
            )
            if "measured_total_time" not in df:
              df["measured_total_time"] = []
            if "wallclock_time" not in df:
              df["wallclock_time"] = []
            df["measured_total_time"] += [float(trt)] * n_iterations
            df["wallclock_time"] += [float(wct)] * n_iterations
            t_row_idx += 1
            n_iterations = 0
      # add number of nodes
      df["Nn"] = [Nn] * len(df["exp"])
      # merge and move to the next time step
      logs_df = pd.concat(
        [logs_df, pd.DataFrame(df)], ignore_index = True
      )
      row_idx = t_row_idx
      if (
          t_row_idx < len(lines) and 
            lines[t_row_idx].startswith("All solutions saved")
        ):
        row_idx += 1
  best_solution_df["social_welfare"] = pd.concat(
    [
      best_sol_df["social_welfare"], 
      pd.DataFrame(best_solution_df["social_welfare"])
    ],
    ignore_index = True
  )
  best_solution_df["centralized"] = pd.concat(
    [
      best_sol_df["centralized"], 
      pd.DataFrame(best_solution_df["centralized"])
    ],
    ignore_index = True
  )
  return logs_df, best_solution_df


def parse_logs(base_folder: str) -> pd.DataFrame:
  social_welfare = pd.DataFrame()
  best_solution_df = {
    "social_welfare": pd.DataFrame(), "centralized": pd.DataFrame()
  }
  for foldername in os.listdir(base_folder):
    complete_path = os.path.join(base_folder, foldername)
    if os.path.isdir(complete_path) and not foldername.startswith("."):
      if "LSP_solution.csv" in os.listdir(complete_path):
        Nn = 1
        with open(
            os.path.join(complete_path, "base_instance_data.json"), "r"
          ) as istream:
          data = json.load(istream)
          Nn = int(data["None"]["Nn"]["None"])
        social_welfare, best_solution_df = parse_log_file(
          complete_path, foldername, social_welfare, best_solution_df, Nn
        )
  return social_welfare, best_solution_df


if __name__ == "__main__":
  base_folder = "solutions/manual"
  social_welfare, best_solution_df = parse_logs(base_folder)
  total_runtime = get_spcoord_runtime(social_welfare, base_folder)
  os.makedirs(os.path.join(base_folder, "postprocessing"), exist_ok = True)
  social_welfare[
    ["exp", "Nn", "time", "measured_total_time", "wallclock_time"]
  ].groupby(["exp", "time"]).mean().reset_index().to_csv(
    os.path.join(base_folder, "postprocessing", "wallclock.csv"), index = False
  )
  # total_runtime.to_csv(os.path.join(base_folder, "spcoord_runtime.csv"))
  # for exp, data in social_welfare.groupby("exp"):
  #   last_it = 0
  #   # social welfare pattern
  #   _, axs = plt.subplots(nrows = 2, ncols = 1, sharex = True, figsize = (12,8))
  #   for t, iter_data in data.groupby("time"):
  #     iter_data["time+iteration"] = iter_data["iteration"] + last_it
  #     # social welfare
  #     iter_data.plot(
  #       x = "time+iteration",
  #       y = "social_welfare",
  #       color = mcolors.TABLEAU_COLORS["tab:blue"],
  #       linestyle = "dashed",
  #       linewidth = 1,
  #       marker = ".",
  #       grid = True,
  #       label = None,
  #       legend = False,
  #       ax = axs[0]
  #     )
  #     # "best" social welfare
  #     iter_data.plot(
  #       x = "time+iteration",
  #       y = "best_social_welfare",
  #       color = "r",
  #       # linestyle = "dashed",
  #       linewidth = 1,
  #       marker = ".",
  #       grid = True,
  #       label = None,
  #       legend = False,
  #       ax = axs[0]
  #     )
  #     # coord termination condition
  #     iter_data["tc"] = 1
  #     iter_data["tc_color"] = [
  #       mcolors.TABLEAU_COLORS["tab:green"] if tc == "optimal" 
  #         else mcolors.TABLEAU_COLORS["tab:red"] for tc in iter_data["coord_tc"]
  #     ]
  #     iter_data.plot.scatter(
  #       x = "time+iteration",
  #       y = "tc",
  #       marker = "*",
  #       c = "tc_color",
  #       ax = axs[1]
  #     )
  #     # next time step
  #     axs[0].axvline(
  #       x = last_it,
  #       linestyle = "dotted",
  #       linewidth = 1,
  #       color = "k"
  #     )
  #     axs[1].axvline(
  #       x = last_it,
  #       linestyle = "dotted",
  #       linewidth = 1,
  #       color = "k"
  #     )
  #     if iter_data["iteration"].max() > 0:
  #       last_it += iter_data["iteration"].max()
  #     else:
  #       last_it += t
  #   plt.title(exp)
  #   plt.show()
