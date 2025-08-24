from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import parse
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
    _, axs = plt.subplots(nrows = 2, ncols = 1, sharex = True)
    total_runtime.drop("tot", axis = "columns").plot(grid = True, ax = axs[0])
    total_runtime["tot"].plot(grid = True, ax = axs[1])
    axs[0].set_ylabel("Single iteration runtime [s]")
    axs[1].set_ylabel("Total runtime [s]")
    axs[1].set_xlabel("Control time period $t$")
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
    complete_path: str, exp: str, logs_df: pd.DataFrame
  ) -> pd.DataFrame:
  # open log file
  lines = []
  with open(os.path.join(complete_path, "out.log"), "r") as istream:
    lines = istream.readlines()
  # loop over lines
  row_idx = 0
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
      while t_row_idx < len(lines) and not (
          lines[t_row_idx].startswith("t = ")
        ):
        if lines[t_row_idx].startswith("    it = "):
          it, psi = parse.parse(
            "    it = {} (psi = {})\n", lines[t_row_idx]
          )
          it = int(it)
          psi = float(psi)
          # loop over iteration info
          it_row_idx = t_row_idx + 1
          while it_row_idx < len(lines) and not (
              lines[it_row_idx].startswith("    it = ") or 
                lines[it_row_idx].startswith("t = ")
            ):
            if "compute_social_welfare" in lines[it_row_idx]:
              _, c_val, val, runtime = parse.parse(
                "        compute_social_welfare: DONE ({}; current: {}; sw: {}; runtime = {})\n",
                lines[it_row_idx]
              )
              df["social_welfare"].append(float(c_val))
              df["best_social_welfare"].append(float(val))
              df["social_welfare_runtime"].append(float(runtime))
            elif "rmp" in lines[it_row_idx]:
              tc, _, runtime = parse.parse(
                "        rmp: DONE ({}; obj = {}; runtime = {})\n", 
                lines[it_row_idx]
              )
              df["coord_tc"].append(tc)
              df["coord_runtime"].append(float(runtime))
            elif "sp" in lines[it_row_idx]:
              _, _, runtime = parse.parse(
                "        sp: DONE  ({}; obj = {}; runtime = {})\n", 
                lines[it_row_idx]
              )
              df["sp_runtime"].append(float(runtime))
            it_row_idx += 1
          # save iteration info
          df["iteration"].append(it)
          df["time"].append(t)
          df["exp"].append(exp)
          # move to the next iteration
          t_row_idx = it_row_idx
      # merge and move to the next time step
      logs_df = pd.concat(
        [logs_df, pd.DataFrame(df)], ignore_index = True
      )
      row_idx = t_row_idx
  return logs_df


def parse_logs(base_folder: str) -> pd.DataFrame:
  social_welfare = pd.DataFrame()
  for foldername in os.listdir(base_folder):
    complete_path = os.path.join(base_folder, foldername)
    if os.path.isdir(complete_path) and not foldername.startswith("."):
      if "LSP_solution.csv" in os.listdir(complete_path):
        social_welfare = parse_log_file(
          complete_path, foldername, social_welfare
        )
  return social_welfare


if __name__ == "__main__":
  base_folder = "solutions/homogeneous_demands/Nf4"
  social_welfare = parse_logs(base_folder)
  # total_runtime = get_spcoord_runtime(social_welfare)
  # total_runtime.to_csv(os.path.join(base_folder, "spcoord_runtime.csv"))
  for exp, data in social_welfare.groupby("exp"):
    last_it = 0
    # social welfare pattern
    _, axs = plt.subplots(nrows = 2, ncols = 1, sharex = True, figsize = (12,8))
    for t, iter_data in data.groupby("time"):
      iter_data["time+iteration"] = iter_data["iteration"] + last_it
      # social welfare
      iter_data.plot(
        x = "time+iteration",
        y = "social_welfare",
        color = mcolors.TABLEAU_COLORS["tab:blue"],
        linestyle = "dashed",
        linewidth = 1,
        marker = ".",
        grid = True,
        label = None,
        legend = False,
        ax = axs[0]
      )
      # "best" social welfare
      iter_data.plot(
        x = "time+iteration",
        y = "best_social_welfare",
        color = "r",
        # linestyle = "dashed",
        linewidth = 1,
        marker = ".",
        grid = True,
        label = None,
        legend = False,
        ax = axs[0]
      )
      # coord termination condition
      iter_data["tc"] = 1
      iter_data["tc_color"] = [
        mcolors.TABLEAU_COLORS["tab:green"] if tc == "optimal" 
          else mcolors.TABLEAU_COLORS["tab:red"] for tc in iter_data["coord_tc"]
      ]
      iter_data.plot.scatter(
        x = "time+iteration",
        y = "tc",
        marker = "*",
        c = "tc_color",
        ax = axs[1]
      )
      # next time step
      axs[0].axvline(
        x = last_it,
        linestyle = "dotted",
        linewidth = 1,
        color = "k"
      )
      axs[1].axvline(
        x = last_it,
        linestyle = "dotted",
        linewidth = 1,
        color = "k"
      )
      if iter_data["iteration"].max() > 0:
        last_it += iter_data["iteration"].max()
      else:
        last_it += t
    plt.title(exp)
    plt.show()
