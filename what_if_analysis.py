from logs_postprocessing import parse_log_file

from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
from typing import Tuple
import pandas as pd
import argparse
import json
import os


def parse_arguments() -> argparse.Namespace:
  """
  Parse input arguments
  """
  parser: argparse.ArgumentParser = argparse.ArgumentParser(
    description = "What-if analysis", 
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument(
    "-f", "--base_folder",
    help = "Base experiment folder",
    type = str,
    required = True
  )
  parser.add_argument(
    "-m", "--milestones",
    help = "List of time limits to consider as milestones",
    nargs = "+",
    default = [0, 10, 30, 60, 120]
  )
  # Parse the arguments
  args: argparse.Namespace = parser.parse_known_args()[0]
  return args


def add_time(
    best_sol_df: pd.DataFrame, logs_df: pd.DataFrame
  ) -> pd.DataFrame:
  new_df = best_sol_df.rename(
    columns = {"best_solution_it": "iteration"}
  ).set_index(
    ["exp","Nn","time","iteration"]).join(
    logs_df.set_index(["exp","Nn","time","iteration"])
  )[["obj", "measured_total_time"]].reset_index().rename(
    columns = {"iteration": "best_solution_it"}
  )
  return new_df


def analyze_final_results(
    last_by_sw: pd.DataFrame, last_by_cobj: pd.DataFrame, output_folder: str
  ):
  # plot
  _, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (10,6), sharey = True)
  b0 = last_by_sw.plot.box(
    grid = True,
    showmeans = True,
    patch_artist = True,
    meanprops = dict(color = mcolors.TABLEAU_COLORS["tab:red"]),
    return_type = "dict",
    fontsize = 14,
    ax = axs[0],
    # label = "by social welfare"
    label = None,
    legend = False
  )
  b1 = last_by_cobj.plot.box(
    grid = True,
    showmeans = True,
    patch_artist = True,
    meanprops = dict(color = mcolors.TABLEAU_COLORS["tab:red"]),
    return_type = "dict",
    fontsize = 14,
    ax = axs[1],
    # label = "by centralized objective"
    label = None,
    legend = False
  )
  # -- colors
  for bplot in [b0, b1]:
    for patch in bplot["boxes"]:
      patch.set_facecolor(mcolors.CSS4_COLORS["lightskyblue"])
    for median in bplot['medians']:
      median.set_color(mcolors.TABLEAU_COLORS["tab:orange"])
    for mean in bplot['means']:
      mean.set_markerfacecolor(mcolors.TABLEAU_COLORS["tab:red"])
      mean.set_markeredgecolor(mcolors.TABLEAU_COLORS["tab:red"])
  plt.savefig(
    os.path.join(output_folder, "best_solution_obj.png"),
    dpi = 300,
    format = "png",
    bbox_inches = "tight"
  )
  plt.close()
  # compute deviation
  last = last_by_sw.join(last_by_cobj,lsuffix="_by_sw",rsuffix="_by_cobj")
  last["iteration_dev"] = (
    last_by_cobj["best_solution_it"] - last_by_sw["best_solution_it"]
  )
  last["obj_dev"] = (
    last_by_cobj["obj"] - last_by_sw["obj"]
  ) / last_by_cobj["obj"] * 100
  # plot
  _, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (10,6))
  b2 = last.plot.box(
    column = "obj_dev",
    grid = True,
    showmeans = True,
    patch_artist = True,
    meanprops = dict(color = mcolors.TABLEAU_COLORS["tab:red"]),
    return_type = "dict",
    fontsize = 14,
    ax = axs[0],
    label = None,
    legend = False
  )
  b3 = last.plot.box(
    column = "iteration_dev",
    grid = True,
    showmeans = True,
    patch_artist = True,
    meanprops = dict(color = mcolors.TABLEAU_COLORS["tab:red"]),
    return_type = "dict",
    fontsize = 14,
    ax = axs[1],
    label = None,
    legend = False
  )
  # -- colors
  for bplot in [b2, b3]:
    for patch in bplot["boxes"]:
      patch.set_facecolor(mcolors.CSS4_COLORS["lightskyblue"])
    for median in bplot['medians']:
      median.set_color(mcolors.TABLEAU_COLORS["tab:orange"])
    for mean in bplot['means']:
      mean.set_markerfacecolor(mcolors.TABLEAU_COLORS["tab:red"])
      mean.set_markeredgecolor(mcolors.TABLEAU_COLORS["tab:red"])
  plt.savefig(
    os.path.join(output_folder, "best_solution_obj_dev.png"),
    dpi = 300,
    format = "png",
    bbox_inches = "tight"
  )
  plt.close()
  # save
  last.reset_index().to_csv(
    os.path.join(output_folder, "best_solution_obj.csv"), index = False
  )


def compute_minmaxavg_in_milestone(
    tvals: pd.DataFrame, milestone: float
  ) -> pd.DataFrame:
  columns = ["Nn","obj","measured_total_time","dev","centralized_dev"]
  metrics_by_milestones = pd.DataFrame()
  # ---- min
  tdf = pd.DataFrame(tvals[columns].min()).transpose()
  tdf["time_limit"] = milestone
  tdf["which"] = "min"
  metrics_by_milestones = pd.concat(
    [metrics_by_milestones, tdf], ignore_index = True
  )
  # ---- max
  tdf = pd.DataFrame(tvals[columns].max()).transpose()
  tdf["time_limit"] = milestone
  tdf["which"] = "max"
  metrics_by_milestones = pd.concat(
    [metrics_by_milestones, tdf], ignore_index = True
  )
  # ---- avg
  tdf = pd.DataFrame(tvals[columns].mean()).transpose()
  tdf["time_limit"] = milestone
  tdf["which"] = "avg"
  metrics_by_milestones = pd.concat(
    [metrics_by_milestones, tdf], ignore_index = True
  )
  return metrics_by_milestones


def compute_progressive_deviation(
    obj_df: pd.DataFrame, milestones: list, output_folder: str
  ):
  progressive_dev = pd.DataFrame()
  for _, vals in obj_df.groupby(["method", "exp", "Nn", "time"]):
    print(vals)
    dev = []
    for idx in range(1, len(vals)):
      dev.append(
        (
          vals.iloc[idx]["obj"] - vals.iloc[0]["obj"]
        ) / vals.iloc[0]["obj"] * 100
      )
    dev_df = vals.iloc[1:].copy(deep = True)
    dev_df["dev"] = dev
    progressive_dev = pd.concat(
      [progressive_dev, dev_df.reset_index()], 
      ignore_index = True
    )
  # match experiment name with description (if available)
  if os.path.exists(os.path.join(output_folder, "../experiments.json")):
    experiments = {}
    with open(
        os.path.join(output_folder, "../experiments.json"), "r"
      ) as ist:
      experiments = json.load(ist)
    exp_description_match = {}
    for exp, exp_description_tuple in zip(
        experiments["faas-macro"], experiments["experiments_list"]
      ):
      exp_description_match[os.path.basename(exp)] = {
        "Nn": int(exp_description_tuple[0]),
        "seed": int(exp_description_tuple[-1]),
        "method": "faas-macro"
      }
    for exp, exp_description_tuple in zip(
        experiments["faas-madea"], experiments["experiments_list"]
      ):
      exp_description_match[os.path.basename(exp)] = {
        "Nn": int(exp_description_tuple[0]),
        "seed": int(exp_description_tuple[-1]),
        "method": "faas-madea"
      }
    # -- add match info
    progressive_dev["seed"] = [
      exp_description_match[exp]["seed"] for exp in progressive_dev["exp"]
    ]
  # load centralized solution (if available)
  if os.path.exists(os.path.join(output_folder, "obj.csv")):
    centralized_obj = pd.read_csv(os.path.join(output_folder, "obj.csv"))
    progressive_dev["centralized_obj"] = None
    for (Nn, seed, time), val in centralized_obj.groupby(["Nn","seed","time"]):
      idxs = progressive_dev[
        (
          progressive_dev["Nn"] == Nn
        ) & (
          progressive_dev["seed"] == seed
        ) & (
          progressive_dev["time"] == time
        )
      ].index
      progressive_dev.loc[
        idxs, "centralized_obj"
      ] = val["LoadManagementModel"].iloc[0]
  progressive_dev["centralized_dev"] = (
    progressive_dev["obj"] - progressive_dev["centralized_obj"]
  ) / progressive_dev["centralized_obj"] * 100
  # save
  progressive_dev.to_csv(
    os.path.join(output_folder, "progressive_dev.csv"), index = False
  )
  # plot and compute metrics by milestones
  metrics_by_milestones = pd.DataFrame()
  for Nn, vals in progressive_dev.groupby("Nn"):
    _, ax = plt.subplots(figsize = (7,3))
    fontsize = 10
    for _, tvals in vals.groupby("time"):
      tvals.plot.scatter(
        x = "measured_total_time",
        y = "centralized_dev",
        c = mcolors.TABLEAU_COLORS["tab:blue"],
        grid = True,
        ax = ax,
        label = None,
        legend = False,
        fontsize = fontsize
      )
    # -- loop over milestones and compute average
    for idx in range(1,len(milestones)):
      tvals = vals[
        (
          vals["measured_total_time"] >= milestones[idx - 1]
        ) & 
        (
          vals["measured_total_time"] < milestones[idx]
        )
      ]
      if len(tvals) > 0:
        idxmax = tvals.groupby(
          ["Nn", "seed", "time"]
        )["measured_total_time"].idxmax()
        tvals_at_max = tvals.loc[idxmax,:]
        ax.axvline(
          x = milestones[idx], linestyle = "dashed", linewidth = 2, color = "k"
        )
        avgdev = tvals_at_max["centralized_dev"].mean()
        avgmtt = tvals_at_max["measured_total_time"].mean()
        ax.plot(
          [avgmtt], 
          [avgdev], 
          '*', 
          color = mcolors.TABLEAU_COLORS["tab:red"],
          markersize = 10
        )
        # -- add to metrics
        mmm = compute_minmaxavg_in_milestone(tvals_at_max, milestones[idx])
        metrics_by_milestones = pd.concat(
          [metrics_by_milestones, mmm], ignore_index = True
        )
    tvals = vals[vals["measured_total_time"] >= milestones[-1]]
    if len(tvals) > 0:
      idxmax = tvals.groupby(
        ["Nn", "seed", "time"]
      )["measured_total_time"].idxmax()
      tvals_at_max = tvals.loc[idxmax,:]
      avgdev = tvals_at_max["centralized_dev"].mean()
      avgmtt = tvals_at_max["measured_total_time"].mean()
      ax.plot(
        [avgmtt], 
        [avgdev], 
        '*', 
        color = mcolors.TABLEAU_COLORS["tab:red"],
        markersize = 10
      )
      # -- add to metrics
      mmm = compute_minmaxavg_in_milestone(tvals_at_max, 3600)
      metrics_by_milestones = pd.concat(
        [metrics_by_milestones, mmm], ignore_index = True
      )
    ax.set_xlabel("Runtime [s]", fontsize = fontsize)
    ax.set_ylabel(
      "Objective deviation\n((FaaS-MACrO - LMM) / LMM) [%]", 
      fontsize = fontsize
    )
    plt.savefig(
      os.path.join(output_folder, f"progressive_dev-Nn_{Nn}.png"),
      dpi = 300,
      format = "png",
      bbox_inches = "tight"
    )
    plt.close()
    metrics_by_milestones.to_csv(
      os.path.join(output_folder, "metrics_by_milestones.csv"), index = False
    )


def find_best_iterations(
    experiment_folder: str
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
  # build folder to store the analysis outcomes
  output_folder = os.path.join(experiment_folder, "postprocessing")
  os.makedirs(output_folder, exist_ok = True)
  # get the number of nodes
  Nn = None
  with open(
      os.path.join(experiment_folder, "base_instance_data.json"), "r"
    ) as istream:
    data = json.load(istream)
    Nn = int(data["None"]["Nn"]["None"])
  # check the method to consider
  method_name = "faas-macro" if os.path.exists(
      os.path.join(experiment_folder, "LSP_solution.csv")
    ) else "faas-madea"
  # parse logs file
  exp = os.path.basename(experiment_folder)
  logs_df, best_sol_df = parse_log_file(
    experiment_folder, 
    exp,
    pd.DataFrame(),
    {"social_welfare": pd.DataFrame(), "centralized": pd.DataFrame()},
    Nn,
    method_name = method_name
  )
  best_sol_df["social_welfare"] = add_time(
    best_sol_df["social_welfare"], logs_df
  )
  best_sol_df["centralized"] = add_time(
    best_sol_df["centralized"], logs_df
  )
  # save
  best_sol_df["social_welfare"].to_csv(
    os.path.join(output_folder, "best_sw_sol_df.csv"), index = False
  )
  best_sol_df["centralized"].to_csv(
    os.path.join(output_folder, "best_c_sol_df.csv"), index = False
  )
  logs_df.to_csv(os.path.join(output_folder, "logs_df.csv"), index = False)
  # plot
  _, axs = plt.subplots(nrows = 2, ncols = 1, figsize=(20,12), sharex = True)
  last_it = [0]
  if len(best_sol_df["social_welfare"]) > 0:
    for t, data in best_sol_df["social_welfare"].groupby("time"):
      data["x"] = data["best_solution_it"] + last_it[-1]
      data.plot(
        x = "x",
        y = "obj",
        ax = axs[0],
        grid = True,
        marker = ".",
        markersize = 10,
        linewidth = 2,
        # label = f"t = {t}"
        label = None,
        legend = False
      )
      axs[0].axvline(
        x = last_it[-1],
        linestyle = "dotted",
        linewidth = 1,
        color = "k"
      )
      last_it.append(last_it[-1] + data["best_solution_it"].max())
  idx = 0
  if len(best_sol_df["centralized"]) > 0:
    for t, data in best_sol_df["centralized"].groupby("time"):
      data["x"] = data["best_solution_it"] + last_it[idx]
      data.plot(
        x = "x",
        y = "obj",
        ax = axs[1],
        grid = True,
        marker = ".",
        markersize = 10,
        linewidth = 2,
        # label = f"t = {t}"
        label = None,
        legend = False
      )
      axs[1].axvline(
        x = last_it[idx],
        linestyle = "dotted",
        linewidth = 1,
        color = "k"
      )
      if len(best_sol_df["social_welfare"]) == 0:
        last_it.append(last_it[-1] + data["best_solution_it"].max())
      idx += 1
  plt.savefig(
    os.path.join(output_folder, "best_solution_obj.png"),
    dpi = 300,
    format = "png",
    bbox_inches = "tight"
  )
  plt.close()
  # add method name
  best_sol_df["social_welfare"]["method"] = method_name
  best_sol_df["centralized"]["method"] = method_name
  return best_sol_df["social_welfare"], best_sol_df["centralized"]


def main(base_folder: str, milestones: list):
  # build folder to store the analysis outcomes
  output_folder = os.path.join(base_folder, "postprocessing")
  os.makedirs(output_folder, exist_ok = True)
  # load information
  by_social_welfare = pd.DataFrame()
  by_centralized_objective = pd.DataFrame()
  for foldername in os.listdir(base_folder):
    experiment_folder = os.path.join(base_folder, foldername)
    if os.path.isdir(experiment_folder) and not foldername.startswith("."):
      if (
          "LSP_solution.csv" in os.listdir(experiment_folder) or 
            "LSPc_solution.csv" in os.listdir(experiment_folder)
        ):
        print(foldername)
        sw, cobj = find_best_iterations(experiment_folder)
        by_social_welfare = pd.concat(
          [by_social_welfare, sw], ignore_index = True
        )
        by_centralized_objective = pd.concat(
          [by_centralized_objective, cobj], ignore_index = True
        )
  # analyze final result
  last_by_sw = by_social_welfare.groupby(
    ["method", "exp", "Nn", "time"]
  ).last()
  last_by_cobj = by_centralized_objective.groupby(
    ["method", "exp", "Nn", "time"]
  ).last()
  analyze_final_results(last_by_sw, last_by_cobj, output_folder)
  # compute progressive deviation
  compute_progressive_deviation(
    by_centralized_objective, milestones, output_folder
  )


if __name__ == "__main__":
  args = parse_arguments()
  base_folder = args.base_folder
  milestones = [int(m) for m in args.milestones]
  main(base_folder, milestones)

