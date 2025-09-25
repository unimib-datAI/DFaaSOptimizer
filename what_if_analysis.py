from logs_postprocessing import parse_log_file

from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
from typing import Tuple
import pandas as pd
import json
import os


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
  # parse logs file
  exp = os.path.basename(experiment_folder)
  logs_df, best_sol_df = parse_log_file(
    experiment_folder, 
    exp,
    pd.DataFrame(),
    {"social_welfare": pd.DataFrame(), "centralized": pd.DataFrame()},
    Nn
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
    idx += 1
  plt.savefig(
    os.path.join(output_folder, "best_solution_obj.png"),
    dpi = 300,
    format = "png",
    bbox_inches = "tight"
  )
  plt.close()
  return best_sol_df["social_welfare"], best_sol_df["centralized"]


def main(base_folder: str):
  # build folder to store the analysis outcomes
  output_folder = os.path.join(base_folder, "postprocessing")
  os.makedirs(output_folder, exist_ok = True)
  # load information
  by_social_welfare = pd.DataFrame()
  by_centralized_objective = pd.DataFrame()
  for foldername in os.listdir(base_folder):
    experiment_folder = os.path.join(base_folder, foldername)
    if os.path.isdir(experiment_folder) and not foldername.startswith("."):
      if "LSP_solution.csv" in os.listdir(experiment_folder):
        print(foldername)
        sw, cobj = find_best_iterations(experiment_folder)
        by_social_welfare = pd.concat(
          [by_social_welfare, sw], ignore_index = True
        )
        by_centralized_objective = pd.concat(
          [by_centralized_objective, cobj], ignore_index = True
        )
  last_by_sw = by_social_welfare.groupby(["exp", "Nn", "time"]).last()
  last_by_cobj = by_centralized_objective.groupby(["exp", "Nn", "time"]).last()
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


if __name__ == "__main__":
  base_folder = "solutions/2024_RussoRusso_spcoord2"
  main(base_folder)

