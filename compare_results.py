from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
from parse import parse
import pandas as pd
import os


def compare_across_folders(
    postprocessing_folders: list, 
    str_format: str, 
    key_label: str,
    plot_folder: str
  ):
  all_obj = pd.DataFrame()
  all_rej = pd.DataFrame()
  all_runtime = pd.DataFrame()
  for postprocessing_folder in postprocessing_folders:
    print(postprocessing_folder)
    obj, rej, runtime = compare_results(
      os.path.join(postprocessing_folder, "postprocessing")
    )
    key, key_val = parse(str_format, os.path.basename(postprocessing_folder))
    # add info
    obj[key] = float(key_val)
    rej[key] = float(key_val)
    runtime[key] = float(key_val)
    # merge
    all_obj = pd.concat([all_obj, obj])
    all_rej = pd.concat([all_rej, rej])
    all_runtime = pd.concat([all_runtime, runtime])
  # plot
  os.makedirs(plot_folder, exist_ok = True)
  dev_plot_by_key(all_obj, all_runtime, all_rej, key, key_label, plot_folder)
  plot_by_key(all_obj, all_runtime, all_rej, key, key_label, plot_folder)
  # save
  all_obj.to_csv(os.path.join(plot_folder, "obj.csv"))
  all_rej.to_csv(os.path.join(plot_folder, "rejections.csv"))
  all_runtime.to_csv(os.path.join(plot_folder, "runtime.csv"))


def compare_results(postprocessing_folder: str):
  obj = pd.read_csv(os.path.join(postprocessing_folder, "obj.csv"))
  rej = pd.read_csv(os.path.join(postprocessing_folder, "rejections.csv"))
  runtime = pd.read_csv(os.path.join(postprocessing_folder, "runtime.csv"))
  dev_plot_by_key(
    obj, runtime, rej, "Nn", "Number of agents", postprocessing_folder
  )
  plot_by_key(
    obj, runtime, rej, "Nn", "Number of agents", postprocessing_folder
  )
  return obj, rej, runtime


def dev_plot_by_key(
    obj: pd.DataFrame, 
    runtime: pd.DataFrame, 
    rej: pd.DataFrame, 
    key: str,
    label: str,
    plot_folder: str
  ):
  f1, axs = plt.subplots(
    nrows = 3, ncols = 1, sharex = True, figsize = (8,18), 
    gridspec_kw = {"hspace": 0.01}
  )
  f2, ax2 = plt.subplots(
    nrows = 2, ncols = 1, sharex = True, figsize = (8,12),
    gridspec_kw = {"hspace": 0.01}
  )
  bplots = [None] * 5
  fontsize = 21
  bplots[0] = (
    "dev",
    obj[[key, "dev"]].plot.box(
      by = key,
      grid = True,
      ax = axs[0],
      showmeans = True,
      patch_artist = True,
      meanprops = dict(color = mcolors.TABLEAU_COLORS["tab:red"]),
      return_type = "dict",
      fontsize = fontsize
    )
  )
  bplots[1] = (
    "dev",
    runtime[[key, "dev"]].plot.box(
      by = key,
      grid = True,
      ax = axs[1],
      showmeans = True,
      patch_artist = True,
      meanprops = dict(color = mcolors.TABLEAU_COLORS["tab:red"]),
      return_type = "dict",
      fontsize = fontsize,
      logy = True
    )
  )
  bplots[2] = (
    "dev",
    rej[[key, "dev"]].plot.box(
      by = key,
      grid = True,
      ax = axs[2],
      showmeans = True,
      patch_artist = True,
      meanprops = dict(color = mcolors.TABLEAU_COLORS["tab:red"]),
      return_type = "dict",
      fontsize = fontsize
    )
  )
  bplots[3] = (
    "iteration",
    runtime[[key, "iteration"]].plot.box(
      by = key,
      grid = True,
      ax = ax2[0],
      showmeans = True,
      patch_artist = True,
      meanprops = dict(color = mcolors.TABLEAU_COLORS["tab:red"]),
      return_type = "dict",
      fontsize = fontsize
    )
  )
  bplots[4] = (
    "best_iteration",
    runtime[[key, "best_iteration"]].plot.box(
      by = key,
      grid = True,
      ax = ax2[1],
      showmeans = True,
      patch_artist = True,
      meanprops = dict(color = mcolors.TABLEAU_COLORS["tab:red"]),
      return_type = "dict",
      fontsize = fontsize
    )
  )
  # horizontal lines (for reference)
  axs[0].axhline(
    y = 0,
    linestyle = "dashed",
    linewidth = 2,
    color = "k"
  )
  axs[1].axhline(
    y = 1,
    linestyle = "dashed",
    linewidth = 2,
    color = "k"
  )
  axs[2].axhline(
    y = 0,
    linestyle = "dashed",
    linewidth = 2,
    color = "k"
  )
  # axis properties
  # -- y
  axs[0].set_ylabel(
    "Objective deviation\n((SP/coord - LMM) / LMM) [%]",
    fontsize = fontsize
  )
  axs[0].set_title(None)
  axs[1].set_ylabel(
    "Runtime deviation\n(SP/coord / LMM) [x]",
    fontsize = fontsize
  )
  axs[1].set_title(None)
  axs[2].set_ylabel(
    "Cloud offloading deviation\n(SP/coord - LMM) [%]",
    fontsize = fontsize
  )
  axs[2].set_title(None)
  ax2[0].set_ylabel(
    "Number of iterations",
    fontsize = fontsize
  )
  ax2[0].set_title(None)
  ax2[1].set_ylabel(
    "Best iteration",
    fontsize = fontsize
  )
  ax2[1].set_title(None)
  # -- x
  axs[2].set_xlabel(
    label,
    fontsize = fontsize
  )
  ax2[1].set_xlabel(
    label,
    fontsize = fontsize
  )
  # colors
  for (key, bplot) in bplots:
    for patch in bplot[key]["boxes"]:
      patch.set_facecolor(mcolors.CSS4_COLORS["lightskyblue"])
    for median in bplot[key]['medians']:
      median.set_color(mcolors.TABLEAU_COLORS["tab:orange"])
    for mean in bplot[key]['means']:
      mean.set_markerfacecolor(mcolors.TABLEAU_COLORS["tab:red"])
      mean.set_markeredgecolor(mcolors.TABLEAU_COLORS["tab:red"])
  f1.savefig(
    os.path.join(plot_folder, "box.png"),
    dpi = 300,
    format = "png",
    bbox_inches = "tight"
  )
  f2.savefig(
    os.path.join(plot_folder, "box_iterations.png"),
    dpi = 300,
    format = "png",
    bbox_inches = "tight"
  )
  plt.close()


def plot_by_key(
    obj: pd.DataFrame, 
    runtime: pd.DataFrame, 
    rej: pd.DataFrame, 
    key: str,
    label: str,
    plot_folder: str
  ):
  _, axs = plt.subplots(
    nrows = 3, ncols = 2, sharex = True, sharey = "row", figsize = (16,18), 
    gridspec_kw = {"hspace": 0.01, "wspace": 0.01}
  )
  bplots = [None] * 3
  fontsize = 21
  bplots[0] = (
    ["LoadManagementModel", "SP/coord"],
    obj[[key, "LoadManagementModel", "SP/coord"]].plot.box(
      by = key,
      grid = True,
      ax = axs[0,:],
      showmeans = True,
      patch_artist = True,
      meanprops = dict(color = mcolors.TABLEAU_COLORS["tab:red"]),
      return_type = "dict",
      fontsize = fontsize
    )
  )
  bplots[1] = (
    ["LoadManagementModel", "SP/coord"],
    runtime[[key, "LoadManagementModel", "SP/coord"]].plot.box(
      by = key,
      grid = True,
      ax = axs[1,:],
      showmeans = True,
      patch_artist = True,
      meanprops = dict(color = mcolors.TABLEAU_COLORS["tab:red"]),
      return_type = "dict",
      fontsize = fontsize,
      # logy = True
    )
  )
  bplots[2] = (
    ["LoadManagementModel", "SP/coord"],
    rej[[key, "LoadManagementModel", "SP/coord"]].plot.box(
      by = key,
      grid = True,
      ax = axs[2,:],
      showmeans = True,
      patch_artist = True,
      meanprops = dict(color = mcolors.TABLEAU_COLORS["tab:red"]),
      return_type = "dict",
      fontsize = fontsize
    )
  )
  # axis properties
  # -- y
  axs[0,0].set_ylabel(
    "Objective function value",
    fontsize = fontsize
  )
  axs[1,0].set_ylabel(
    "Runtime [s]",
    fontsize = fontsize
  )
  axs[2,0].set_ylabel(
    "Cloud offloading [%]",
    fontsize = fontsize
  )
  # -- x
  axs[2,0].set_xlabel(
    label,
    fontsize = fontsize
  )
  axs[2,1].set_xlabel(
    label,
    fontsize = fontsize
  )
  # -- title
  axs[0,0].set_title(None)
  axs[0,1].set_title(None)
  axs[1,0].set_title(None)
  axs[1,1].set_title(None)
  axs[2,0].set_title(None)
  axs[2,1].set_title(None)
  # colors
  colors = [
    mcolors.CSS4_COLORS["lightgreen"],
    mcolors.CSS4_COLORS["lightpink"]
  ]
  for ridx, (keys, bplot) in enumerate(bplots):
    for cidx, (key, color) in enumerate(zip(keys, colors)):
      for patch in bplot[key]["boxes"]:
        patch.set_facecolor(color)
      for median in bplot[key]['medians']:
        median.set_color(mcolors.TABLEAU_COLORS["tab:orange"])
      for mean in bplot[key]['means']:
        mean.set_markerfacecolor(mcolors.TABLEAU_COLORS["tab:red"])
        mean.set_markeredgecolor(mcolors.TABLEAU_COLORS["tab:red"])
      # -- legend
      if ridx == 0:
        axs[0,cidx].legend([bplot[key]["boxes"][0]], [key], fontsize=fontsize)
  plt.savefig(
    os.path.join(plot_folder, "box_detailed.png"),
    dpi = 300,
    format = "png",
    bbox_inches = "tight"
  )
  plt.close()


if __name__ == "__main__":
  postprocessing_folders = [
    "/Users/federicafilippini/Documents/ServerBackups/my_gurobi_vm/fixed_sum_auto/2024_RussoRusso-3classes-fixed_sum_auto_avg-0_10-k_10-eef_0.1-spcoord_greedy",
    "/Users/federicafilippini/Documents/ServerBackups/my_gurobi_vm/fixed_sum_auto/2024_RussoRusso-3classes-fixed_sum_auto_avg-0_10-k_10-eef_0.25-spcoord_greedy",
    "/Users/federicafilippini/Documents/ServerBackups/my_gurobi_vm/fixed_sum_auto/2024_RussoRusso-3classes-fixed_sum_auto_avg-0_10-k_10-eef_0.5-spcoord_greedy",
    "/Users/federicafilippini/Documents/ServerBackups/my_gurobi_vm/fixed_sum_auto/2024_RussoRusso-3classes-fixed_sum_auto_avg-0_10-k_10-eef_0.75-spcoord_greedy"
  ]
  # for postprocessing_folder in postprocessing_folders:
  #   print(postprocessing_folder)
  #   compare_results(os.path.join(postprocessing_folder, "postprocessing"))
  compare_across_folders(
    postprocessing_folders, 
    "024_RussoRusso-3classes-fixed_sum_auto_avg-0_10-k_10-{}_{}-spcoord_greedy",
    "Edge-exposed fraction",
    "/Users/federicafilippini/Documents/ServerBackups/my_gurobi_vm/fixed_sum_auto/postprocessing_by_eef-spcoord_greedy"
  )