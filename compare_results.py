from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import os


def compare_results(postprocessing_folder: str):
  obj = pd.read_csv(os.path.join(postprocessing_folder, "obj.csv"))
  rej = pd.read_csv(os.path.join(postprocessing_folder, "rejections.csv"))
  runtime = pd.read_csv(os.path.join(postprocessing_folder, "runtime.csv"))
  dev_plot_by_Nn(obj, runtime, rej, postprocessing_folder)
  plot_by_Nn(obj, runtime, rej, postprocessing_folder)


def dev_plot_by_Nn(
    obj: pd.DataFrame, 
    runtime: pd.DataFrame, 
    rej: pd.DataFrame, 
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
    obj[["Nn", "dev"]].plot.box(
      by = "Nn",
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
    runtime[["Nn", "dev"]].plot.box(
      by = "Nn",
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
    rej[["Nn", "dev"]].plot.box(
      by = "Nn",
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
    runtime[["Nn", "iteration"]].plot.box(
      by = "Nn",
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
    runtime[["Nn", "best_iteration"]].plot.box(
      by = "Nn",
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
    "Number of agents",
    fontsize = fontsize
  )
  ax2[1].set_xlabel(
    "Number of agents",
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


def plot_by_Nn(
    obj: pd.DataFrame, 
    runtime: pd.DataFrame, 
    rej: pd.DataFrame, 
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
    obj[["Nn", "LoadManagementModel", "SP/coord"]].plot.box(
      by = "Nn",
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
    runtime[["Nn", "LoadManagementModel", "SP/coord"]].plot.box(
      by = "Nn",
      grid = True,
      ax = axs[1,:],
      showmeans = True,
      patch_artist = True,
      meanprops = dict(color = mcolors.TABLEAU_COLORS["tab:red"]),
      return_type = "dict",
      fontsize = fontsize,
      logy = True
    )
  )
  bplots[2] = (
    ["LoadManagementModel", "SP/coord"],
    rej[["Nn", "LoadManagementModel", "SP/coord"]].plot.box(
      by = "Nn",
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
  axs[1,0].set_xlabel(
    "Number of agents",
    fontsize = fontsize
  )
  axs[1,1].set_xlabel(
    "Number of agents",
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


if __name__ == "__main__":
  postprocessing_folders = [
    "/Users/federicafilippini/Documents/ServerBackups/my_gurobi_vm/fixed_sum_auto/2024_RussoRusso-3classes-fixed_sum_auto_avg-0_10-spcoord_greedy"
  ]
  for postprocessing_folder in postprocessing_folders:
    print(postprocessing_folder)
    compare_results(os.path.join(postprocessing_folder, "postprocessing"))
