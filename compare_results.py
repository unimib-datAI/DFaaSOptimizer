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
  f1, axs = plt.subplots(nrows = 3, ncols = 1, sharex = True, figsize = (8,18))
  f2, ax2 = plt.subplots(nrows = 2, ncols = 1, sharex = True, figsize = (8,12))
  bplots = [None] * 5
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
      fontsize = 14
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
      fontsize = 14,
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
      fontsize = 14
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
      fontsize = 14
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
      fontsize = 14
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
    fontsize = 14
  )
  axs[0].set_title(None)
  axs[1].set_ylabel(
    "Runtime deviation\n(SP/coord / LMM) [x]",
    fontsize = 14
  )
  axs[1].set_title(None)
  axs[2].set_ylabel(
    "Rejections deviation\n(SP/coord - LMM) [%]",
    fontsize = 14
  )
  axs[2].set_title(None)
  ax2[0].set_ylabel(
    "Number of iterations",
    fontsize = 14
  )
  ax2[0].set_title(None)
  ax2[1].set_ylabel(
    "Best iteration",
    fontsize = 14
  )
  ax2[1].set_title(None)
  # -- x
  axs[2].set_xlabel(
    "Number of agents",
    fontsize = 14
  )
  ax2[1].set_xlabel(
    "Number of agents",
    fontsize = 14
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
    nrows = 3, ncols = 2, sharex = True, sharey = "row", figsize = (16,18)
  )
  bplots = [None] * 3
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
      fontsize = 14
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
      fontsize = 14
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
      fontsize = 14
    )
  )
  # axis properties
  # -- y
  axs[0,0].set_ylabel(
    "Objective function value",
    fontsize = 14
  )
  # axs[0].set_title(None)
  axs[1,0].set_ylabel(
    "Runtime [s]",
    fontsize = 14
  )
  # axs[1].set_title(None)
  axs[2,0].set_ylabel(
    "Percentage number of rejections [%]",
    fontsize = 14
  )
  # axs[2].set_title(None)
  # -- x
  axs[2,0].set_xlabel(
    "Number of agents",
    fontsize = 14
  )
  axs[2,1].set_xlabel(
    "Number of agents",
    fontsize = 14
  )
  # colors
  colors = [
    mcolors.CSS4_COLORS["lightgreen"],
    mcolors.CSS4_COLORS["lightpink"]
  ]
  for (keys, bplot) in bplots:
    for key, color in zip(keys, colors):
      for patch in bplot[key]["boxes"]:
        patch.set_facecolor(color)
      for median in bplot[key]['medians']:
        median.set_color(mcolors.TABLEAU_COLORS["tab:orange"])
      for mean in bplot[key]['means']:
        mean.set_markerfacecolor(mcolors.TABLEAU_COLORS["tab:red"])
        mean.set_markeredgecolor(mcolors.TABLEAU_COLORS["tab:red"])
  plt.savefig(
    os.path.join(plot_folder, "box_detailed.png"),
    dpi = 300,
    format = "png",
    bbox_inches = "tight"
  )


if __name__ == "__main__":
  postprocessing_folder = "solutions/homogeneous_demands/heterogeneous_memory_req/Nf_5-alpha_0.5_1.0-beta_0.5_0.95-p_1.0-greedy_product-maxit_50-pat_10/postprocessing"
  compare_results(postprocessing_folder)
