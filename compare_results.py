from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import os


def compare_results(postprocessing_folder: str):
  obj = pd.read_csv(os.path.join(postprocessing_folder, "obj.csv"))
  runtime = pd.read_csv(os.path.join(postprocessing_folder, "runtime.csv"))
  plot_by_Nn(obj, runtime, postprocessing_folder)


def plot_by_Nn(obj: pd.DataFrame, runtime: pd.DataFrame, plot_folder: str):
  _, axs = plt.subplots(nrows = 4, ncols = 1, sharex = True, figsize = (8,24))
  bplots = [None] * 4
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
      fontsize = 14
    )
  )
  bplots[2] = (
    "iteration",
    runtime[["Nn", "iteration"]].plot.box(
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
    "best_iteration",
    runtime[["Nn", "best_iteration"]].plot.box(
      by = "Nn",
      grid = True,
      ax = axs[3],
      showmeans = True,
      patch_artist = True,
      meanprops = dict(color = mcolors.TABLEAU_COLORS["tab:red"]),
      return_type = "dict",
      fontsize = 14
    )
  )
  # axis properties
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
    "Number of iterations",
    fontsize = 14
  )
  axs[2].set_title(None)
  axs[3].set_ylabel(
    "Best iteration",
    fontsize = 14
  )
  axs[3].set_title(None)
  axs[3].set_xlabel(
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
  plt.savefig(
    os.path.join(plot_folder, "box.png"),
    dpi = 300,
    format = "png",
    bbox_inches = "tight"
  )


if __name__ == "__main__":
  postprocessing_folder = "solutions/homogeneous_demands/Nf_4-alpha_0.5_1.0-beta_0.75_1.25-p_1.0-model/postprocessing"
  compare_results(postprocessing_folder)
