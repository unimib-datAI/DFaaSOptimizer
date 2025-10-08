from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
from parse import parse
import pandas as pd
import os


def compare_across_folders(
    postprocessing_folders: list, 
    str_format: str, 
    key_label: str,
    models: list,
    plot_folder: str
  ):
  all_obj = pd.DataFrame()
  all_rej = pd.DataFrame()
  all_runtime = pd.DataFrame()
  for postprocessing_folder in postprocessing_folders:
    print(postprocessing_folder)
    key, key_val = parse(str_format, os.path.basename(postprocessing_folder))
    obj, rej, runtime = compare_results(
      os.path.join(postprocessing_folder, "postprocessing"), 
      "Nn", 
      "Number of agents",
      models
    )
    # add info
    obj[key] = int(key_val) if key != "eef" else round(
      float(key_val) * 100, 2
    )
    rej[key] = int(key_val) if key != "eef" else round(
      float(key_val) * 100, 2
    )
    runtime[key] = int(key_val) if key != "eef" else round(
      float(key_val) * 100, 2
    )
    # merge
    all_obj = pd.concat([all_obj, obj])
    all_rej = pd.concat([all_rej, rej])
    all_runtime = pd.concat([all_runtime, runtime])
  # 
  if "ScaledOnSumLMM" in models:
    all_obj.rename(columns = {"ScaledOnSumLMM": "LoadManagementModel"}, inplace = True)
    all_rej.rename(columns = {"ScaledOnSumLMM": "LoadManagementModel"}, inplace = True)
    all_runtime.rename(columns = {"ScaledOnSumLMM": "LoadManagementModel"}, inplace = True)
    models = ["LoadManagementModel", "SP/coord"]
  # plot
  os.makedirs(plot_folder, exist_ok = True)
  dev_plot_by_key(all_obj, all_runtime, all_rej, key, key_label, plot_folder)
  plot_by_key(
    all_obj, 
    all_runtime, 
    all_rej, 
    models,
    key, 
    key_label, 
    plot_folder
  )
  # save
  all_obj.to_csv(os.path.join(plot_folder, "obj.csv"))
  all_rej.to_csv(os.path.join(plot_folder, "rejections.csv"))
  all_runtime.to_csv(os.path.join(plot_folder, "runtime.csv"))


def compare_results(
    postprocessing_folder: str, 
    key: str, 
    key_label: str,
    models: list
  ):
  obj = pd.read_csv(os.path.join(postprocessing_folder, "obj.csv"))
  rej = pd.read_csv(os.path.join(postprocessing_folder, "rejections.csv"))
  runtime = pd.read_csv(os.path.join(postprocessing_folder, "runtime.csv"))
  dev_plot_by_key(
    obj, runtime, rej, key, key_label, postprocessing_folder
  )
  plot_by_key(
    obj, 
    runtime, 
    rej, 
    models,
    key, 
    key_label, 
    postprocessing_folder
  )
  return obj, rej, runtime


def compare_single_model(
    postprocessing_folders: list, 
    str_format: str, 
    key_label: str,
    plot_folder: str,
    baseline = None,
    filter_by = None,
    keep_only = None,
    drop_value = None
  ):
  all_obj = pd.DataFrame()
  all_runtime = pd.DataFrame()
  for postprocessing_folder in postprocessing_folders:
    print(postprocessing_folder)
    # -- objective function value
    obj = pd.read_csv(
      os.path.join(postprocessing_folder, "postprocessing", "obj.csv")
    )
    obj.rename(columns = {"obj": "LoadManagementModel"}, inplace = True)
    if filter_by is not None and filter_by in obj:
      if keep_only is not None:
        obj = obj[obj[filter_by] == keep_only]
      elif drop_value is not None:
        obj = obj[obj[filter_by] != drop_value]
    # -- runtime
    runtime = pd.read_csv(
      os.path.join(postprocessing_folder, "postprocessing", "runtime.csv")
    )
    runtime.rename(columns= {"runtime": "LoadManagementModel"}, inplace = True)
    if filter_by is not None and filter_by in obj:
      if keep_only is not None:
        runtime = runtime[runtime[filter_by] == keep_only]
      elif drop_value is not None:
        runtime = runtime[runtime[filter_by] != drop_value]
    # add info
    key, key_val = parse(str_format, os.path.basename(postprocessing_folder))
    obj[key] = int(key_val)
    runtime[key] = int(key_val)
    # merge
    all_obj = pd.concat([all_obj, obj])
    all_runtime = pd.concat([all_runtime, runtime])
  # plot
  os.makedirs(plot_folder, exist_ok = True)
  plot_by_key(
    all_obj, 
    all_runtime, 
    None,
    ["LoadManagementModel"],
    key, 
    key_label, 
    plot_folder
  )
  # save
  all_obj.to_csv(os.path.join(plot_folder, "obj.csv"))
  all_runtime.to_csv(os.path.join(plot_folder, "runtime.csv"))
  # compute deviation (if a baseline is provided)
  if baseline is not None:
    obj_dev = pd.DataFrame()
    runtime_dev = pd.DataFrame()
    bo = all_obj[all_obj[key] == baseline]
    br = all_runtime[all_runtime[key] == baseline]
    for key_val, key_data in all_obj.groupby(key):
      if key_val != baseline:
        # -- obj
        dev = bo.join(
          key_data.set_index(["Nn", "seed", "time"]), 
          on = ["Nn", "seed", "time"], 
          lsuffix = "_baseline"
        )
        df = dev[["Nn", "seed", "time"]].copy(deep = True)
        df["dev"] = (
          dev["LoadManagementModel"].values - 
            dev["LoadManagementModel_baseline"].values
        ) / dev["LoadManagementModel"].values * 100
        df[key] = key_val
        obj_dev = pd.concat([obj_dev, df], ignore_index = True)
        # -- runtime
        rt = all_runtime[all_runtime[key] == key_val]
        rt_dev = br.join(
          rt.set_index(["Nn", "seed", "time"]), 
          on = ["Nn", "seed", "time"], 
          lsuffix = "_baseline"
        )
        df = rt_dev[["Nn", "seed", "time"]].copy(deep = True)
        df["dev"] = (
          rt_dev["LoadManagementModel"].values / 
            rt_dev["LoadManagementModel_baseline"].values
        )
        df[key] = key_val
        runtime_dev = pd.concat([runtime_dev, df], ignore_index = True)
    dev_plot_by_key(obj_dev, runtime_dev, None, key, key_label, plot_folder)


def dev_plot_by_key(
    obj: pd.DataFrame, 
    runtime: pd.DataFrame, 
    rej: pd.DataFrame, 
    key: str,
    label: str,
    plot_folder: str
  ):
  nrows = 3 if rej is not None else 2
  f1, axs = plt.subplots(
    nrows = nrows, ncols = 1, sharex = True, figsize = (8,6 * nrows), 
    gridspec_kw = {"hspace": 0.02}
  )
  f2 = None
  bplots = [None] * nrows
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
  if rej is not None:
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
    axs[2].axhline(
      y = 0,
      linestyle = "dashed",
      linewidth = 2,
      color = "k"
    )
    axs[2].set_ylabel(
      "Cloud offloading deviation\n(SP/coord - LMM) [%]",
      fontsize = fontsize
    )
    axs[2].set_title(None)
  # -- x
  axs[-1].set_xlabel(
    label,
    fontsize = fontsize
  )
  if "iteration" in runtime and "best_iteration" in runtime:
    f2, ax2 = plt.subplots(
      nrows = 2, ncols = 1, sharex = True, figsize = (8,12),
      gridspec_kw = {"hspace": 0.02}
    )
    bplots += [None, None]
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
    # axis properties
    # -- y
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
  if f2 is not None:
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
    models: list,
    key: str,
    label: str,
    plot_folder: str
  ):
  nrows = 3 if rej is not None else 2
  ncols = len(models)
  _, axs = plt.subplots(
    nrows = nrows, 
    ncols = ncols, 
    sharex = True, 
    sharey = "row", 
    figsize = (8 * ncols, 6 * nrows), 
    gridspec_kw = {"hspace": 0.02, "wspace": 0.01}
  )
  bplots = [None] * nrows
  fontsize = 21
  bplots[0] = (
    models,
    obj[[key] + models].plot.box(
      by = key,
      grid = True,
      ax = axs[0] if ncols == 1 else axs[0,:],
      showmeans = True,
      patch_artist = True,
      meanprops = dict(color = mcolors.TABLEAU_COLORS["tab:red"]),
      return_type = "dict",
      fontsize = fontsize
    )
  )
  bplots[1] = (
    models,
    runtime[[key] + models].plot.box(
      by = key,
      grid = True,
      ax = axs[1] if ncols == 1 else axs[1,:],
      showmeans = True,
      patch_artist = True,
      meanprops = dict(color = mcolors.TABLEAU_COLORS["tab:red"]),
      return_type = "dict",
      fontsize = fontsize,
      # logy = True
    )
  )
  if nrows > 2:
    bplots[2] = (
      models,
      rej[[key] + models].plot.box(
        by = key,
        grid = True,
        ax = axs[2] if ncols == 1 else axs[2,:],
        showmeans = True,
        patch_artist = True,
        meanprops = dict(color = mcolors.TABLEAU_COLORS["tab:red"]),
        return_type = "dict",
        fontsize = fontsize
      )
    )
  # axis properties
  # -- y
  if ncols > 1:
    axs[0,0].set_ylabel(
      "Objective function value",
      fontsize = fontsize
    )
    axs[1,0].set_ylabel(
      "Runtime [s]",
      fontsize = fontsize
    )
    if nrows > 2:
      axs[2,0].set_ylabel(
        "Cloud offloading [%]",
        fontsize = fontsize
      )
    # -- x
    axs[-1,0].set_xlabel(
      label,
      fontsize = fontsize
    )
    axs[-1,1].set_xlabel(
      label,
      fontsize = fontsize
    )
  else:
    axs[0].set_ylabel(
      "Objective function value",
      fontsize = fontsize
    )
    axs[1].set_ylabel(
      "Runtime [s]",
      fontsize = fontsize
    )
    if nrows > 2:
      axs[2].set_ylabel(
        "Cloud offloading [%]",
        fontsize = fontsize
      )
    # -- x
    axs[-1].set_xlabel(
      label,
      fontsize = fontsize
    )
  # -- title
  plt.setp(axs, title = None)
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
        if ncols > 1:
          axs[0,cidx].legend(
            [bplot[key]["boxes"][0]], [key], fontsize=fontsize
          )
        else:
          axs[0].legend(
            [bplot[key]["boxes"][0]], [key], fontsize=fontsize
          )
  plt.savefig(
    os.path.join(plot_folder, "box_detailed.png"),
    dpi = 300,
    format = "png",
    bbox_inches = "tight"
  )
  plt.close()


if __name__ == "__main__":
  postprocessing_folders = [
    "/Users/federicafilippini/Documents/ServerBackups/my_gurobi_vm/fixed_sum_auto/2024_RussoRusso-3classes-0_10-varyingF-spcoord_optimal"
  ]
  for postprocessing_folder in postprocessing_folders:
    print(postprocessing_folder)
    compare_results(
      os.path.join(postprocessing_folder, "postprocessing"),
      "Nf",
      "Number of functions",
      ["LoadManagementModel", "SP/coord"]
    )
  # compare_across_folders(
  #   postprocessing_folders, 
  #   "2024_RussoRusso-3classes-fixed_sum_auto_avg-0_10-k_10-{}_{}-spcoord_greedy",
  #   "Edge-exposed fraction [%]",
  #   ["SP/coord", "ScaledOnSumLMM"],
  #   "/Users/federicafilippini/Documents/ServerBackups/my_gurobi_vm/fixed_sum_auto/varyingEef/reversed/postprocessing_by_eef"
  # )
  # compare_single_model(
  #   postprocessing_folders, 
  #   "2024_RussoRusso-3classes-fixed_sum_auto_avg-0_10-centralized_{}_{}",
  #   "Time limit [s]",
  #   "/Users/federicafilippini/Documents/ServerBackups/my_gurobi_vm/fixed_sum_auto/centralized/postprocessing_by_TL",
  #   baseline = 5,
  #   filter_by = "Nn",
  #   drop_value = 200
  # )