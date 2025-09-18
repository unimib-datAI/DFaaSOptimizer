from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
from parse import parse
import seaborn as sns
import pandas as pd
import argparse
import os


def parse_arguments() -> argparse.Namespace:
  """
  Parse input arguments
  """
  parser: argparse.ArgumentParser = argparse.ArgumentParser(
    description = "Compare results", 
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument(
    "-i", "--postprocessing_folders",
    help = "Results folder (or list of result folders)",
    nargs = "+",
    required = True
  )
  parser.add_argument(
    "--run",
    help = "What to do",
    choices = [
      "compare_results", "compare_across_folders", "compare_single_model"
    ],
    default = "compare_results"
  )
  parser.add_argument(
    "-o", "--common_output_folder",
    help = "Path to a folder where to save plots",
    type = str,
    default = None
  )
  parser.add_argument(
  "--loop_over",
  help = "Key to loop over",
  type = str,
  default = "Nn"
  )
  parser.add_argument(
  "--loop_over_label",
  help = "Label to be attached to the loop-over key",
  type = str,
  default = "Number of agents"
  )
  parser.add_argument(
    "--models",
    help = "List of model names",
    nargs = "*",
    default = ["LoadManagementModel", "FaaS-MACrO"]
  )
  parser.add_argument(
  "--filter_by",
  help = "Key to filter",
  type = str,
  default = None
  )
  parser.add_argument(
  "--keep_only",
  help = "Unique value to keep",
  type = int,
  default = None
  )
  parser.add_argument(
  "--drop_value",
  help = "Unique value to drop",
  type = int,
  default = None
  )
  parser.add_argument(
  "--folder_parse_format",
  help = "Format to parse the folder name",
  type = str,
  default = None
  )
  parser.add_argument(
  "--single_model_baseline",
  help = "Baseline for time comparison",
  type = int,
  default = None
  )
  # Parse the arguments
  args: argparse.Namespace = parser.parse_known_args()[0]
  return args


def compare_across_folders(
    postprocessing_folders: list, 
    str_format: str, 
    key_label: str,
    models: list,
    plot_folder: str,
    filter_by = None,
    keep_only = None,
    drop_value = None
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
    if filter_by is not None and filter_by in obj and filter_by in runtime:
      if keep_only is not None:
        obj = obj[obj[filter_by] == keep_only]
        rej = rej[rej[filter_by] == keep_only]
        runtime = runtime[runtime[filter_by] == keep_only]
      elif drop_value is not None:
        obj = obj[obj[filter_by] != drop_value]
        rej = rej[rej[filter_by] != drop_value]
        runtime = runtime[runtime[filter_by] != drop_value]
    # merge
    all_obj = pd.concat([all_obj, obj])
    all_rej = pd.concat([all_rej, rej])
    all_runtime = pd.concat([all_runtime, runtime])
  # 
  if "ScaledOnSumLMM" in models:
    all_obj.rename(
      columns = {"ScaledOnSumLMM": "LoadManagementModel"}, inplace = True
    )
    all_rej.rename(
      columns = {"ScaledOnSumLMM": "LoadManagementModel"}, inplace = True
    )
    all_runtime.rename(
      columns = {"ScaledOnSumLMM": "LoadManagementModel"}, inplace = True
    )
    models = ["LoadManagementModel", "FaaS-MACrO"]
  # plot
  os.makedirs(plot_folder, exist_ok = True)
  dev_plot_by_key(all_obj, all_runtime, all_rej, key, key_label, plot_folder)
  dev_barplot_by_key(all_obj, all_runtime, all_rej, key, key_label, plot_folder)
  plot_by_key(
    all_obj, 
    all_runtime, 
    all_rej, 
    models,
    key, 
    key_label, 
    plot_folder
  )
  violinplot_by_key(
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
  for k in ["LSP", "SP/coord"]:
    if k in obj:
      obj.rename(columns = {k: "FaaS-MACrO"}, inplace = True)
  rej = None
  if os.path.exists(os.path.join(postprocessing_folder, "rejections.csv")):
    rej = pd.read_csv(os.path.join(postprocessing_folder, "rejections.csv"))
    for k in ["LSP", "SP/coord"]:
      if k in rej:
        rej.rename(columns = {k: "FaaS-MACrO"}, inplace = True)
  runtime = pd.read_csv(os.path.join(postprocessing_folder, "runtime.csv"))
  for k in ["LSP", "SP/coord"]:
    if k in runtime:
      runtime.rename(columns = {k: "FaaS-MACrO"}, inplace = True)
  dev_plot_by_key(
    obj, runtime, rej, key, key_label, postprocessing_folder
  )
  dev_barplot_by_key(obj, runtime, rej, key, key_label, postprocessing_folder)
  plot_by_key(
    obj, 
    runtime, 
    rej, 
    models,
    key, 
    key_label, 
    postprocessing_folder
  )
  violinplot_by_key(
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
    obj.rename(columns = {"obj": "ScaledOnSumLMM"}, inplace = True)
    # -- runtime
    runtime = pd.read_csv(
      os.path.join(postprocessing_folder, "postprocessing", "runtime.csv")
    )
    runtime.rename(columns= {"runtime": "ScaledOnSumLMM"}, inplace = True)
    # add info
    key, key_val = parse(str_format, os.path.basename(postprocessing_folder))
    obj[key] = int(key_val)
    runtime[key] = int(key_val)
    if filter_by is not None and filter_by in obj and filter_by in runtime:
      if keep_only is not None:
        obj = obj[obj[filter_by] == keep_only]
        runtime = runtime[runtime[filter_by] == keep_only]
      elif drop_value is not None:
        obj = obj[obj[filter_by] != drop_value]
        runtime = runtime[runtime[filter_by] != drop_value]
    # merge
    all_obj = pd.concat([all_obj, obj])
    all_runtime = pd.concat([all_runtime, runtime])
  # plot
  os.makedirs(plot_folder, exist_ok = True)
  plot_by_key(
    all_obj, 
    all_runtime, 
    None,
    ["ScaledOnSumLMM"],
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
          dev["ScaledOnSumLMM"].values - 
            dev["ScaledOnSumLMM_baseline"].values
        ) / dev["ScaledOnSumLMM"].values * 100
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
          rt_dev["ScaledOnSumLMM"].values / 
            rt_dev["ScaledOnSumLMM_baseline"].values
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
    "Objective deviation\n((FaaS-MACrO - LMM) / LMM) [%]",
    fontsize = fontsize
  )
  axs[0].set_title(None)
  axs[1].set_ylabel(
    "Runtime deviation\n(FaaS-MACrO / LMM) [x]",
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
      "Cloud offloading deviation\n(FaaS-MACrO - LMM) [%]",
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


def dev_barplot_by_key(
    obj: pd.DataFrame, 
    runtime: pd.DataFrame, 
    rej: pd.DataFrame, 
    key: str,
    label: str,
    plot_folder: str
  ):
  nrows = 1
  ncols = 3 if rej is not None else 2
  # f1, axs = plt.subplots(
  #   nrows = nrows, ncols = ncols, figsize = (12 * ncols, 6 * nrows), 
  #   gridspec_kw = {"wspace": 0.2}
  # )
  # bplots = [None] * ncols
  fontsize = 21
  # bplots[0] = (
  #   "dev",
  #   sns.violinplot(
  #     data = obj,#[[key, "dev"]],
  #     x = key,
  #     y = "dev",
  #     ax = axs[0],
  #     inner = "quart",
  #     color = mcolors.CSS4_COLORS["lightskyblue"],
  #     density_norm = "count"
  #   )
  # )
  # bplots[1] = (
  #   "dev",
  #   sns.violinplot(
  #     data = runtime,#[[key, "dev"]],
  #     x = key,
  #     y = "dev",
  #     ax = axs[1],
  #     log_scale = (False,True),
  #     inner = "quart",
  #     color = mcolors.CSS4_COLORS["lightskyblue"],
  #     density_norm = "count"
  #   )
  # )
  # if rej is not None:
  #   bplots[2] = (
  #     "dev",
  #     sns.violinplot(
  #       data = rej,#[[key, "dev"]],
  #       x = key,
  #       y = "dev",
  #       ax = axs[2],
  #       inner = "quart",
  #       color = mcolors.CSS4_COLORS["lightskyblue"],
  #       density_norm = "count"
  #     )
  #   )
  # # horizontal lines (for reference)
  # axs[0].axhline(
  #   y = 0,
  #   linestyle = "dashed",
  #   linewidth = 2,
  #   color = "k"
  # )
  # axs[1].axhline(
  #   y = 1,
  #   linestyle = "dashed",
  #   linewidth = 2,
  #   color = "k"
  # )
  # # axis properties
  # # -- y
  # axs[0].set_ylabel(
  #   "Objective deviation\n((FaaS-MACrO - LMM) / LMM) [%]",
  #   fontsize = fontsize
  # )
  # axs[1].set_ylabel(
  #   "Runtime deviation\n(FaaS-MACrO / LMM) [x]",
  #   fontsize = fontsize
  # )
  # if rej is not None:
  #   axs[2].axhline(
  #     y = 0,
  #     linestyle = "dashed",
  #     linewidth = 2,
  #     color = "k"
  #   )
  #   axs[2].set_ylabel(
  #     "Cloud offloading deviation\n(FaaS-MACrO - LMM) [%]",
  #     fontsize = fontsize
  #   )
  # # -- common properties
  # plt.setp(axs, title = None)
  # for idx in range(len(axs)):
  #   axs[idx].set_xlabel(
  #     label,
  #     fontsize = fontsize
  #   )
  #   axs[idx].grid(True)
  #   axs[idx].set_xticks(
  #     axs[idx].get_xticks(), 
  #     axs[idx].get_xticklabels(), 
  #     fontsize = fontsize
  #   )
  #   axs[idx].set_yticks(
  #     axs[idx].get_yticks(), 
  #     axs[idx].get_yticklabels(), 
  #     fontsize = fontsize
  #   )
  #   axs[idx].legend(fontsize = fontsize)
  # f1.savefig(
  #   os.path.join(plot_folder, "violin.png"),
  #   dpi = 300,
  #   format = "png",
  #   bbox_inches = "tight"
  # )
  # plt.close()
  ##
  f2, axs2 = plt.subplots(
    nrows = nrows, ncols = ncols, figsize = (12 * ncols, 7 * nrows), 
    gridspec_kw = {"wspace": 0.2}
  )
  group = obj.groupby(key)
  data = pd.DataFrame({
    key: list(group.groups.keys()),
    "avg": group.mean()["dev"].values.tolist(),
    "min": group.min()["dev"].values.tolist(),
    "max": group.max()["dev"].values.tolist()
  })
  data.plot.bar(
    x = key,
    ax = axs2[0],
    grid = True,
    fontsize = fontsize,
    rot = 0
  )
  group = runtime.groupby(key)
  data = pd.DataFrame({
    key: list(group.groups.keys()),
    "avg": group.mean()["dev"].values.tolist(),
    "min": group.min()["dev"].values.tolist(),
    "max": group.max()["dev"].values.tolist()
  })
  data.plot.bar(
    x = key,
    ax = axs2[1],
    grid = True,
    fontsize = fontsize,
    rot = 0,
    logy = True
  )
  if ncols > 2:
    group = rej.groupby(key)
    data = pd.DataFrame({
      key: list(group.groups.keys()),
      "avg": group.mean()["dev"].values.tolist(),
      "min": group.min()["dev"].values.tolist(),
      "max": group.max()["dev"].values.tolist()
    })
    data.plot.bar(
      x = key,
      ax = axs2[2],
      grid = True,
      fontsize = fontsize,
      rot = 0
    )
  # horizontal lines (for reference)
  axs2[0].axhline(
    y = 0,
    linestyle = "dashed",
    linewidth = 2,
    color = "k"
  )
  axs2[1].axhline(
    y = 1,
    linestyle = "dashed",
    linewidth = 2,
    color = "k"
  )
  # axis properties
  # -- y
  axs2[0].set_ylabel(
    "Objective deviation\n((FaaS-MACrO - LMM) / LMM) [%]",
    fontsize = fontsize
  )
  axs2[1].set_ylabel(
    "Runtime deviation\n(FaaS-MACrO / LMM) [x]",
    fontsize = fontsize
  )
  if rej is not None:
    axs2[2].axhline(
      y = 0,
      linestyle = "dashed",
      linewidth = 2,
      color = "k"
    )
    axs2[2].set_ylabel(
      "Cloud offloading deviation\n(FaaS-MACrO - LMM) [%]",
      fontsize = fontsize
    )
  # -- common properties
  plt.setp(axs2, title = None)
  for idx in range(len(axs2)):
    axs2[idx].set_xlabel(
      label,
      fontsize = fontsize
    )
    axs2[idx].legend(fontsize = fontsize)
  f2.savefig(
    os.path.join(plot_folder, "bars.png"),
    dpi = 300,
    format = "png",
    bbox_inches = "tight"
  )
  plt.close()


def violinplot_by_key(
    obj: pd.DataFrame, 
    runtime: pd.DataFrame, 
    rej: pd.DataFrame, 
    models: list,
    key: str,
    label: str,
    plot_folder: str
  ):
  nrows = 1
  ncols = 3 if rej is not None else 2
  _, axs = plt.subplots(
    nrows = nrows, 
    ncols = ncols,  
    figsize = (12 * ncols, 6 * nrows), 
    gridspec_kw = {"wspace": 0.15}
  )
  bplots = [None] * ncols
  fontsize = 21
  # -- objective function value
  data = {key: [], "obj": [], "method": []}
  for model in models:
    data[key] += obj[key].values.tolist()
    data["obj"] += obj[model].values.tolist()
    data["method"] += [model] * len(obj)
  data = pd.DataFrame(data)
  bplots[0] = (
    models,
    sns.violinplot(
      data = data,
      x = key,
      y = "obj",
      hue = "method",
      split = True, 
      inner = "quart",
      ax = axs[0],
      palette = {
        models[0]: mcolors.CSS4_COLORS["lightgreen"],
        models[1]: mcolors.CSS4_COLORS["lightpink"]
      },
      density_norm = "count"
    )
  )
  # -- runtime
  data = {key: [], "obj": [], "method": []}
  for model in models:
    data[key] += runtime[key].values.tolist()
    data["obj"] += runtime[model].values.tolist()
    data["method"] += [model] * len(runtime)
  data = pd.DataFrame(data)
  bplots[1] = (
    models,
    sns.violinplot(
      data = data,
      x = key,
      y = "obj",
      hue = "method",
      split = True, 
      inner = "quart",
      ax = axs[1],
      log_scale = (False, True),
      palette = {
        models[0]: mcolors.CSS4_COLORS["lightgreen"],
        models[1]: mcolors.CSS4_COLORS["lightpink"]
      },
      density_norm = "count"
    )
  )
  if ncols > 2:
    # -- rejections
    data = {key: [], "obj": [], "method": []}
    for model in models:
      data[key] += rej[key].values.tolist()
      data["obj"] += rej[model].values.tolist()
      data["method"] += [model] * len(rej)
    data = pd.DataFrame(data)
    bplots[2] = (
      models,
      sns.violinplot(
        data = data,
        x = key,
        y = "obj",
        hue = "method",
        split = True, 
        inner = "quart",
        ax = axs[2],
        palette = {
          models[0]: mcolors.CSS4_COLORS["lightgreen"],
          models[1]: mcolors.CSS4_COLORS["lightpink"]
        },
        density_norm = "count",
        cut = 0
      )
    )
  # axis properties
  # -- y
  axs[0].set_ylabel(
    "Objective function value",
    fontsize = fontsize
  )
  axs[1].set_ylabel(
    "Runtime [s]",
    fontsize = fontsize
  )
  if ncols > 2:
    axs[2].set_ylabel(
      "Cloud offloading [%]",
      fontsize = fontsize
    )
  # -- some common properties
  for idx in range(len(axs)):
    axs[idx].set_xlabel(
      label,
      fontsize = fontsize
    )
    axs[idx].grid(True)
    axs[idx].set_xticks(
      axs[idx].get_xticks(), 
      axs[idx].get_xticklabels(), 
      fontsize = fontsize
    )
    axs[idx].set_yticks(
      axs[idx].get_yticks(), 
      axs[idx].get_yticklabels(), 
      fontsize = fontsize
    )
    axs[idx].legend(fontsize = fontsize)
    # if idx == 0:
    #   axs[idx].legend(fontsize = fontsize)
    # else:
    #   axs[idx].legend().set_visible(False)
  # -- title
  plt.setp(axs, title = None)
  plt.savefig(
    os.path.join(plot_folder, "violin_detailed.png"),
    dpi = 300,
    format = "png",
    bbox_inches = "tight"
  )
  plt.close()


if __name__ == "__main__":
  # parse arguments
  args = parse_arguments()
  postprocessing_folders = args.postprocessing_folders
  common_output_folder = args.common_output_folder
  what_to_do = args.run
  loop_over = args.loop_over
  loop_over_label = args.loop_over_label
  models = args.models
  filter_by = args.filter_by
  keep_only = args.keep_only
  drop_value = args.drop_value
  folder_parse_format = args.folder_parse_format
  single_model_baseline = args.single_model_baseline
  # build list of postprocessing folders if only the base was provided
  if len(postprocessing_folders) == 1:
    if "experiments.json" not in os.listdir(postprocessing_folders[0]):
      base_folder = postprocessing_folders[0]
      postprocessing_folders = [
        os.path.join(
          base_folder, f
        ) for f in os.listdir(base_folder) if not f.startswith(".") and \
          not f.startswith("post")
      ]
  # check what to do
  if what_to_do == "compare_results":
    if common_output_folder is not None:
      print(
        "WARNING: `common_output_folder` is ignored in this setting."
        " All results will be saved in <postprocessing_folder>/postprocessing"
      )
    for postprocessing_folder in postprocessing_folders:
      print(postprocessing_folder)
      compare_results(
        os.path.join(postprocessing_folder, "postprocessing"),
        loop_over,
        loop_over_label,
        models
      )
  elif what_to_do == "compare_across_folders":
    # for eef,
    # loop_over_label = "Edge-exposed fraction [%]"
    # models = ["FaaS-MACrO", "ScaledOnSumLMM"]
    if keep_only is not None and drop_value is not None:
      print(
        "WARNING: both `keep_only` and `drop_value` set."
        " `drop_value` will be ignored"
      )
    if folder_parse_format is None:
      raise KeyError(
        f"It is mandatory to set `folder_parse_format` in {what_to_do} mode"
      )
    compare_across_folders(
      postprocessing_folders, 
      folder_parse_format,
      loop_over_label,
      models,
      common_output_folder,
      filter_by = filter_by,
      keep_only = keep_only,
      drop_value = drop_value
    )
  elif what_to_do == "compare_single_model":
    # usually:
    # loop_over_label = "Time limit [s]"
    # single_model_baseline = 10
    compare_single_model(
      postprocessing_folders, 
      folder_parse_format,
      loop_over_label,
      common_output_folder,
      baseline = single_model_baseline,
      filter_by = filter_by,
      keep_only = keep_only,
      drop_value = drop_value
    )
