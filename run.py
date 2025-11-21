from logs_postprocessing import parse_log_file, get_faasmacro_runtime
from run_centralized_model import load_configuration
from run_centralized_model import run as run_centralized
from run_faasmacro import run as run_iterations
from postprocessing import load_models_results

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from copy import deepcopy
from parse import parse
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import json
import os

import logging
logging.getLogger('pyomo.core').setLevel(logging.ERROR)


def parse_arguments() -> argparse.Namespace:
  """
  Parse input arguments
  """
  parser: argparse.ArgumentParser = argparse.ArgumentParser(
    description = "run", 
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument(
    "-c", "--config",
    help = "Configuration file",
    type = str,
    default = "config.json"
  )
  parser.add_argument(
    "--n_experiments",
    help = "Number of experiments to run for each configuration",
    type = int,
    default = 3
  )
  parser.add_argument(
    "--run_centralized_only",
    default = False,
    action = "store_true"
  )
  parser.add_argument(
    "--run_faasmacro_only",
    default = False,
    action = "store_true"
  )
  parser.add_argument(
    "--postprocessing_only",
    default = False,
    action = "store_true"
  )
  parser.add_argument(
    "--generate_only",
    default = False,
    action = "store_true"
  )
  parser.add_argument(
    "--postprocessing_list",
    default = False,
    action = "store_true"
  )
  parser.add_argument(
    "--fix_r",
    default = False,
    action = "store_true"
  )
  parser.add_argument(
    "-j", "--sp_parallelism",
    help = "Number of parallel processes to start (-1: auto, 0: sequential)",
    type = int,
    default = -1
  )
  parser.add_argument(
    "--enable_plotting",
    default = False,
    action = "store_true"
  )
  parser.add_argument(
    "--loop_over",
    help = "Key to loop over",
    type = str,
    default = "Nn"
  )
  # Parse the arguments
  args: argparse.Namespace = parser.parse_known_args()[0]
  return args


def generate_experiments_list(exp_values, seed, n_experiments):
  rng = np.random.default_rng(seed)
  # list of exp values
  exp_list = exp_values.get("values", [])
  if len(exp_list) == 0:
    step = exp_values.get("step", 1)
    exp_list = list(range(exp_values["min"], exp_values["max"] + step, step))
  # seed(s)
  seed_list = [seed] + rng.integers(
    1000, 10000, endpoint = True, size = (n_experiments - 1,)
  ).tolist()
  # list of experiments
  return [[exp_value, int(s)] for s in seed_list for exp_value in exp_list]


def load_obj_value(solution_folder: str) -> pd.DataFrame:
  obj = pd.DataFrame()
  if os.path.exists(os.path.join(solution_folder, "obj.csv")):
    obj = pd.read_csv(
      os.path.join(solution_folder, "obj.csv")
    )
    obj = obj.loc[:,~obj.columns.str.startswith("Unnamed")]
  return obj


def load_termination_condition(
    solution_folder: str, centralized: bool = False
  ) -> pd.DataFrame:
  tc = pd.DataFrame()
  if os.path.exists(
      os.path.join(solution_folder, "termination_condition.csv")
    ):
    tc = pd.read_csv(
      os.path.join(solution_folder, "termination_condition.csv")
    )
    if not centralized:
      tc.rename(columns = {"Unnamed: 0": "time"}, inplace = True)
      criterion = []
      iteration = []
      deviation = []
      best_it = []
      for s in tc["0"]:
        c, i, d, b, bc, trt = [None] * 6
        if "total runtime" in s:
          if "centralized" in s:
            c, i, d, b, bc, trt = parse(
              "{} (it: {}; obj. deviation: {}; best it: {}; best centralized it: {}; total runtime: {})", 
              s
            )
          else:
            c, i, d, b, trt = parse(
              "{} (it: {}; obj. deviation: {}; best it: {}; total runtime: {})", 
              s
            )
        elif "best it" in s:
          c, i, d, b = parse(
            "{} (it: {}; obj. deviation: {}; best it: {})", 
            s
          )
        else:
          c, i, d = parse(
            "{} (it: {}; obj. deviation: {})", 
            s
          )
        if c.startswith("reached time limit"):
          c1, c2 = parse("reached time limit: {} >= {}", c)
          c = f"reached time limit ({c2})"
        criterion.append(c)
        iteration.append(int(i))
        deviation.append(float(d))
        best_it.append(int(b) if b is not None else b)
      tc.drop("0", axis = "columns", inplace = True)
      tc["criterion"] = criterion
      tc["iteration"] = iteration
      tc["deviation"] = deviation
      tc["best_iteration"] = best_it
    else:
      tc["time"] = tc.index
  return tc


def merge_sol_dict(c_res: dict, i_res: dict) -> pd.DataFrame:
  res = c_res["tot"].join(i_res["tot"])
  res["time"] = "tot"
  for key in c_res:
    if key != "tot":
      time = int(key.split(" ")[-1])
      df = c_res[key].join(i_res[key])
      df["time"] = time
      res = pd.concat([res, df])
  return res


def plot_total_count(df: pd.DataFrame, plot_filename: str):
  df[df["time"]=="tot"].drop(columns = "time").plot.bar(
    rot = 0,
    logy = True
  )
  plt.grid(True, which = "both")
  plt.savefig(plot_filename, dpi = 300, format = "png", bbox_inches = "tight")
  plt.close()


def results_postprocessing(
    solution_folders: dict, 
    base_folder: str,
    loop_over: str
  ):
  # prepare folder to store plots
  plot_folder = os.path.join(base_folder, "postprocessing")
  os.makedirs(plot_folder, exist_ok = True)
  all_obj_values = pd.DataFrame()
  all_rej_values = pd.DataFrame()
  all_runtime_values = pd.DataFrame()
  all_i_tc = pd.DataFrame()
  ping_pong_list = []
  # loop over experiments
  for exp_description_tuple, c_folder, i_folder in zip(
      solution_folders["experiments_list"], 
      solution_folders["centralized"],
      solution_folders["faas-macro"]
    ):
    print(f"Postprocessing exp: {exp_description_tuple}")
    # prepare folder to store results
    exp_description = "_".join([str(s) for s in exp_description_tuple])
    exp_plot_folder = os.path.join(plot_folder, exp_description)
    os.makedirs(exp_plot_folder, exist_ok = True)
    # local_count, fwd_count, rej_count, replicas, ping_pong
    if c_folder is not None and i_folder is not None:
      c_res = load_models_results(c_folder, ["LoadManagementModel"])
      i_res = load_models_results(i_folder, ["LSP"])
      if len(c_res[0]["by_function"]) > 0 and len(i_res[0]["by_function"]) > 0:
        # check ping-pong problems
        if len(c_res[-1]["LoadManagementModel"]) > 0:
          ping_pong_list.append([exp_description, "centralized", c_folder])
        if len(i_res[-1]["LSP"]) > 0:
          ping_pong_list.append([exp_description, "faas-macro", i_folder])
        # merge solutions
        local_count = merge_sol_dict(
          c_res[0]["by_function"], i_res[0]["by_function"]
        )
        fwd_count = merge_sol_dict(
          c_res[1]["by_function"], i_res[1]["by_function"]
        )
        rej_count = merge_sol_dict(
          c_res[2]["by_function"], i_res[2]["by_function"]
        )
        # plot
        plot_total_count(
          local_count, os.path.join(exp_plot_folder, "loc_by_function.png")
        )
        plot_total_count(
          fwd_count, os.path.join(exp_plot_folder, "fwd_by_function.png")
        )
        plot_total_count(
          rej_count, os.path.join(exp_plot_folder, "rej_by_function.png")
        )
        # save
        local_count.to_csv(os.path.join(exp_plot_folder, "loc_by_function.csv"))
        fwd_count.to_csv(os.path.join(exp_plot_folder, "fwd_by_function.csv"))
        rej_count.to_csv(os.path.join(exp_plot_folder, "rej_by_function.csv"))
        # objective function value
        c_obj = load_obj_value(c_folder)
        i_obj = load_obj_value(i_folder)
        obj = c_obj.join(i_obj)
        # total rejections
        all_rej = rej_count.groupby("time").sum()
        all_req = (
          local_count.groupby("time").sum() + fwd_count.groupby("time").sum()
        ) + all_rej
        all_rej = all_rej / all_req * 100
        all_rej.rename(columns = {"LSP": "FaaS-MACrO"}, inplace = True)
        # plot
        _, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (16,6))
        obj.plot(marker = ".", grid = True, ax = axs[0])
        all_rej.drop("tot").plot(marker = ".", grid = True, ax = axs[1])
        axs[0].set_xlabel("Control time period $t$")
        axs[1].set_xlabel("Control time period $t$")
        axs[0].set_ylabel("Objective function value")
        axs[1].set_ylabel("Total percentage of rejections")
        plt.savefig(
          os.path.join(exp_plot_folder, "obj.png"),
          dpi = 300,
          format = "png",
          bbox_inches = "tight"
        )
        plt.close()
        # compute deviation
        obj["dev"] = (
          obj["FaaS-MACrO"] - obj["LoadManagementModel"]
        ) / obj["LoadManagementModel"] * 100
        all_rej["dev"] = (
          all_rej["FaaS-MACrO"] - all_rej["LoadManagementModel"]
        )
        # plot
        _, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (16,6))
        obj["dev"].plot(marker = ".", grid = True, ax = axs[0])
        axs[0].axhline(
          y = obj["dev"].mean(), 
          color = mcolors.TABLEAU_COLORS["tab:red"],
          linewidth = 2
        )
        all_rej["dev"].drop("tot").plot(marker = ".", grid = True, ax = axs[1])
        axs[1].axhline(
          y = all_rej["dev"].drop("tot").mean(), 
          color = mcolors.TABLEAU_COLORS["tab:red"],
          linewidth = 2
        )
        axs[0].set_xlabel("Control time period $t$")
        axs[1].set_xlabel("Control time period $t$")
        axs[0].set_ylabel(
          "Objective function deviation ((FaaS-MACrO - LMM) / LMM )[%]"
        )
        axs[1].set_ylabel(
          "Percentage rejections deviation (FaaS-MACrO - LMM) [%]"
        )
        plt.savefig(
          os.path.join(exp_plot_folder, "obj_deviation.png"),
          dpi = 300,
          format = "png",
          bbox_inches = "tight"
        )
        plt.close()
        # save
        obj.to_csv(os.path.join(exp_plot_folder, "obj.csv"), index = False)
        # merge
        obj["time"] = obj.index
        obj[loop_over] = exp_description_tuple[0]
        obj["seed"] = exp_description_tuple[1]
        all_obj_values = pd.concat([all_obj_values, obj], ignore_index = True)
        #
        all_rej.drop("tot", inplace = True)
        all_rej["time"] = all_rej.index
        all_rej[loop_over] = exp_description_tuple[0]
        all_rej["seed"] = exp_description_tuple[1]
        all_rej_values = pd.concat(
          [all_rej_values, all_rej], ignore_index = True
        )
        # termination condition
        i_tc = load_termination_condition(i_folder)
        _, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (20,5))
        i_tc.plot(
          x = "time",
          y = "iteration",
          marker = ".",
          grid = True,
          ax = axs[0]
        )
        if not i_tc["best_iteration"].isnull().all():
          i_tc.plot(
            x = "time",
            y = "best_iteration",
            marker = ".",
            color = mcolors.TABLEAU_COLORS["tab:orange"],
            grid = True,
            ax = axs[0]
          )
        i_tc["criterion"].value_counts().plot.bar(
          rot = 0,
          grid = True,
          ax = axs[1]
        )
        axs[0].set_xlabel("Control time period $t$")
        axs[0].set_ylabel("Number of iterations")
        axs[1].set_xlabel(None)
        plt.savefig(
          os.path.join(exp_plot_folder, "iterations.png"),
          dpi = 300,
          format = "png",
          bbox_inches = "tight"
        )
        plt.close()
        # -- merge
        i_tc[loop_over] = exp_description_tuple[0]
        i_tc["seed"] = exp_description_tuple[1]
        all_i_tc = pd.concat([all_i_tc, i_tc], ignore_index = True)
    # runtime
    c_runtime = pd.DataFrame()
    i_runtime = pd.DataFrame()
    if c_folder is not None and os.path.exists(
        os.path.join(c_folder, "runtime.csv")
      ):
      c_runtime = pd.read_csv(os.path.join(c_folder, "runtime.csv"))
    if i_folder is not None:
      logs_df, _ = parse_log_file(
        i_folder, 
        exp_description, 
        pd.DataFrame(), 
        {}, 
        int(exp_description_tuple[0])
      )
      i_runtime = get_faasmacro_runtime(logs_df, exp_plot_folder)
    # plot runtime comparison
    if len(c_runtime) > 0 and len(i_runtime) > 0:
      runtime_comparison = pd.DataFrame({
        "LoadManagementModel": c_runtime["LoadManagementModel"],
        "FaaS-MACrO": i_runtime["tot"],
        "iteration": i_tc["iteration"],
        "best_iteration": i_tc["best_iteration"],
      })
      _, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (12,8))
      runtime_comparison.plot(grid = True, marker = ".", ax = axs[0])
      runtime_comparison["dev"] = (
        runtime_comparison["FaaS-MACrO"] / runtime_comparison["LoadManagementModel"]
      )
      runtime_comparison["dev"].plot(
        grid = True, marker = ".", ax = axs[1]
      )
      axs[1].axhline(
        y = runtime_comparison["dev"].mean(),
        color = mcolors.TABLEAU_COLORS["tab:red"]
      )
      axs[0].set_ylabel("Runtime [s]", fontsize = 14)
      axs[1].set_ylabel("Runtime deviation [x]", fontsize = 14)
      plt.savefig(
        os.path.join(exp_plot_folder, "runtime_comparison.png"),
        dpi = 300,
        format = "png",
        bbox_inches = "tight"
      )
      plt.close()
      # merge
      runtime_comparison["time"] = runtime_comparison.index
      runtime_comparison[loop_over] = exp_description_tuple[0]
      runtime_comparison["seed"] = exp_description_tuple[1]
      all_runtime_values = pd.concat(
        [all_runtime_values, runtime_comparison], ignore_index = True
      )
  # cumulative plot
  if len(all_obj_values) > 0:
    # -- save
    all_obj_values.to_csv(
      os.path.join(plot_folder, "obj.csv"), index = False
    )
    all_rej_values.to_csv(
      os.path.join(plot_folder, "rejections.csv"), index = False
    )
    all_runtime_values.to_csv(
      os.path.join(plot_folder, "runtime.csv"), index = False
    )
    all_i_tc.to_csv(
      os.path.join(plot_folder, "i_termination_condition.csv"), index = False
    )
    # -- plot
    for exp_value, objs in all_obj_values.groupby(loop_over):
      rejs = all_rej_values[all_rej_values[loop_over] == exp_value]
      rtvs = all_runtime_values[
        all_runtime_values[loop_over] == exp_value
      ].copy(deep = True)
      i_tc = all_i_tc[all_i_tc[loop_over] == exp_value]
      fig, axs = plt.subplots(
        nrows = 2, ncols = 2, figsize = (12, 8), sharex = True,
        gridspec_kw = {"hspace": 0.02}
      )
      fig2, axs2 = plt.subplots(
        nrows = 1, ncols = 3, figsize = (18, 4), sharex = True,
        gridspec_kw = {"hspace": 0.02}
      )
      for seed, obj in objs.groupby("seed"):
        rej = rejs[rejs["seed"] == seed]
        rtv = rtvs[rtvs["seed"] == seed]
        # deviation
        obj.plot(
          x = "time", 
          y = "dev", 
          ax = axs[0,1], 
          color = mcolors.TABLEAU_COLORS["tab:green"], 
          linewidth = 1, 
          marker = ".", 
          grid = True,
          legend = False
        )
        rej.plot(
          x = "time", 
          y = "dev", 
          ax = axs[1,1], 
          color = mcolors.TABLEAU_COLORS["tab:green"], 
          linewidth = 1, 
          marker = ".", 
          grid = True,
          legend = False
        )
        rtv.plot(
          x = "time", 
          y = "dev", 
          ax = axs2[1], 
          color = mcolors.TABLEAU_COLORS["tab:green"], 
          linewidth = 1, 
          marker = ".", 
          grid = True,
          legend = False
        )
        if not rtv["best_iteration"].isnull().all():
          rtv.plot(
            x = "time", 
            y = "best_iteration", 
            ax = axs2[2], 
            color = mcolors.TABLEAU_COLORS["tab:pink"], 
            linewidth = 1, 
            marker = ".", 
            grid = True,
            legend = False
          )
        rtv.plot(
          x = "time", 
          y = "iteration", 
          ax = axs2[2], 
          color = mcolors.TABLEAU_COLORS["tab:pink"], 
          linewidth = 1, 
          linestyle = "dashed",
          marker = ".", 
          grid = True,
          legend = False
        )
        # centralized
        obj.plot(
          x = "time", 
          y = "LoadManagementModel", 
          ax = axs[0,0], 
          color = mcolors.TABLEAU_COLORS["tab:blue"], 
          linewidth = 1,
          grid = True,
          legend = False
        )
        rej.plot(
          x = "time", 
          y = "LoadManagementModel", 
          ax = axs[1,0], 
          color = mcolors.TABLEAU_COLORS["tab:blue"], 
          linewidth = 1,
          grid = True,
          legend = False
        )
        rtv.plot(
          x = "time", 
          y = "LoadManagementModel", 
          ax = axs2[0], 
          color = mcolors.TABLEAU_COLORS["tab:blue"], 
          linewidth = 1,
          grid = True,
          legend = False
        )
        # FaaS-MACrO
        obj.plot(
          x = "time", 
          y = "FaaS-MACrO", 
          ax = axs[0,0], 
          color = mcolors.TABLEAU_COLORS["tab:orange"], 
          linewidth = 1,
          grid = True,
          legend = False
        )
        rej.plot(
          x = "time", 
          y = "FaaS-MACrO", 
          ax = axs[1,0], 
          color = mcolors.TABLEAU_COLORS["tab:orange"], 
          linewidth = 1,
          grid = True,
          legend = False
        )
        rtv.plot(
          x = "time", 
          y = "FaaS-MACrO", 
          ax = axs2[0], 
          color = mcolors.TABLEAU_COLORS["tab:orange"], 
          linewidth = 1,
          grid = True,
          legend = False
        )
      fig3, axs3 = plt.subplots(
        nrows = 2, ncols = 1, figsize = (12, 6), sharex = True,
        gridspec_kw = {"hspace": 0.02}
      )
      rtvs["idx"] = rtvs.index
      rtvs.plot.scatter(
        x = "idx",
        y = "LoadManagementModel",
        ax = axs3[0],
        c = mcolors.TABLEAU_COLORS["tab:blue"],
        grid = True,
        label = "LoadManagementModel"
      )
      rtvs.plot.scatter(
        x = "idx",
        y = "FaaS-MACrO",
        ax = axs3[0],
        c = mcolors.TABLEAU_COLORS["tab:orange"],
        grid = True,
        label = "FaaS-MACrO"
      )
      if "dev" in rtvs:
        rtvs.plot.scatter(
          x = "idx",
          y = "dev",
          ax = axs3[1],
          c = mcolors.TABLEAU_COLORS["tab:green"],
          grid = True
        )
      # average
      avg = objs.groupby("time").mean(numeric_only = True)
      avg_rej = rejs.groupby("time").mean(numeric_only = True)
      avg_rtv = rtvs.groupby("time").mean(numeric_only = True)
      # -- deviation
      avg.plot(
        y = "dev",
        ax = axs[0,1],
        color = mcolors.TABLEAU_COLORS["tab:red"],
        linewidth = 2,
        marker = ".", 
        grid = True,
        label = "Average deviation ((FaaS-MACrO - LMM) / LMM) [%]"
      )
      avg_rej.plot(
        y = "dev",
        ax = axs[1,1],
        color = mcolors.TABLEAU_COLORS["tab:red"],
        linewidth = 2,
        marker = ".", 
        grid = True,
        label = "Average deviation (FaaS-MACrO - LMM) [%]"
      )
      if "dev" in avg_rtv:
        avg_rtv.plot(
          y = "dev",
          ax = axs2[1],
          color = mcolors.TABLEAU_COLORS["tab:red"],
          linewidth = 2,
          marker = ".", 
          grid = True,
          label = "Average deviation (FaaS-MACrO / LMM) [x]"
        )
      if "best_iteration" in avg_rtv:
        avg_rtv.plot(
          y = "best_iteration",
          ax = axs2[2],
          color = "black",
          linewidth = 2,
          marker = ".", 
          grid = True,
          label = "Best iteration"
        )
      avg_rtv.plot(
        y = "iteration",
        ax = axs2[2],
        color = "black",
        linewidth = 1,
        linestyle = "dashed",
        marker = ".", 
        grid = True,
        label = "# iterations"
      )
      # -- centralized
      avg.plot(
        y = "LoadManagementModel",
        ax = axs[0,0],
        color = mcolors.TABLEAU_COLORS["tab:blue"],
        linewidth = 2,
        grid = True,
        label = "Average LMM"
      )
      avg_rej.plot(
        y = "LoadManagementModel",
        ax = axs[1,0],
        color = mcolors.TABLEAU_COLORS["tab:blue"],
        linewidth = 2,
        grid = True,
        label = "Average LMM"
      )
      avg_rtv.plot(
        y = "LoadManagementModel",
        ax = axs2[0],
        color = mcolors.TABLEAU_COLORS["tab:blue"],
        linewidth = 2,
        grid = True,
        label = "Average LMM"
      )
      axs3[0].axhline(
        y = rtvs["LoadManagementModel"].mean(),
        color = mcolors.TABLEAU_COLORS["tab:blue"],
        linewidth = 2
      )
      # -- FaaS-MACrO
      avg.plot(
        y = "FaaS-MACrO",
        ax = axs[0,0],
        color = mcolors.TABLEAU_COLORS["tab:orange"],
        linewidth = 2,
        grid = True,
        label = "Average FaaS-MACrO"
      )
      avg_rej.plot(
        y = "FaaS-MACrO",
        ax = axs[1,0],
        color = mcolors.TABLEAU_COLORS["tab:orange"],
        linewidth = 2,
        grid = True,
        label = "Average FaaS-MACrO"
      )
      if "FaaS-MACrO" in avg_rtv:
        avg_rtv.plot(
          y = "FaaS-MACrO",
          ax = axs2[0],
          color = mcolors.TABLEAU_COLORS["tab:orange"],
          linewidth = 2,
          grid = True,
          label = "Average FaaS-MACrO"
        )
        axs3[0].axhline(
          y = rtvs["FaaS-MACrO"].mean(),
          color = mcolors.TABLEAU_COLORS["tab:orange"],
          linewidth = 2
        )
        axs3[1].axhline(
          y = rtvs["dev"].mean(),
          color = mcolors.TABLEAU_COLORS["tab:green"],
          linewidth = 2
        )
      axs[1,0].set_xlabel("Control time period $t$")
      axs[1,1].set_xlabel("Control time period $t$")
      axs[0,0].set_ylabel("Objective function value")
      axs[0,1].set_ylabel("Objective function deviation [%]")
      axs[1,0].set_ylabel("Total percentage of rejections [%]")
      axs[1,1].set_ylabel("Percentage rejections deviation [%]")
      axs2[0].set_xlabel("Control time period $t$")
      axs2[1].set_xlabel("Control time period $t$")
      axs2[2].set_xlabel("Control time period $t$")
      axs2[0].set_ylabel("Runtime [s]")
      axs2[1].set_ylabel("Runtime deviation [x]")
      axs2[2].set_ylabel("Number of iterations")
      axs3[-1].set_xlabel("Experiment")
      axs3[0].set_ylabel("Runtime [s]")
      axs3[1].set_ylabel("Runtime deviation [x]")
      fig.savefig(
        os.path.join(plot_folder, f"obj-{loop_over}_{exp_value}.png"),
        dpi = 300,
        format = "png",
        bbox_inches = "tight"
      )
      plt.close(fig)
      fig2.savefig(
        os.path.join(plot_folder, f"runtime-{loop_over}_{exp_value}.png"),
        dpi = 300,
        format = "png",
        bbox_inches = "tight"
      )
      plt.close(fig2)
      fig3.savefig(
        os.path.join(
          plot_folder, f"linear_runtime-{loop_over}_{exp_value}.png"
        ),
        dpi = 300,
        format = "png",
        bbox_inches = "tight"
      )
      plt.close(fig3)
      # termination condition
      i_tc["criterion"].value_counts().plot.bar(
        rot = 0,
        grid = True
      )
      plt.savefig(
        os.path.join(plot_folder, f"i_tc-{loop_over}_{exp_value}.png"),
        dpi = 300,
        format = "png",
        bbox_inches = "tight"
      )
      plt.close()
  # termination condition
  if "criterion" in all_i_tc:
    _, ax = plt.subplots(figsize = (20,6))
    all_i_tc["criterion"].value_counts(normalize = True).plot.bar(
      rot = 0,
      grid = True,
      ax = ax,
      fontsize = 21
    )
    ax.set_xlabel("Stopping criterion", fontsize = 21)
    ax.set_ylabel("Frequency", fontsize = 21)
    plt.savefig(
      os.path.join(plot_folder, "i_termination_condition.png"),
      dpi = 300,
      format = "png",
      bbox_inches = "tight"
    )
    plt.close()
  # save ping-pong problems info
  with open(os.path.join(plot_folder, "ping_pong_problems.txt"), "w") as ost:
    for el in ping_pong_list:
      ost.write(f"{el}\n")


def run(
    base_config: dict, 
    base_solution_folder: str, 
    n_experiments: int, 
    run_centralized_only: bool,
    run_faasmacro_only: bool,
    generate_only: bool,
    fix_r: bool,
    sp_parallelism: int,
    enable_plotting: bool,
    loop_over: str
  ):
  seed = base_config["seed"]
  log_on_file = True if base_config["verbose"] > 0 else False
  exp_values = base_config["limits"][loop_over]
  disable_plotting = not enable_plotting
  from_instances = base_config["limits"].get("path", None)
  # generate list of experiments
  experiments_list = generate_experiments_list(exp_values, seed, n_experiments)
  # load list of already-run experiments (if any)
  solution_folders = {
    "experiments_list": [], "centralized": [], "faas-macro": []
  }
  if os.path.exists(os.path.join(base_solution_folder, "experiments.json")):
    with open(
        os.path.join(base_solution_folder, "experiments.json"), "r"
      ) as ist:
      solution_folders = json.load(ist)
  # load list of previous instances (if required)
  old_instance_paths = {}
  if from_instances is not None:
    with open(os.path.join(from_instances, "experiments.json"), "r") as ist:
      old_instance_paths = json.load(ist)
  # loop over the experiments list
  for exp_value, seed in tqdm(experiments_list):
    # check if the experiment is still to run
    run_c = False
    run_i = False
    experiment_idx = None
    try:
      experiment_idx = solution_folders["experiments_list"].index(
        [exp_value, seed]
      )
      if not (run_faasmacro_only or generate_only) and ((
          len(solution_folders["centralized"]) <= experiment_idx
        ) or (
          solution_folders["centralized"][experiment_idx] is None
        )):
        run_c = True
      if not (run_centralized_only or generate_only) and ((
          len(solution_folders["faas-macro"]) <= experiment_idx
        ) or (
          solution_folders["faas-macro"][experiment_idx] is None
        )):
        run_i = True
    except ValueError:
      run_c = True if not (run_faasmacro_only or generate_only) else False
      run_i = True if not (run_centralized_only or generate_only) else False
    # if the experiment is still to run...
    if run_c or run_i or generate_only:
      # -- update configuration
      config = deepcopy(base_config)
      config["limits"][loop_over].pop("values", None)
      config["limits"][loop_over]["min"] = exp_value
      config["limits"][loop_over]["max"] = exp_value
      config["seed"] = seed
      # -- look for old instance path (if required)
      if "experiments_list" in old_instance_paths:
        try:
          old_exp_idx = old_instance_paths["experiments_list"].index(
            [exp_value, seed]
          )
          old_exp_path = None
          if "centralized" in old_instance_paths:
            old_exp_path = old_instance_paths["centralized"][
              old_exp_idx
            ]
          elif "faas-macro" in old_instance_paths:
            old_exp_path = old_instance_paths["faas-macro"][
              old_exp_idx
            ]
          config["limits"]["path"] = old_exp_path
          if config["limits"]["load"]["trace_type"] == "load_existing":
            config["limits"]["load"]["path"] = old_exp_path
        except Exception:
          pass
      # -- solve centralized model
      c_folder = None
      if run_c or generate_only:
        c_folder = run_centralized(
          config, 
          log_on_file = log_on_file, 
          disable_plotting = disable_plotting,
          generate_only = generate_only
        )
        solution_folders["centralized"].append(c_folder)
      else:
        if experiment_idx is not None:
          c_folder = solution_folders["centralized"][experiment_idx]
      # -- solve iterative model
      if fix_r:
        config["opt_solution_folder"] = c_folder
      if run_i:
        i_folder = run_iterations(
          config, 
          sp_parallelism,
          log_on_file = log_on_file, 
          disable_plotting = disable_plotting
        )
        solution_folders["faas-macro"].append(i_folder)
      # -- save info
      if experiment_idx is None:
        solution_folders["experiments_list"].append([exp_value, seed])
      # -- save
      with open(
        os.path.join(base_solution_folder, "experiments.json"), "w"
      ) as ost:
        ost.write(json.dumps(solution_folders, indent = 2))
  # immediate postprocessing
  results_postprocessing(solution_folders, base_solution_folder, loop_over)


if __name__ == "__main__":
  args = parse_arguments()
  config_file = args.config
  n_experiments = args.n_experiments
  run_centralized_only = args.run_centralized_only
  run_faasmacro_only = args.run_faasmacro_only
  postprocessing_only = args.postprocessing_only
  generate_only = args.generate_only
  postprocessing_list = args.postprocessing_list
  fix_r = args.fix_r
  sp_parallelism = args.sp_parallelism
  enable_plotting = args.enable_plotting
  loop_over = args.loop_over
  # load configuration file
  base_config = load_configuration(config_file)
  base_solution_folder = base_config["base_solution_folder"]
  if not postprocessing_only:
    # write configuration file in the base folder
    os.makedirs(base_solution_folder, exist_ok = True)
    with open(os.path.join(base_solution_folder, "config.json"), "w") as ost:
      ost.write(json.dumps(base_config, indent = 2))
    # run
    run(
      base_config, 
      base_solution_folder, 
      n_experiments, 
      run_centralized_only,
      run_faasmacro_only,
      generate_only,
      fix_r,
      sp_parallelism,
      enable_plotting,
      loop_over
    )
  else:
    if not postprocessing_list:
      solution_folders = {}
      with open(
        os.path.join(base_solution_folder, "experiments.json"), "r"
      ) as ist:
        solution_folders = json.load(ist)
      results_postprocessing(solution_folders, base_solution_folder, loop_over)
    else:
      for foldername in os.listdir(base_solution_folder):
        if not foldername.startswith("."):
          bsf = os.path.join(base_solution_folder, foldername)
          print(f"{'-'*80}\n{bsf}")
          solution_folders = {}
          with open(
            os.path.join(bsf, "experiments.json"), "r"
          ) as ist:
            solution_folders = json.load(ist)
          results_postprocessing(solution_folders, bsf, loop_over)
