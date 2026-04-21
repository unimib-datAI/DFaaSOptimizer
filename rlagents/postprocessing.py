from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import Tuple
import pandas as pd
import numpy as np
import argparse
import json
import ast
import os


def parse_arguments() -> argparse.Namespace:
  """
  Parse input arguments
  """
  parser: argparse.ArgumentParser = argparse.ArgumentParser(
    description = "Results postprocessing", 
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument(
    "--base_exp_folder",
    help = "Base results folder",
    type = str,
    default = "results"
  )
  parser.add_argument(
    "--exp_name",
    help = "Experiment name",
    type = str,
    default = None
  )
  parser.add_argument(
    "--exp_json",
    help = "JSON file(s) where the experiments list is saved",
    type = str,
    nargs = "+",
    default = None
  )
  parser.add_argument(
    "--plot_iterations",
    type = int,
    nargs = "+",
    default = []
  )
  parser.add_argument(
    "--plot_reward_only",
    default = False,
    action = "store_true"
  )
  parser.add_argument(
    "--eval_only",
    default = False,
    action = "store_true"
  )
  parser.add_argument(
    "--reload_all",
    default = False,
    action = "store_true"
  )
  parser.add_argument(
    "--maw",
    type = int,
    help = "Moving average window",
    default = 10
  )
  # Parse the arguments
  args: argparse.Namespace = parser.parse_known_args()[0]
  return args


def load_json_file(
    exp_folder: str,
    last_iter: int,
    fname: str = "evaluations",
    reload_all: bool = False
  ) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
  """
  Load results from the evaluations.json file
  """
  all_hist_stats = pd.DataFrame()
  all_episode_hist_stats = pd.DataFrame()
  all_policy_hist_stats = pd.DataFrame()
  files_exist = False
  if not reload_all:
    summary_folder = os.path.join(exp_folder, "summary", fname)
    files_exist, sfx = summary_files_exist(summary_folder)
    compression = "gzip" if sfx != "" else None
    if files_exist:
      print("Loading existing files...")
      all_hist_stats = pd.read_csv(
        os.path.join(summary_folder, f"all_hist_stats.csv{sfx}"), 
        compression = compression
      )
      print("  done (all_hist_stats)")
      all_episode_hist_stats = pd.read_csv(
        os.path.join(summary_folder, f"all_episode_hist_stats.csv{sfx}"), 
        compression = compression
      )
      print("  done (all_episode_hist_stats)")
      all_policy_hist_stats = pd.read_csv(
        os.path.join(summary_folder, f"all_policy_hist_stats.csv{sfx}"), 
        compression = compression
      )
      print("  done (all_policy_hist_stats)")
  # load evaluations file
  if not files_exist:
    results = []
    with open(os.path.join(exp_folder, f"{fname}.json"), "r") as istream:
      if fname.startswith("eval"):
        results = json.load(istream)["evaluations"]
    pfx = "" if fname == "result" else "after_"
    for result_line in results:
      it = result_line[f"{pfx}training_iteration"]
      print(f"Iter {it}")
      # process results only if they are not already available
      if it > last_iter:
        # hist stats
        hist_stats = {
          k: v for k, v in result_line["hist_stats"].items() if not (
            "episode_" in k or "policy_" in k or "worker_index" in k
          )
        }
        hist_stats = pd.DataFrame(hist_stats)
        episode_hist_stats = pd.DataFrame({
          k: v for k, v in result_line["hist_stats"].items() if "episode_" in k
        })
        policy_hist_stats = pd.DataFrame({
          k: v for k, v in result_line["hist_stats"].items() if "policy_" in k
        })
        # add information about the episode number
        hist_stats["episode"] = [-1] * len(hist_stats)
        # for _, temp in hist_stats.groupby("current_time"):
        #   hist_stats.loc[temp.index, "episode"] = range(len(temp))
        # concatenate
        hist_stats["iter"] = [it] * len(hist_stats)
        hist_stats["step"] = range(len(hist_stats))
        episode_hist_stats["iter"] = [it] * len(episode_hist_stats)
        policy_hist_stats["iter"] = [it] * len(policy_hist_stats)
        all_hist_stats = pd.concat(
          [all_hist_stats, hist_stats], ignore_index = True
        )
        all_episode_hist_stats = pd.concat(
          [all_episode_hist_stats, episode_hist_stats], ignore_index = True
        )
        all_policy_hist_stats = pd.concat(
          [all_policy_hist_stats, policy_hist_stats], ignore_index = True
        )
  return (
    all_hist_stats, 
    all_episode_hist_stats, 
    all_policy_hist_stats, 
    compression,
    not files_exist
  )



def load_progress_file(
    exp_folder: str, 
    last_iter: int, 
    fname: str = "progress", 
    reload_all: bool = False
  ) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
  """
  Load results from the progress.csv file
  """
  all_hist_stats = pd.DataFrame()
  all_episode_hist_stats = pd.DataFrame()
  all_policy_hist_stats = pd.DataFrame()
  files_exist = False
  compression = None
  if not reload_all:
    summary_folder = os.path.join(exp_folder, "summary", fname)
    files_exist, sfx = summary_files_exist(summary_folder)
    compression = "gzip" if sfx != "" else None
    if files_exist:
      print("Loading existing files...")
      all_hist_stats = pd.read_csv(
        os.path.join(summary_folder, f"all_hist_stats.csv{sfx}"), 
        compression = compression
      )
      print("  done (all_hist_stats)")
      all_episode_hist_stats = pd.read_csv(
        os.path.join(summary_folder, f"all_episode_hist_stats.csv{sfx}"), 
        compression = compression
      )
      print("  done (all_episode_hist_stats)")
      all_policy_hist_stats = pd.read_csv(
        os.path.join(summary_folder, f"all_policy_hist_stats.csv{sfx}"), 
        compression = compression
      )
      print("  done (all_policy_hist_stats)")
  # load progress file
  if not files_exist:
    progress = pd.DataFrame()
    if os.path.exists(os.path.join(exp_folder, f"{fname}.csv")):
      progress = pd.read_csv(os.path.join(exp_folder, f"{fname}.csv"))
    elif os.path.exists(os.path.join(exp_folder, f"{fname}.csv.gz")):
      progress = pd.read_csv(
        os.path.join(exp_folder, f"{fname}.csv.gz"), compression = "gzip"
      )
      compression = "gzip"
    elif os.path.exists(os.path.join(exp_folder, f"{fname}.json")):
      return load_json_file(
        exp_folder, last_iter, fname, reload_all
      )
    # build dataframes
    pfx = "" if fname == "progress" else "after_"
    for it in progress[f"{pfx}training_iteration"]:
      print(f"Iter {it}")
      # process results only if they are not already available
      if it > last_iter:
        row = progress[progress[f"{pfx}training_iteration"] == it]
        # hist stats
        hist_stats_cols = [
          c for c in row.columns if c.startswith("env_runners/hist_stats/")
        ]
        hist_stats_dict = {}
        for c in hist_stats_cols:
          hist_stats_dict[c.split("/")[-1]] = ast.literal_eval(
            row[c].iloc[0]
          )
        hist_stats = {
          k: v for k, v in hist_stats_dict.items() if not (
            "episode_" in k or "policy_" in k or "worker_index" in k
          )
        }
        hist_stats = pd.DataFrame(hist_stats)
        episode_hist_stats = pd.DataFrame({
          k: v for k, v in hist_stats_dict.items() if "episode_" in k
        })
        policy_hist_stats = pd.DataFrame({
          k: v for k, v in hist_stats_dict.items() if "policy_" in k
        })
        # add information about the episode number
        hist_stats["episode"] = [-1] * len(hist_stats)
        # for _, temp in hist_stats.groupby("current_time"):
        #   hist_stats.loc[temp.index, "episode"] = range(len(temp))
        # concatenate
        hist_stats["iter"] = [it] * len(hist_stats)
        hist_stats["step"] = range(len(hist_stats))
        episode_hist_stats["iter"] = [it] * len(episode_hist_stats)
        policy_hist_stats["iter"] = [it] * len(policy_hist_stats)
        all_hist_stats = pd.concat(
          [all_hist_stats, hist_stats], ignore_index = True
        )
        all_episode_hist_stats = pd.concat(
          [all_episode_hist_stats, episode_hist_stats], ignore_index = True
        )
        all_policy_hist_stats = pd.concat(
          [all_policy_hist_stats, policy_hist_stats], ignore_index = True
        )
  return (
    all_hist_stats, 
    all_episode_hist_stats, 
    all_policy_hist_stats, 
    compression,
    not files_exist
  )


def plot_action(
    df: pd.DataFrame, 
    agents: list, 
    plot_folder: str = None, 
    suffix: str = "",
    use_previous: bool = False
  ):
  to_plot = df
  to_plot.index = range(len(to_plot))
  pfx = "previous_" if use_previous else ""
  _, axs = plt.subplots(
    nrows = len(agents), ncols = 1, figsize = (30,4 * len(agents))
  )
  for idx, agent in enumerate(agents):
    sfx = f"-{agent}" if agent != "default_policy" else ""
    # -- input load
    to_plot[f"{pfx}input_rate{sfx}"].plot(
      linewidth = 2,
      marker = ".",
      color = "k",
      ax = axs[idx],
      label = None,
      legend = False
    )
    # -- action
    to_plot[[
      f"loc{sfx}", 
      f"total_fwd{sfx}", 
      f"rej{sfx}"
    ]].plot.bar(
      stacked = True,
      ax = axs[idx],
      label = None,
      legend = False
    )
    axs[idx].set_ylabel(agent)
    axs[idx].grid(visible = True, axis = "y")
  if plot_folder is not None:
    plt.savefig(
      os.path.join(plot_folder, f"actions{suffix}.png"),
      dpi = 300,
      format = "png",
      bbox_inches = "tight"
    )
    plt.close()
  else:
    plt.title(title_key)
    plt.show()


def plot_forward(
    df: pd.DataFrame, agents: list, plot_folder: str = None, suffix: str = ""
  ):
  to_plot = df
  to_plot.index = range(len(to_plot))
  _, axs = plt.subplots(
    nrows = len(agents), ncols = 1, figsize = (30,4 * len(agents))
  )
  for idx, agent in enumerate(agents):
    # -- total number of forwarded requests
    to_plot[f"total_fwd-{agent}"].plot(
      linewidth = 2,
      marker = ".",
      color = "k",
      ax = axs[idx],
      label = None,
      legend = False
    )
    # -- details
    to_plot[[
      f"fwd_to_{neighbor}-{agent}" 
        for neighbor in agents if neighbor != agent
    ]].plot.bar(
      stacked = True,
      ax = axs[idx],
      # label = None,
      # legend = False
    )
    axs[idx].set_ylabel(agent)
    axs[idx].grid(visible = True, axis = "y")
    axs[idx].legend(loc = "center left", bbox_to_anchor = (1, 0.5))
  if plot_folder is not None:
    plt.savefig(
      os.path.join(plot_folder, f"offloading{suffix}.png"),
      dpi = 300,
      format = "png",
      bbox_inches = "tight"
    )
    plt.close()
  else:
    plt.title(title_key)
    plt.show()


def plot_moving_average(
    data: pd.DataFrame, 
    columns: list, 
    window: int, 
    plot_folder: str = None,
    title_key: str = "training",
    y_threshold: float = None,
    alpha: float = 1.0
  ):
  """
  Plot the moving average over the given window of the results in the listed 
  columns
  """
  columns = sorted(columns)
  # prepare style
  all_linestyles = ["solid", "dashed", "dotted"]
  all_colors = list(
    mcolors.TABLEAU_COLORS.values()
  ) + list(
    mcolors.BASE_COLORS.values()
  ) + list(
    mcolors.CSS4_COLORS.values()
  )
  colors = []
  linestyles = []
  prefixes = []
  for column in columns:
    # -- linestyle
    if "-" in column and column.split("-")[0] not in prefixes:
      new_linestyle = all_linestyles[len(prefixes)]
      prefixes.append(column.split("-")[0])
    elif "-" not in column:
      new_linestyle = "solid"
    linestyles.append(new_linestyle)
    # -- color
    idx = 0
    if "-" in column or column.endswith("_reward"):
      if "-" in column:
        agent = column.split("-")[-1]
        if "default" not in agent:
          idx = int(agent)
      elif column.startswith("policy_") and column.endswith("_reward"):
        agent = column.replace("policy_", "")
        agent = agent.replace("_reward", "")
        if agent != "default":
          idx = int(agent)
    else:
      idx += len(colors)
    colors.append(all_colors[idx])
  # compute average
  avg = data[columns].rolling(
    window = window,
    min_periods = 1
  ).mean()
  min_iter = data["iter"].min()
  max_iter = data["iter"].max()
  # plot
  _, ax = plt.subplots()
  for column, color, linestyle in zip(columns, colors, linestyles):
    avg[column].plot(
      color = color, linestyle = linestyle, ax = ax, alpha = alpha
    )
  if y_threshold is not None:
    ax.axhline(y_threshold, color = "k", linestyle = "dashed", linewidth = 2)
  plt.grid()
  plt.legend(loc = "center left", bbox_to_anchor = (1, 0.5))
  if plot_folder is not None:
    plt.savefig(
      os.path.join(
        plot_folder, 
        f"{title_key}_moving_average_{min_iter}_{max_iter}.png"
      ),
      dpi = 300,
      format = "png",
      bbox_inches = "tight"
    )
    plt.close()
  else:
    plt.title(title_key)
    plt.show()


def aggregate_over_neighbors(
    df: pd.DataFrame, agent: str, key: str, how: str = "avg"
  ) -> np.array:
  colnames_in = ["avg_resp_time_fwd_to"] if key == "resp_time" else [
    "fwd_to", "_rejected"
  ]
  colnames_out = ["_ndb", "_ndf", "_rt"] if key == "resp_time" else []
  cols = [
    c for c in df.columns if all(
      [k in c for k in colnames_in]
    ) and not any(
      [k in c for k in colnames_out]
    ) and c.endswith(
      f"-{agent}"
    )
  ]
  res = pd.Series()
  if how == "avg":
    res = df[cols].mean(axis = 1).values
  elif how == "sum":
    res = df[cols].sum(axis = 1).values
  else:
    raise ValueError(f"Unknown aggregation: {how}")
  return res


def summary_files_exist(summary_folder: str) -> bool:
  files_exist = True
  sfx = ""
  if not os.path.exists(os.path.join(summary_folder, "all_hist_stats.csv")):
    if os.path.exists(os.path.join(summary_folder, "all_hist_stats.csv.gz")):
      sfx = ".gz"
    else:
      files_exist = False
  if files_exist:
    if not os.path.exists(
        os.path.join(summary_folder, f"all_episode_hist_stats.csv{sfx}")
      ):
      files_exist = False
    if files_exist:
      if not os.path.exists(
          os.path.join(summary_folder, f"all_policy_hist_stats.csv{sfx}")
        ):
        files_exist = False
  return files_exist, sfx


def unpack_step_values(
    all_hist_stats: pd.DataFrame
  ) -> Tuple[pd.DataFrame, set]:
  to_convert = all_hist_stats.select_dtypes("object").copy(deep = True)
  new_df = pd.DataFrame()
  all_agents = set()
  for col in to_convert:
    print(col)
    df_col = pd.DataFrame()
    if isinstance(to_convert[col].iloc[0], dict):
      agents = list(to_convert[col].iloc[0].keys())
      for iteration in to_convert[col].index:
        df_dict = {}
        n_steps = 0
        for agent in agents:
          df_dict[f"{col}-{agent}"] = to_convert.loc[iteration,col][agent]
          n_steps = len(to_convert.loc[iteration,col][agent])
          all_agents.add(agent)
        df_dict["step"] = range(n_steps)
        df_dict["iter"] = iteration
        # merge
        df_col = pd.concat([df_col, pd.DataFrame(df_dict)], ignore_index=True)
    # join
    if len(new_df) == 0:
      new_df = df_col
    else:
      new_df = df_col.join(
        new_df.set_index(["iter", "step"], drop = True),
        on = ["iter", "step"],
        how = "inner"
      )
  return new_df, all_agents


def build_column_names(basename: str, agents: list) -> list:
  colnames = []
  if len(agents) > 1 or agents[0] != "default_policy":
    colnames = [f"{basename}-{a}" for a in agents]
  else:
    colnames = [basename]
  return colnames


def expand_agents_data(
    all_hist_stats: pd.DataFrame
  ) -> Tuple[pd.DataFrame, list]:
  agents = set()
  expanded_hist_stats = {}
  for col in all_hist_stats.columns:
    firstrow = all_hist_stats[col].iloc[0]
    if isinstance(firstrow, list):
      Nn = len(firstrow)
      Nf = len(firstrow[0]) if col != "fwd" else len(firstrow[0][0])
      for n in range(1, Nn+1):
        for f in range(1, Nf+1):
          agents.add(f"{n}_{f}")
          for idx in range(len(all_hist_stats[col])):
            if col != "fwd":
              if f"{col}-{n}_{f}" not in expanded_hist_stats:
                expanded_hist_stats[f"{col}-{n}_{f}"] = []
              if all_hist_stats[col].iloc[idx] is not None:
                expanded_hist_stats[f"{col}-{n}_{f}"].append(
                  all_hist_stats[col].iloc[idx][n-1][f-1]
                )
              else:
                expanded_hist_stats[f"{col}-{n}_{f}"].append(None)
            else:
              for j in range(1, Nn+1):
                if f"{col}_to_{j}_{f}-{n}_{f}" not in expanded_hist_stats:
                  expanded_hist_stats[f"{col}_to_{j}_{f}-{n}_{f}"] = []
                if all_hist_stats[col].iloc[idx] is not None:
                  expanded_hist_stats[f"{col}_to_{j}_{f}-{n}_{f}"].append(
                    all_hist_stats[col].iloc[idx][n-1][j-1][f-1]
                  )
                else:
                  expanded_hist_stats[f"{col}_to_{j}_{f}-{n}_{f}"].append(None)
    elif isinstance(firstrow, dict):
      nodes = list(firstrow.keys())
      functions = [
        int(k.replace("xyz_", "")) for k in firstrow[nodes[0]] if k != "r"
      ]
      for n in nodes:
        for f in functions:
          agents.add(f"{n}_{f}")
          for idx in range(len(all_hist_stats[col])):
            if f"action_n_replicas-{n}_{f}" not in expanded_hist_stats:
              expanded_hist_stats[f"action_n_replicas-{n}_{f}"] = []
            if f"action-{n}_{f}" not in expanded_hist_stats:
              expanded_hist_stats[f"action-{n}_{f}"] = []
            if all_hist_stats[col].iloc[idx] is not None:
              expanded_hist_stats[f"action_n_replicas-{n}_{f}"].append(
                all_hist_stats[col].iloc[idx][n]["r"]
              )
              expanded_hist_stats[f"action-{n}_{f}"].append(
                all_hist_stats[col].iloc[idx][n][f"xyz_{f}"]
              )
            else:
              expanded_hist_stats[f"action_n_replicas-{n}_{f}"].append(None)
              expanded_hist_stats[f"action-{n}_{f}"].append(None)
    else:
      expanded_hist_stats[col] = all_hist_stats[col].values
  expanded_hist_stats = pd.DataFrame(expanded_hist_stats)
  expanded_hist_stats["iter"] = all_hist_stats["iter"]
  return expanded_hist_stats, agents


def single_exp_postprocessing(
    exp_folder: str, 
    reload_all: bool = False,
    moving_average_window: int = 10, 
    last_iter: int = 0,
    plot_reward_only: bool = False,
    plot_iterations: list = [],
    eval_only: bool = False
  ) -> dict:
  scenarios = ["evaluations"] if eval_only else ["progress", "evaluations"]
  results = {}
  for scenario in scenarios:
    print(80 * "-")
    # create folder to store plots
    plot_folder = os.path.join(exp_folder, "plots", scenario)
    os.makedirs(plot_folder, exist_ok = True)
    # load data
    (
      all_hist_stats, 
      all_episode_hist_stats, 
      all_policy_hist_stats, 
      compression,
      new_results_loaded
    ) = load_progress_file(
      exp_folder, last_iter, scenario, reload_all
    )
    # identify agents
    agents = [
      "_".join(
        c.split("_")[1:-1]
      ) for c in all_policy_hist_stats.columns if c != "iter"
    ]
    # plot episode reward moving average
    plot_moving_average(
      all_episode_hist_stats, 
      ["episode_reward"], 
      moving_average_window, 
      plot_folder, 
      "episode_total_reward"
    )
    # -- by node
    plot_moving_average(
      all_policy_hist_stats, 
      [c for c in all_policy_hist_stats.columns if c != "iter"], 
      moving_average_window, 
      plot_folder, 
      "by_node_reward"
    )
    # compute by-episode average
    summary_folder = os.path.join(exp_folder, "summary", scenario)
    sfx = ".gz" if compression is not None else ""
    avg_stats_unpacked = pd.DataFrame()
    expanded_agents = deepcopy(agents)
    new_results_loaded = True
    if new_results_loaded:
      if len(agents) == 1 and agents[0] == "default_policy":
        all_hist_stats, expanded_agents = expand_agents_data(all_hist_stats)
      avg_stats_unpacked = all_hist_stats.groupby(
        "iter"
      ).mean(numeric_only = True).reset_index()
      # save summary results
      os.makedirs(summary_folder, exist_ok = True)
      all_hist_stats.to_csv(
        os.path.join(summary_folder, f"all_hist_stats.csv{sfx}"), 
        compression = compression,
        index = False
      )
      all_episode_hist_stats.to_csv(
        os.path.join(summary_folder, f"all_episode_hist_stats.csv{sfx}"), 
        compression = compression,
        index = False
      )
      all_policy_hist_stats.to_csv(
        os.path.join(summary_folder, f"all_policy_hist_stats.csv{sfx}"), 
        compression = compression,
        index = False
      )
      avg_stats_unpacked.to_csv(
        os.path.join(summary_folder, f"avg_stats_unpacked.csv{sfx}"), 
        compression = compression,
        index = False
      )
    else:
      avg_stats_unpacked = pd.read_csv(
        os.path.join(summary_folder, f"avg_stats_unpacked.csv{sfx}"), 
        compression = compression
      )
      print("  done (avg_stats_unpacked)")
    # plot detailed results
    if not plot_reward_only:
      # -- utility
      plot_moving_average(
        avg_stats_unpacked, 
        build_column_names(
          "loc_utility", agents
        ) + build_column_names(
          "fwd_utility", agents
        ) + build_column_names(
          "cloud_penalty", agents
        ), 
        moving_average_window, 
        plot_folder, 
        "utilities"
      )
      # -- centralized objective
      try:
        plot_moving_average(
          avg_stats_unpacked, 
          build_column_names("previous_cobj", agents), 
          moving_average_window, 
          plot_folder, 
          "cobj"
        )
      except KeyError:
        plot_moving_average(
          avg_stats_unpacked, 
          build_column_names("cobj", agents), 
          moving_average_window, 
          plot_folder, 
          "cobj"
        )
      # -- number of replicas
      plot_moving_average(
        avg_stats_unpacked, 
        build_column_names("n_replicas", expanded_agents),
        moving_average_window, 
        plot_folder, 
        "n_replicas_avg"
      )
      # -- "average" actions
      plot_action(avg_stats_unpacked, expanded_agents, plot_folder)
      # -- "average" detailed forwarding choices
      plot_forward(avg_stats_unpacked, expanded_agents, plot_folder)
      # -- detailed info in specific iterations
      for iteration in plot_iterations:
        if iteration in all_hist_stats["iter"].values:
          # ---- actions and detailed forwarding
          plot_action(
            all_hist_stats[all_hist_stats["iter"] == iteration], 
            expanded_agents, 
            plot_folder,
            f"-iter_{iteration}",
            use_previous = True
          )
          plot_forward(
            all_hist_stats[all_hist_stats["iter"] == iteration], 
            expanded_agents, 
            plot_folder,
            f"-iter_{iteration}"
          )
          # -- number of replicas
          plot_moving_average(
            all_hist_stats, 
            build_column_names("n_replicas", expanded_agents), 
            moving_average_window, 
            plot_folder, 
            f"n_replicas_detailed-iter_{iteration}",
            alpha = 0.7
          )
          # -- centralized objective
          try:
            plot_moving_average(
              all_hist_stats, 
              build_column_names("previous_cobj", agents),
              moving_average_window, 
              plot_folder, 
              f"cobj_detailed-iter_{iteration}",
              alpha = 0.7
            )
          except KeyError:
            plot_moving_average(
              all_hist_stats, 
              build_column_names("cobj", agents),
              moving_average_window, 
              plot_folder, 
              f"cobj_detailed-iter_{iteration}",
              alpha = 0.7
            )
    results[scenario] = {
      "all_hist_stats": all_hist_stats,
      "all_episode_hist_stats": all_episode_hist_stats,
      "all_policy_hist_stats": all_policy_hist_stats,
      "avg_stats_unpacked": avg_stats_unpacked
    }
  return results


def multiple_exp_postprocessing(
    base_res_folder: str, 
    exp_jsons: list, 
    moving_average_window: int = 10, 
    last_iter: int = 0,
    plot_reward_only: bool = False,
    plot_iterations: list = [],
    eval_only: bool = False
  ):
  # loop over experiments
  all_results = {}
  for exp_json in exp_jsons:
    # get method name
    method = os.path.basename(exp_json).split(".")[0].replace(
      "experiments_", ""
    )
    print(f'{30 * "#"} {method} {30 * "#"}')
    # load experiments dictionary
    exp_dict = {}
    with open(exp_json, "r") as istream:
      exp_dict = json.load(istream)
    # get all experiment folders
    base_exp_folder = os.path.join(base_res_folder, f"results_{method}")
    exp_folders = os.listdir(base_exp_folder)
    for n, k, network_idx, exp_seed, exp_suffix, elapsed_time in zip(
        exp_dict["n"], 
        exp_dict["k"], 
        exp_dict["network_idx"],
        exp_dict["exp_seed"],
        exp_dict["exp_suffix"],
        exp_dict["elapsed_time"]
      ):
      # -- find exp folder
      exp_folder = None
      idx = 0
      while idx < len(exp_folders) and exp_folder is None:
        if exp_folders[idx].endswith(exp_suffix):
          exp_folder = exp_folders[idx]
        idx += 1
      exp_folder = os.path.join(base_exp_folder, exp_folder)
      # -- post-process single experiment
      exp_results = {"progress": {}, "evaluations": {}}
      if os.path.exists(os.path.join(exp_folder, "summary")):
        for scenario in ["progress", "evaluations"]:
          print(f"{exp_folder} -- {scenario}")
          for sname in os.listdir(os.path.join(exp_folder, "summary", scenario)):
            if sname.endswith(".csv") or sname.endswith(".csv.gz"):
              compression = "gzip" if sname.endswith(".csv.gz") else None
              exp_results[scenario][sname.split(".")[0]] = pd.read_csv(
                os.path.join(exp_folder, "summary", scenario, sname),
                compression = compression
              )
      else:
        exp_results = single_exp_postprocessing(
          exp_folder,
          reload_all = True,
          moving_average_window = moving_average_window,
          last_iter = last_iter,
          plot_reward_only = plot_reward_only,
          plot_iterations = plot_iterations,
          eval_only = eval_only
        )
      # -- concat
      for scenario in ["progress", "evaluations"]:
        if scenario not in all_results:
          all_results[scenario] = {}
        for key, val in exp_results.get(scenario, {}).items():
          if key not in all_results[scenario]:
            all_results[scenario][key] = pd.DataFrame()
          # ----- add exp info
          val["n"] = n
          val["k"] = k
          val["network_idx"] = network_idx
          val["exp_seed"] = exp_seed
          val["exp_suffix"] = exp_suffix
          val["elapsed_time"] = elapsed_time
          val["method"] = method
          all_results[scenario][key] = pd.concat(
            [all_results[scenario][key], val], ignore_index = True
          )
  # summary
  colors = list(mcolors.TABLEAU_COLORS.values())
  for scenario, res_dict in all_results.items():
    episode_result = res_dict["all_episode_hist_stats"]
    # create folder to store plots
    plot_folder = os.path.join(base_res_folder, "plots", scenario)
    os.makedirs(plot_folder, exist_ok = True)
    # plot
    nrows = len(episode_result["n"].unique())
    ncols = max(len(episode_result["k"].unique()) // 2, 1)
    # -- reward
    _, axs = plt.subplots(
      nrows = nrows,
      ncols = ncols,
      figsize = (8 * ncols, 6 * nrows)
    )
    axs = np.atleast_2d(axs).T
    ridx = 0
    for n, n_data in episode_result.groupby("n"):
      cidx = 0
      for k, k_data in n_data.groupby("k"):
        for coloridx, (method, data) in enumerate(k_data.groupby("method")):
          min_val = data.groupby("iter").min(numeric_only = True)
          max_val = data.groupby("iter").max(numeric_only = True)
          avg_val = data.groupby("iter").mean(numeric_only = True)
          avg_val["episode_reward"].plot(
            ax = axs[ridx,cidx],
            grid = True,
            color = colors[coloridx],
            label = f"n = {n}; k = {k} ({method})"
          )
          axs[ridx,cidx].fill_between(
            x = avg_val.index,
            y1 = min_val["episode_reward"],
            y2 = max_val["episode_reward"],
            color = colors[coloridx],
            alpha = 0.4
          )
        axs[ridx,cidx].set_ylabel("Episode reward", fontsize = 14)
        axs[ridx,cidx].set_xlabel("Iteration", fontsize = 14)
        axs[ridx,cidx].legend(fontsize = 14)
        cidx += 1
      ridx += 1
    plt.savefig(
      os.path.join(plot_folder, "episode_reward.png"),
      dpi = 300,
      format = "png",
      bbox_inches = "tight"
    )
    plt.close()
    # -- elapsed time
    if scenario == "progress":
      episode_result["time_per_iter"] = episode_result[
        "elapsed_time"
      ] / episode_result.groupby("method")["iter"].transform("max")
      min_val = pd.DataFrame(
        episode_result.groupby(["n", "k", "method"]).min(numeric_only = True)[
          "time_per_iter"
        ]
      )
      max_val = pd.DataFrame(
        episode_result.groupby(["n", "k", "method"]).max(numeric_only = True)[
          "time_per_iter"
        ]
      )
      avg_val = pd.DataFrame(
        episode_result.groupby(["n", "k", "method"]).mean(numeric_only = True)[
          "time_per_iter"
        ]
      )
      to_plot = min_val.join(max_val, lsuffix = "_min", rsuffix = "_max").join(
        avg_val
      ).rename(
        columns = {
          "time_per_iter_min": "min",
          "time_per_iter_max": "max",
          "time_per_iter": "avg"
        }
      ).reset_index()
      nrows = len(to_plot["n"].unique())
      ncols = len(to_plot["k"].unique())
      wdt = len(to_plot["method"].unique())
      _, axs = plt.subplots(
        nrows = nrows,
        ncols = ncols,
        figsize = (3*wdt*ncols,6*nrows)
      )
      axs = np.atleast_2d(axs).T
      ridx = 0
      for n,n_to_plot in to_plot.groupby("n"):
        cidx = 0
        for k,k_to_plot in n_to_plot.groupby("k"):
          k_to_plot[["method","min","max","avg"]].plot.bar(
            x = "method",
            grid = True, 
            rot = 0, 
            fontsize = 14, 
            ax = axs[ridx,cidx]
          )
          axs[ridx,cidx].set_ylabel("Time per iteration [s]", fontsize = 14)
          # axs[ridx,cidx].set_xlabel("(# nodes, degree)", fontsize = 14)
          cidx += 1
        ridx += 1
      plt.savefig(
        os.path.join(plot_folder, "time_per_iter.png"),
        dpi = 300,
        format = "png",
        bbox_inches = "tight"
      )
      plt.close()


if __name__ == "__main__":
  # parse arguments
  args = parse_arguments()
  base_exp_folder = args.base_exp_folder
  plot_iterations = args.plot_iterations
  plot_reward_only = args.plot_reward_only
  eval_only = args.eval_only
  exp_name = args.exp_name
  exp_json = args.exp_json
  reload_all = args.reload_all
  moving_average_window = args.maw
  # post-process single experiment (if the exp name is provided)
  if exp_name is not None:
    exp_folder = os.path.join(base_exp_folder, exp_name)
    single_exp_postprocessing(
      exp_folder, 
      reload_all = reload_all,
      moving_average_window = moving_average_window,
      plot_reward_only = plot_reward_only, 
      plot_iterations = plot_iterations,
      eval_only = eval_only
    )
  # post-process multiple experiments (if the exp json is provided)
  elif exp_json is not None:
    multiple_exp_postprocessing(
      base_exp_folder, 
      exp_json,
      moving_average_window = moving_average_window,
      plot_reward_only = plot_reward_only, 
      plot_iterations = plot_iterations,
      eval_only = eval_only
    )
  else:
    raise ValueError("One between exp_name and exp_json must be provided")
