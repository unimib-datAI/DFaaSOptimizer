from logs_postprocessing import parse_log_file

import matplotlib.pyplot as plt
import pandas as pd
import parse
import os


def main(experiment_folder: str, Nn: int):
  exp = os.path.basename(experiment_folder)
  logs_df, best_sol_df = parse_log_file(
    experiment_folder, 
    exp,
    pd.DataFrame(),
    {"social_welfare": pd.DataFrame(), "centralized": pd.DataFrame()},
    Nn
  )
  best_sol_df["social_welfare"].to_csv(
    os.path.join(experiment_folder, "best_sw_sol_df.csv"), index = False
  )
  best_sol_df["centralized"].to_csv(
    os.path.join(experiment_folder, "best_c_sol_df.csv"), index = False
  )
  logs_df.to_csv(os.path.join(experiment_folder, "logs_df.csv"), index = False)
  _, axs = plt.subplots(nrows = 2, ncols = 1, figsize=(20,12), sharex = True)
  last_it = 0
  for t, data in best_sol_df["social_welfare"].groupby("time"):
    data["x"] = data["best_solution_it"] + last_it
    data.plot(
      x = "x",
      y = "obj",
      ax = axs[0],
      grid = True,
      marker = ".",
      markersize = 10,
      linewidth = 2
    )
    axs[0].axvline(
      x = last_it,
      linestyle = "dotted",
      linewidth = 1,
      color = "k"
    )
    last_it += data["best_solution_it"].max()
  last_it = 0
  for t, data in best_sol_df["centralized"].groupby("time"):
    data["x"] = data["best_solution_it"] + last_it
    data.plot(
      x = "x",
      y = "obj",
      ax = axs[1],
      grid = True,
      marker = ".",
      markersize = 10,
      linewidth = 2
    )
    axs[1].axvline(
      x = last_it,
      linestyle = "dotted",
      linewidth = 1,
      color = "k"
    )
    last_it += data["best_solution_it"].max()
  plt.savefig(
    os.path.join(experiment_folder, "best_solution_obj.png"),
    dpi = 300,
    format = "png",
    bbox_inches = "tight"
  )
  plt.close()


if __name__ == "__main__":
  experiment_folder = "solutions/manual/2025-09-11_10-28-33.047502"
  Nn = 100
  main(experiment_folder, Nn)

