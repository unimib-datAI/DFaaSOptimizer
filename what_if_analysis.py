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
    pd.DataFrame(),
    Nn
  )
  best_sol_df.to_csv(os.path.join(experiment_folder, "best_sol_df.csv"), index = False)
  logs_df.to_csv(os.path.join(experiment_folder, "logs_df.csv"), index = False)
  _, ax = plt.subplots(figsize=(20,6))
  last_it = 0
  for t, data in best_sol_df.groupby("time"):
    data["x"] = data["best_solution_it"] + last_it
    data.plot(
      x = "x",
      y = "obj",
      ax = ax
    )
    ax.axvline(
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
  experiment_folder = "solutions/prova4/2025-09-09_13-27-10.236987"
  Nn = 100

  main(experiment_folder, Nn)
