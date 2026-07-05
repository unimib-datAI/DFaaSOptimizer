from __future__ import annotations

"""M7 parameter-sweep driver: run plasma.runner.run over a grid of
solver_options['plasma'] combinations and rank by objective."""

import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd

from plasma.runner import run as plasma_run


def _last10_mean(series: np.ndarray) -> float:
  return float(series[-10:].mean())


def sweep(config: dict, grid: dict, lmm_obj: "np.ndarray | None" = None) -> pd.DataFrame:
  """grid: {option_name: [values]} over solver_options['plasma'] keys.
  Runs plasma.runner.run for every combination (itertools.product),
  returns a DataFrame with one row per combo: the option values,
  mean objective, last-10-mean objective, and (if lmm_obj given)
  full/last-10 gap %. Deterministic: same config seed for every combo."""
  keys = list(grid.keys())
  rows = []
  for values in itertools.product(*(grid[k] for k in keys)):
    combo = dict(zip(keys, values))
    cfg = json.loads(json.dumps(config))
    cfg["solver_options"].setdefault("plasma", {}).update(combo)
    folder = plasma_run(cfg, parallelism=0, log_on_file=True,
                        disable_plotting=True)
    obj = pd.read_csv(os.path.join(folder, "obj.csv"))["Plasma"].to_numpy()
    row = dict(combo)
    row["obj_mean"] = float(obj.mean())
    row["obj_last10"] = _last10_mean(obj)
    if lmm_obj is not None:
      lmm = np.asarray(lmm_obj)
      row["gap_full"] = float(
        ((obj - lmm[: len(obj)]) / lmm[: len(obj)]).mean() * 100
      )
      n = min(10, len(obj), len(lmm))
      row["gap_last10"] = float(
        ((obj[-n:] - lmm[-n:]) / lmm[-n:]).mean() * 100
      )
    rows.append(row)
  return pd.DataFrame(rows)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("-c", "--config", required=True, help="config json path")
  parser.add_argument("--grid", required=True, help="JSON grid dict")
  parser.add_argument("--lmm", default=None, help="obj.csv for LMM/oracle gap")
  args = parser.parse_args()
  with open(args.config) as f:
    config = json.load(f)
  grid = json.loads(args.grid)
  lmm_obj = None
  if args.lmm:
    lmm_obj = pd.read_csv(args.lmm).iloc[:, 0].to_numpy()
  df = sweep(config, grid, lmm_obj=lmm_obj)
  sort_col = "gap_full" if "gap_full" in df.columns else "obj_mean"
  print(df.sort_values(sort_col).to_string(index=False))


if __name__ == "__main__":
  main()
