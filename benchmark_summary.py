#!/usr/bin/env python3
"""Standalone gap + runtime summary for a benchmark run.

Reads the raw per-method obj.csv / runtime.csv written under each solution
folder (mapped by <base>/experiments.json), so it does not depend on the
run.py postprocessing, which currently assumes the default centralized model
name and breaks with model_variant="tight".

Objective is social welfare (higher is better); the centralized run is the
reference optimum. Reported deviation, matching run.py:
    dev% = (obj_method - obj_centralized) / obj_centralized * 100
(negative => below the centralized optimum). Runtime is the solver time each
method reports in runtime.csv, summed over the horizon.

Usage:
    uv run python benchmark_summary.py solutions/bench_A solutions/bench_B ...
    uv run python benchmark_summary.py solutions/bench_A --csv out.csv
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

BASELINE = "centralized"


def _single_column_sum(path):
  """Sum the single data column of a one-column csv (obj.csv / runtime.csv)."""
  if not os.path.exists(path):
    return np.nan
  df = pd.read_csv(path)
  if df.shape[1] == 0 or len(df) == 0:
    return np.nan
  # centralized files may carry the model name as column; decentralized carry
  # the method label or "tot" -- in every case there is exactly one data column
  col = df.select_dtypes("number")
  if col.shape[1] == 0:
    return np.nan
  return float(col.iloc[:, 0].sum())


def _resolve(base, folder):
  if folder is None:
    return None
  for candidate in (folder, os.path.join(base, os.path.basename(str(folder)))):
    if candidate and os.path.isdir(candidate):
      return candidate
  return folder if os.path.isdir(str(folder)) else None


def summarize_base(base):
  exp_path = os.path.join(base, "experiments.json")
  if not os.path.exists(exp_path):
    print(f"! {base}: no experiments.json (run not completed?)", file=sys.stderr)
    return pd.DataFrame()
  folders = json.load(open(exp_path))
  experiments = folders.get("experiments_list", [])
  methods = [m for m in folders if m != "experiments_list"]
  rows = []
  for idx, exp in enumerate(experiments):
    for method in methods:
      flist = folders.get(method) or []
      folder = _resolve(base, flist[idx]) if idx < len(flist) else None
      if folder is None:
        rows.append({"instance": os.path.basename(base), "experiment": str(exp),
                     "method": method, "obj": np.nan, "runtime_s": np.nan})
        continue
      rows.append({
        "instance": os.path.basename(base),
        "experiment": str(exp),
        "method": method,
        "obj": _single_column_sum(os.path.join(folder, "obj.csv")),
        "runtime_s": _single_column_sum(os.path.join(folder, "runtime.csv")),
      })
  df = pd.DataFrame(rows)
  if df.empty:
    return df
  # deviation vs the centralized baseline, per (instance, experiment)
  df["dev_pct"] = np.nan
  for (_, _), grp in df.groupby(["instance", "experiment"]):
    base_row = grp[grp["method"] == BASELINE]
    if base_row.empty or not np.isfinite(base_row["obj"].iloc[0]):
      continue
    ref = base_row["obj"].iloc[0]
    if ref == 0:
      continue
    df.loc[grp.index, "dev_pct"] = (grp["obj"] - ref) / ref * 100.0
  return df


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("bases", nargs="+", help="benchmark base folders (solutions/bench_A ...)")
  ap.add_argument("--csv", help="also write the full long-form table here")
  args = ap.parse_args()

  full = pd.concat([summarize_base(b) for b in args.bases], ignore_index=True)
  if full.empty:
    print("no results found", file=sys.stderr)
    sys.exit(1)

  pd.set_option("display.max_rows", None)
  pd.set_option("display.width", 160)
  # average over experiments per (instance, method)
  agg = (full.groupby(["instance", "method"], sort=False)
              .agg(n=("obj", lambda s: int(s.notna().sum())),
                   obj=("obj", "mean"),
                   dev_pct=("dev_pct", "mean"),
                   runtime_s=("runtime_s", "mean"))
              .reset_index())
  for instance in agg["instance"].unique():
    sub = agg[agg["instance"] == instance].copy()
    sub = sub.sort_values("dev_pct", ascending=False, na_position="last")
    print(f"\n=== {instance} (obj = welfare, higher better; dev% vs centralized) ===")
    print(sub.to_string(index=False,
                        formatters={"obj": "{:.4g}".format,
                                    "dev_pct": lambda v: "baseline" if v == 0
                                    else ("n/a" if pd.isna(v) else f"{v:+.2f}%"),
                                    "runtime_s": lambda v: "n/a" if pd.isna(v)
                                    else f"{v:.3f}"}))
  if args.csv:
    full.to_csv(args.csv, index=False)
    print(f"\nfull table -> {args.csv}")


if __name__ == "__main__":
  main()
