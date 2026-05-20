from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyomo.environ as pyo

from hierarchical_auction.runner import run as run_hierarchical
from run_centralized_model import run as run_centralized
from run_faasmacro import run as run_distributed


DEFAULT_SIZES = [20, 40, 50]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_SOLVER = "gurobi"
MODEL_ORDER = ["centralized", "distributed", "hierarchical"]


def parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Run planar degree-3 benchmark across centralized, distributed, and hierarchical models",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument(
    "--sizes",
    nargs="+",
    type=int,
    default=DEFAULT_SIZES,
    help="Node counts to benchmark",
  )
  parser.add_argument(
    "--seeds",
    nargs="+",
    type=int,
    default=DEFAULT_SEEDS,
    help="Seeds used to generate each graph instance",
  )
  parser.add_argument(
    "--solver-name",
    type=str,
    default=DEFAULT_SOLVER,
    help="MILP solver name",
  )
  parser.add_argument(
    "--output-root",
    type=Path,
    default=Path("solutions") / "planar_benchmark",
    help="Root output directory",
  )
  return parser.parse_args()


def _require_gurobi(solver_name: str) -> None:
  solver = pyo.SolverFactory(solver_name)
  if not solver.available(exception_flag=False):
    raise SystemExit(f"{solver_name} solver is not available")


def build_base_config(output_root: Path, nn: int, seed: int, solver_name: str = DEFAULT_SOLVER) -> dict[str, Any]:
  return {
    "base_solution_folder": str(output_root / f"n{nn}" / f"seed{seed}"),
    "seed": seed,
    "limits": {
      "Nn": {"min": nn, "max": nn},
      "Nf": {"min": 2, "max": 2},
      "neighborhood": {"type": "planar", "degree": 3},
      "weights": {
        "alpha": {"min": 1.0, "max": 1.5},
        "beta_multiplier": {"min": 1.5, "max": 2.5},
        "gamma": {"min": 0.05, "max": 0.15},
        "delta_multiplier": {"min": 0.1, "max": 0.2},
      },
      "demand": {"values": [1.0, 1.2]},
      "memory_capacity": {"values": [12] * nn},
      "memory_requirement": {"values": [2, 3]},
      "max_utilization": {"min": 0.65, "max": 0.75},
      "load": {"trace_type": "fixed_sum", "values": [2.0 * nn, 3.0 * nn]},
    },
    "solver_name": solver_name,
    "solver_options": {
      "general": {"TimeLimit": 60, "OutputFlag": 0},
      "auction": {
        "epsilon": 0.01,
        "eta": [0.5, 0.3, 0.15],
        "zeta": 0.1,
        "latency_weight": 0.0,
        "fairness_weight": 0.0,
      },
    },
    "max_iterations": 20,
    "patience": 5,
    "sw_patience": 5,
    "max_steps": 8,
    "min_run_time": 1,
    "max_run_time": 1,
    "run_time_step": 1,
    "checkpoint_interval": 1,
    "max_hierarchy_depth": 3,
    "tolerance": 1e-6,
    "verbose": 0,
  }


def _read_scalar_from_csv(csv_path: Path, preferred_column: str | None = None) -> float:
  df = pd.read_csv(csv_path)
  value_frame = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
  if preferred_column and preferred_column in value_frame.columns:
    series = pd.to_numeric(value_frame[preferred_column], errors="coerce").dropna()
    if len(series):
      return float(series.iloc[-1])
  numeric = value_frame.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
  if numeric.empty:
    raise ValueError(f"no numeric data found in {csv_path}")
  row = numeric.iloc[-1].dropna()
  if row.empty:
    raise ValueError(f"no numeric row found in {csv_path}")
  return float(row.iloc[-1])


def _read_termination_text(csv_path: Path) -> str:
  df = pd.read_csv(csv_path)
  return " | ".join(df.astype(str).to_numpy().ravel())


def _collect_run_result(
  model: str,
  folder: str,
  wallclock_s: float,
) -> dict[str, Any]:
  run_folder = Path(folder)
  preferred_column = {
    "centralized": "LoadManagementModel",
    "distributed": "FaaS-MACrO",
    "hierarchical": "HierarchicalAuction",
  }[model]
  return {
    "model": model,
    "folder": str(run_folder),
    "objective": _read_scalar_from_csv(run_folder / "obj.csv", preferred_column),
    "wallclock_s": round(wallclock_s, 6),
    "termination": _read_termination_text(run_folder / "termination_condition.csv"),
  }


def aggregate_results(raw: pd.DataFrame) -> pd.DataFrame:
  ordered = raw.copy()
  ordered["model"] = pd.Categorical(ordered["model"], categories=MODEL_ORDER, ordered=True)
  grouped = ordered.groupby(["Nn", "model"], as_index=False, observed=True).agg(
    objective_mean=("objective", "mean"),
    objective_std=("objective", lambda s: s.std(ddof=1) if len(s) > 1 else 0.0),
    wallclock_mean_s=("wallclock_s", "mean"),
    wallclock_std_s=("wallclock_s", lambda s: s.std(ddof=1) if len(s) > 1 else 0.0),
    runs=("seed", "count"),
  )
  grouped["model"] = pd.Categorical(grouped["model"], categories=MODEL_ORDER, ordered=True)
  return grouped.sort_values(["Nn", "model"]).reset_index(drop=True)


def render_html_report(
  summary: pd.DataFrame,
  raw: pd.DataFrame,
  html_path: Path,
  meta: dict[str, object],
) -> None:
  display_summary = summary.rename(
    columns={
      "objective_mean": "Objective mean",
      "objective_std": "Objective std",
      "wallclock_mean_s": "Wallclock mean (s)",
      "wallclock_std_s": "Wallclock std (s)",
      "runs": "Runs",
    }
  )
  display_raw = raw.rename(
    columns={
      "Nn": "Nodes",
      "seed": "Seed",
      "model": "Model",
      "objective": "Objective",
      "wallclock_s": "Wallclock (s)",
      "termination": "Termination",
      "folder": "Folder",
    }
  )
  sizes = meta.get("sizes", [])
  seeds = meta.get("seeds", [])
  generated_at = meta.get("generated_at", "")
  html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Planar Benchmark Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; }}
    h1, h2 {{ margin-bottom: 0.4rem; }}
    .meta {{ margin-bottom: 1.5rem; }}
    .meta span {{ display: inline-block; margin-right: 16px; }}
    table {{ border-collapse: collapse; margin-bottom: 1.5rem; width: 100%; }}
    th, td {{ border: 1px solid #d1d5db; padding: 6px 8px; text-align: left; }}
    th {{ background: #f3f4f6; }}
    tbody tr:nth-child(even) {{ background: #fafafa; }}
  </style>
</head>
<body>
  <h1>Planar Benchmark</h1>
  <div class="meta">
    <span><strong>Generated at:</strong> {generated_at}</span>
    <span><strong>Sizes:</strong> {sizes}</span>
    <span><strong>Seeds:</strong> {seeds}</span>
  </div>
  <h2>Summary</h2>
  {display_summary.to_html(index=False, border=0, escape=False)}
  <h2>Raw Results</h2>
  {display_raw.to_html(index=False, border=0, escape=False)}
</body>
</html>
"""
  html_path.write_text(html, encoding="utf-8")


def write_report(
  raw: pd.DataFrame,
  summary: pd.DataFrame,
  output_root: Path,
  meta: dict[str, object],
) -> None:
  output_root.mkdir(parents=True, exist_ok=True)
  raw.to_csv(output_root / "raw_results.csv", index=False)
  summary.to_csv(output_root / "summary.csv", index=False)
  render_html_report(summary, raw, output_root / "summary.html", meta)


def run_benchmark(
  output_root: Path,
  sizes: list[int],
  seeds: list[int],
  solver_name: str = DEFAULT_SOLVER,
) -> pd.DataFrame:
  _require_gurobi(solver_name)
  benchmark_root = output_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
  benchmark_root.mkdir(parents=True, exist_ok=True)
  rows: list[dict[str, Any]] = []

  for nn in sizes:
    for seed in seeds:
      config = build_base_config(benchmark_root, nn, seed, solver_name=solver_name)
      for model_name, runner in [
        ("centralized", run_centralized),
        ("distributed", run_distributed),
        ("hierarchical", run_hierarchical),
      ]:
        model_root = benchmark_root / f"n{nn}" / f"seed{seed}" / model_name
        config["base_solution_folder"] = str(model_root)
        started = time.perf_counter()
        runner_kwargs = {"disable_plotting": True}
        if model_name != "centralized":
          runner_kwargs["parallelism"] = 0
        folder = runner(config, **runner_kwargs)
        wallclock_s = time.perf_counter() - started
        rows.append({
          "Nn": nn,
          "seed": seed,
          "model": model_name,
          **_collect_run_result(model_name, folder, wallclock_s),
        })

  raw = pd.DataFrame(rows)
  raw["model"] = pd.Categorical(raw["model"], categories=MODEL_ORDER, ordered=True)
  raw.sort_values(["Nn", "seed", "model"], inplace=True)
  raw.reset_index(drop=True, inplace=True)
  summary = aggregate_results(raw)
  write_report(
    raw,
    summary,
    benchmark_root,
    meta={
      "sizes": sizes,
      "seeds": seeds,
      "solver_name": solver_name,
      "generated_at": datetime.now().isoformat(timespec="seconds"),
    },
  )
  return raw


def main() -> None:
  args = parse_arguments()
  raw = run_benchmark(
    output_root=args.output_root,
    sizes=args.sizes,
    seeds=args.seeds,
    solver_name=args.solver_name,
  )
  print(json.dumps({
    "raw_rows": len(raw),
    "summary_rows": len(aggregate_results(raw)),
    "output_root": str(args.output_root),
  }, indent=2))


if __name__ == "__main__":
  main()
