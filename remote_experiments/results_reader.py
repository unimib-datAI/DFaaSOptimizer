"""Read a single experiment's objective/runtime from its output CSVs."""

from __future__ import annotations

from pathlib import Path
from math import isfinite

import pandas as pd


def _first_column_mean(run_dir: Path, filename: str, prefer: str | None = None) -> float | None:
  matches = sorted(Path(run_dir).rglob(filename))
  if not matches:
    return None
  try:
    frame = pd.read_csv(matches[0])
  except (pd.errors.EmptyDataError, pd.errors.ParserError):
    return None
  if frame.empty or frame.shape[1] == 0:
    return None
  column = frame[prefer] if prefer in frame else frame.iloc[:, 0]
  values = pd.to_numeric(column, errors="coerce")
  if not values.map(isfinite).all():
    return None
  mean = float(values.mean())
  return mean if isfinite(mean) else None


def read_run_objective(run_dir: Path) -> float | None:
  return _first_column_mean(run_dir, "obj.csv")


def read_run_runtime(run_dir: Path) -> float | None:
  # harmonized methods: modelled algorithm time (utils/step_time.py); older files: one column
  return _first_column_mean(run_dir, "runtime.csv", prefer="algorithm_s")


def run_failed(run_dir: Path) -> bool:
  return read_run_objective(run_dir) is None
