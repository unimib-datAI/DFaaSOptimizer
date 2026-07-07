"""Read a single experiment's objective/runtime from its output CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _first_column_mean(run_dir: Path, filename: str) -> float | None:
  matches = sorted(Path(run_dir).rglob(filename))
  if not matches:
    return None
  frame = pd.read_csv(matches[0])
  if frame.empty or frame.shape[1] == 0:
    return None
  return float(frame.iloc[:, 0].mean())


def read_run_objective(run_dir: Path) -> float | None:
  return _first_column_mean(run_dir, "obj.csv")


def read_run_runtime(run_dir: Path) -> float | None:
  return _first_column_mean(run_dir, "runtime.csv")


def run_failed(run_dir: Path) -> bool:
  return read_run_objective(run_dir) is None
