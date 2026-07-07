"""Select survivor algorithms from screening results by relative-to-best objective."""

from __future__ import annotations

import statistics
from pathlib import Path

from .batch import Batch
from .results_reader import read_run_objective, read_run_runtime


class SurvivorSelectionError(RuntimeError):
  """Raised when screening results are too degenerate to promote survivors."""


def _instance_key(experiment) -> tuple[int, int, int]:
  limits = experiment.graph_params
  return (limits["Nn"]["min"], limits["Nf"]["min"], experiment.seed)


def select_survivors(
    batch: Batch,
    results_dir: Path,
    *,
    anchors: tuple[str, ...] = ("centralized", "hierarchical-madea"),
    n: int = 4,
    min_valid_fraction: float = 0.8,
  ) -> list[str]:
  results_dir = Path(results_dir)
  obj_by_algo: dict[str, dict[tuple, float]] = {}
  runtime_by_algo: dict[str, list[float]] = {}
  valid = 0
  for e in batch.experiments:
    objective = read_run_objective(results_dir / e.id)
    if objective is None:
      continue
    valid += 1
    obj_by_algo.setdefault(e.algorithm, {})[_instance_key(e)] = objective
    runtime = read_run_runtime(results_dir / e.id)
    if runtime is not None:
      runtime_by_algo.setdefault(e.algorithm, []).append(runtime)

  if valid < min_valid_fraction * len(batch.experiments):
    raise SurvivorSelectionError(
      f"only {valid}/{len(batch.experiments)} screening runs valid "
      f"(< {min_valid_fraction:.0%}); refusing to promote"
    )

  # obj_best per instance across every algorithm that ran it
  best: dict[tuple, float] = {}
  for per_instance in obj_by_algo.values():
    for key, value in per_instance.items():
      best[key] = max(value, best.get(key, float("-inf")))

  candidates = [a for a in obj_by_algo if a not in anchors]
  scored = []
  for algo in candidates:
    deficits = [
      (best[key] - obj) / best[key] * 100.0
      for key, obj in obj_by_algo[algo].items()
    ]
    reldef = statistics.fmean(deficits)
    runtime = statistics.median(runtime_by_algo.get(algo, [float("inf")]))
    scored.append((reldef, runtime, algo))

  if len(scored) < n:
    raise SurvivorSelectionError(
      f"only {len(scored)} candidate algorithms produced valid runs; need {n}"
    )
  scored.sort(key=lambda t: (t[0], t[1]))
  return [algo for _, _, algo in scored[:n]]
