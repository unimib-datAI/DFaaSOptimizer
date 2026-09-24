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
  runtime_by_algo: dict[str, dict[tuple, float]] = {}
  expected_by_algo: dict[str, set[tuple]] = {}
  valid = 0
  for e in batch.experiments:
    key = _instance_key(e)
    expected_by_algo.setdefault(e.algorithm, set()).add(key)
    objective = read_run_objective(results_dir / e.id)
    if objective is None:
      continue
    valid += 1
    obj_by_algo.setdefault(e.algorithm, {})[key] = objective
    runtime = read_run_runtime(results_dir / e.id)
    if runtime is not None:
      runtime_by_algo.setdefault(e.algorithm, {})[key] = runtime

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

  candidates = [
    algo for algo, results in obj_by_algo.items()
    if algo not in anchors
    and len(results) >= min_valid_fraction * len(expected_by_algo[algo])
  ]
  if len(candidates) < n:
    raise SurvivorSelectionError(
      f"only {len(candidates)} candidate algorithms have adequate valid coverage; need {n}"
    )
  shared = set.intersection(*(set(obj_by_algo[algo]) for algo in candidates))
  expected = set.union(*(expected_by_algo[algo] for algo in candidates))
  if not shared or len(shared) < min_valid_fraction * len(expected):
    raise SurvivorSelectionError(
      f"only {len(shared)}/{len(expected)} shared instances; refusing to promote"
    )
  scored = []
  for algo in candidates:
    deficits = [
      (best[key] - obj_by_algo[algo][key]) / (abs(best[key]) or 1.0) * 100.0
      for key in sorted(shared)
    ]
    reldef = statistics.fmean(deficits)
    runtime = statistics.median(
      runtime_by_algo.get(algo, {}).get(key, float("inf")) for key in sorted(shared)
    )
    scored.append((reldef, runtime, algo))

  scored.sort(key=lambda t: (t[0], t[1]))
  return [algo for _, _, algo in scored[:n]]
