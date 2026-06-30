"""Example pluggable suite: every algorithm across a couple of seeds, from eval_smoke.json."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from ..batch import Experiment
from . import register_suite

_BASE_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config_files" / "eval_smoke.json"

ALGORITHMS = (
  "centralized", "faas-macro", "faas-macro-v0", "faas-madea", "hierarchical",
  "faas-diffuse", "faas-powd", "faas-br-s", "faas-br-r", "faas-br-o",
)


@register_suite("smoke")
def build(
    seeds: tuple[int, ...] = (42, 43),
    algorithms: tuple[str, ...] = ALGORITHMS,
  ) -> list[Experiment]:
  base_config = json.loads(_BASE_CONFIG_PATH.read_text())
  limits = base_config["limits"]
  graph_params = {k: v for k, v in limits.items() if k != "load"}
  load_params = limits["load"]
  experiments = []
  for seed in seeds:
    for algorithm in algorithms:
      config = copy.deepcopy(base_config)
      config["seed"] = seed
      experiment_id = f"smoke-{algorithm}-seed{seed}"
      config["base_solution_folder"] = f"solutions/{experiment_id}"
      experiments.append(Experiment(
        id=experiment_id, suite="smoke", algorithm=algorithm, seed=seed,
        graph_params=graph_params, load_params=load_params, config=config,
      ))
  return experiments
