"""Full experiment suites for the hierarchical-model journal evaluation."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from ..batch import Experiment
from . import register_suite

_BASE_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config_files" / "eval_full.json"

PILOT_SEEDS = tuple(range(1, 11))
CONFIRMATORY_SEEDS = tuple(range(1001, 1031))

ALL_ALGORITHMS = (
  "centralized", "faas-macro", "faas-macro-v0", "faas-madea", "hierarchical",
  "faas-diffuse", "faas-powd", "faas-br-s", "faas-br-r", "faas-br-o",
)
NON_CENTRALIZED_ALGORITHMS = tuple(a for a in ALL_ALGORITHMS if a != "centralized")
REPRESENTATIVE_ALGORITHMS = (
  "hierarchical", "faas-macro", "faas-madea", "faas-diffuse", "faas-powd", "faas-br-o",
)
TRADEOFF_ALGORITHMS = ("hierarchical", "faas-madea", "faas-diffuse", "faas-powd")


def _base_config() -> dict:
  return json.loads(_BASE_CONFIG_PATH.read_text())


def _set_dimensions(config: dict, nodes: int, functions: int) -> None:
  limits = config["limits"]
  limits["Nn"] = {"min": nodes, "max": nodes}
  limits["Nf"] = {"min": functions, "max": functions}
  repeats = (functions + 1) // 2
  limits["demand"]["values"] = ([1.0, 1.2] * repeats)[:functions]
  limits["memory_requirement"]["values"] = ([2, 3] * repeats)[:functions]


def _new_config(nodes: int, functions: int, topology: dict) -> dict:
  config = _base_config()
  _set_dimensions(config, nodes, functions)
  config["limits"]["neighborhood"] = copy.deepcopy(topology)
  return config


def _experiment(
    suite: str, cell: str, algorithm: str, seed: int, config: dict,
  ) -> Experiment:
  experiment_id = f"{suite}-{cell}-{algorithm}-s{seed}"
  config["seed"] = seed
  config["base_solution_folder"] = f"solutions/{experiment_id}"
  limits = config["limits"]
  return Experiment(
    id=experiment_id,
    suite=suite,
    algorithm=algorithm,
    seed=seed,
    graph_params={k: copy.deepcopy(v) for k, v in limits.items() if k != "load"},
    load_params=copy.deepcopy(limits["load"]),
    config=config,
  )


@register_suite("paper-e0-pilot")
def build_e0(
    seeds: tuple[int, ...] = PILOT_SEEDS,
    algorithms: tuple[str, ...] = (
      "centralized", "hierarchical", "faas-macro", "faas-madea",
    ),
  ) -> list[Experiment]:
  suite = "paper-e0-pilot"
  return [
    _experiment(
      suite, f"n{nodes}-f2-er03", algorithm, seed,
      _new_config(nodes, 2, {"p": 0.3}),
    )
    for nodes in (10, 20, 50)
    for algorithm in algorithms
    for seed in seeds
  ]


@register_suite("paper-e1-quality-runtime")
def build_e1(
    seeds: tuple[int, ...] = CONFIRMATORY_SEEDS,
    algorithms: tuple[str, ...] = ALL_ALGORITHMS,
  ) -> list[Experiment]:
  suite = "paper-e1-quality-runtime"
  return [
    _experiment(
      suite, f"n{nodes}-f{functions}-planar3", algorithm, seed,
      _new_config(nodes, functions, {"type": "planar", "degree": 3}),
    )
    for nodes in (10, 20, 30)
    for functions in (2, 4)
    for algorithm in algorithms
    for seed in seeds
  ]


@register_suite("paper-e2-scalability")
def build_e2(
    seeds: tuple[int, ...] = CONFIRMATORY_SEEDS,
  ) -> list[Experiment]:
  suite = "paper-e2-scalability"
  experiments = []
  for nodes in (10, 20, 50, 100, 200):
    algorithms = NON_CENTRALIZED_ALGORITHMS + (
      ("centralized",) if nodes <= 20 else ()
    )
    for functions in (2, 4, 8):
      for algorithm in algorithms:
        for seed in seeds:
          experiments.append(_experiment(
            suite, f"n{nodes}-f{functions}-reg3", algorithm, seed,
            _new_config(nodes, functions, {"k": 3}),
          ))
  return experiments


@register_suite("paper-e3-topology")
def build_e3(
    seeds: tuple[int, ...] = CONFIRMATORY_SEEDS,
    algorithms: tuple[str, ...] = REPRESENTATIVE_ALGORITHMS,
  ) -> list[Experiment]:
  suite = "paper-e3-topology"
  topologies = (
    ("planar3", {"type": "planar", "degree": 3}),
    ("reg3", {"k": 3}), ("reg5", {"k": 5}), ("reg7", {"k": 7}),
    ("er01", {"p": 0.1}), ("er02", {"p": 0.2}), ("er03", {"p": 0.3}),
  )
  return [
    _experiment(
      suite, topology_name, algorithm, seed,
      _new_config(50, 4, topology),
    )
    for topology_name, topology in topologies
    for algorithm in algorithms
    for seed in seeds
  ]


def _apply_robustness_condition(config: dict, condition: str) -> None:
  limits = config["limits"]
  if condition == "load-low":
    limits["load"].update({"min": {"min": 2, "max": 5}, "max": {"min": 20, "max": 40}})
  elif condition == "load-high":
    limits["load"].update({"min": {"min": 10, "max": 20}, "max": {"min": 100, "max": 150}})
  elif condition == "memory-scarce":
    limits["memory_capacity"] = {"min": 8, "max": 8}
  elif condition == "memory-ample":
    limits["memory_capacity"] = {"min": 24, "max": 24}
  elif condition == "nodes-homogeneous":
    limits["demand"] = {"values": [1.1] * 4}
    limits["memory_capacity"] = {"min": 12, "max": 12}
    limits["max_utilization"] = {"min": 0.7, "max": 0.7}
  elif condition == "nodes-heterogeneous":
    limits["demand"] = {"type": "heterogeneous", "min": 0.5, "max": 2.0}
    limits["memory_capacity"] = {"repeated_values": [[0.5, 8], [0.5, 24]]}


@register_suite("paper-e4-robustness")
def build_e4(
    seeds: tuple[int, ...] = CONFIRMATORY_SEEDS,
    algorithms: tuple[str, ...] = REPRESENTATIVE_ALGORITHMS,
  ) -> list[Experiment]:
  suite = "paper-e4-robustness"
  conditions = (
    "baseline", "load-low", "load-high", "memory-scarce", "memory-ample",
    "nodes-homogeneous", "nodes-heterogeneous",
  )
  experiments = []
  for condition in conditions:
    for algorithm in algorithms:
      for seed in seeds:
        config = _new_config(50, 4, {"type": "planar", "degree": 3})
        _apply_robustness_condition(config, condition)
        experiments.append(_experiment(suite, condition, algorithm, seed, config))
  return experiments


@register_suite("paper-e5-dynamics")
def build_e5(
    seeds: tuple[int, ...] = CONFIRMATORY_SEEDS,
    algorithms: tuple[str, ...] = REPRESENTATIVE_ALGORITHMS,
  ) -> list[Experiment]:
  suite = "paper-e5-dynamics"
  experiments = []
  for trace_type in ("sinusoidal", "clipped", "fixed_sum_minmax"):
    for algorithm in algorithms:
      for seed in seeds:
        config = _new_config(50, 4, {"type": "planar", "degree": 3})
        config.update({
          "max_steps": 100, "min_run_time": 0,
          "max_run_time": 100, "run_time_step": 1,
        })
        config["limits"]["load"]["trace_type"] = trace_type
        experiments.append(_experiment(
          suite, trace_type.replace("_", "-"), algorithm, seed, config,
        ))
  return experiments


def _apply_ablation(config: dict, variant: str) -> None:
  auction = config["solver_options"]["auction"]
  if variant.startswith("depth"):
    config["max_hierarchy_depth"] = int(variant.removeprefix("depth"))
  elif variant.startswith("eta"):
    auction["eta"] = {"eta0": 0.0, "eta01": 0.1, "eta05": 0.5}[variant]
  elif variant.startswith("eps"):
    auction["epsilon"] = {"eps0001": 0.001, "eps01": 0.1}[variant]


@register_suite("paper-e6-ablation")
def build_e6(
    seeds: tuple[int, ...] = CONFIRMATORY_SEEDS,
  ) -> list[Experiment]:
  suite = "paper-e6-ablation"
  variants = (
    "base", "depth1", "depth2", "depth4", "depth5",
    "eta0", "eta01", "eta05", "eps0001", "eps01",
  )
  topologies = (
    ("planar3", {"type": "planar", "degree": 3}),
    ("reg3", {"k": 3}),
  )
  experiments = []
  for variant in variants:
    for nodes in (20, 50):
      for topology_name, topology in topologies:
        for seed in seeds:
          config = _new_config(nodes, 4, topology)
          _apply_ablation(config, variant)
          experiments.append(_experiment(
            suite, f"{variant}-n{nodes}-{topology_name}",
            "hierarchical", seed, config,
          ))
  return experiments


@register_suite("paper-e7-tradeoffs")
def build_e7(
    seeds: tuple[int, ...] = CONFIRMATORY_SEEDS,
    algorithms: tuple[str, ...] = TRADEOFF_ALGORITHMS,
  ) -> list[Experiment]:
  suite = "paper-e7-tradeoffs"
  weights = (
    ("l0-f0", 0, 0), ("l025-f0", 0.25, 0), ("l1-f0", 1, 0),
    ("l0-f025", 0, 0.25), ("l0-f1", 0, 1),
    ("l025-f025", 0.25, 0.25), ("l1-f1", 1, 1),
  )
  topologies = (
    ("planar3", {"type": "planar", "degree": 3}),
    ("reg3", {"k": 3}),
  )
  experiments = []
  for weight_name, latency_weight, fairness_weight in weights:
    for topology_name, topology in topologies:
      for algorithm in algorithms:
        for seed in seeds:
          config = _new_config(50, 4, topology)
          section = {
            "hierarchical": "auction", "faas-madea": "auction",
            "faas-diffuse": "diffusion", "faas-powd": "powerd",
          }[algorithm]
          options = config["solver_options"][section]
          options["latency_weight"] = latency_weight
          options["fairness_weight"] = fairness_weight
          experiments.append(_experiment(
            suite, f"{weight_name}-{topology_name}", algorithm, seed, config,
          ))
  return experiments
