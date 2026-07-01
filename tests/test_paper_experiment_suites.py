import numpy as np

from generators.generate_data import generate_data
from remote_experiments.batch import Batch
from remote_experiments.definitions import list_suites
from remote_experiments.definitions.paper import (
  build_e0, build_e1, build_e2, build_e3, build_e4, build_e5, build_e6, build_e7,
)


def test_paper_suites_are_registered():
  expected = {
    "paper-e0-pilot", "paper-e1-quality-runtime", "paper-e2-scalability",
    "paper-e3-topology", "paper-e4-robustness", "paper-e5-dynamics",
    "paper-e6-ablation", "paper-e7-tradeoffs",
  }
  assert expected <= set(list_suites())


def test_e0_default_count():
  assert len(build_e0()) == 120


def test_e1_default_count_and_unique_ids():
  experiments = build_e1()
  assert len(experiments) == 1800
  assert len({e.id for e in experiments}) == 1800


def test_e1_expands_function_vectors_and_output_folder():
  experiments = build_e1(seeds=(1001,), algorithms=("hierarchical",))
  four_functions = next(
    e for e in experiments if e.config["limits"]["Nf"]["min"] == 4
  )
  limits = four_functions.config["limits"]
  assert limits["demand"]["values"] == [1.0, 1.2, 1.0, 1.2]
  assert limits["memory_requirement"]["values"] == [2, 3, 2, 3]
  assert four_functions.config["base_solution_folder"] == f"solutions/{four_functions.id}"


def test_e2_count_and_centralized_size_limit():
  experiments = build_e2()
  assert len(experiments) == 4230
  centralized_sizes = {
    e.config["limits"]["Nn"]["min"]
    for e in experiments if e.algorithm == "centralized"
  }
  assert centralized_sizes == {10, 20}


def test_e3_count_and_topology_coverage():
  assert len(build_e3()) == 1260
  experiments = build_e3(seeds=(1001,))
  topologies = {
    tuple(sorted(e.config["limits"]["neighborhood"].items()))
    for e in experiments
  }
  assert topologies == {
    (("degree", 3), ("type", "planar")),
    (("k", 3),), (("k", 5),), (("k", 7),),
    (("p", 0.1),), (("p", 0.2),), (("p", 0.3),),
  }


def test_e4_count_and_conditions():
  experiments = build_e4(seeds=(1001,))
  conditions = {
    "baseline", "load-low", "load-high", "memory-scarce", "memory-ample",
    "nodes-homogeneous", "nodes-heterogeneous",
  }
  assert len(experiments) == 7 * 6
  assert {
    next(condition for condition in conditions if f"-{condition}-" in e.id)
    for e in experiments
  } == conditions


def test_e5_count_trace_coverage_and_steps():
  experiments = build_e5(seeds=(1001,))
  assert len(experiments) == 3 * 6
  assert {e.config["limits"]["load"]["trace_type"] for e in experiments} == {
    "sinusoidal", "clipped", "fixed_sum_minmax",
  }
  assert {e.config["max_steps"] for e in experiments} == {100}


def test_e6_default_count_and_hierarchical_only():
  experiments = build_e6()
  assert len(experiments) == 1200
  assert {e.algorithm for e in experiments} == {"hierarchical"}


def test_e7_default_count_and_weight_pairs():
  experiments = build_e7()
  assert len(experiments) == 1680
  sections = {
    "hierarchical": "auction", "faas-madea": "auction",
    "faas-diffuse": "diffusion", "faas-powd": "powerd",
  }
  assert {e.algorithm for e in experiments} == set(sections)
  assert {
    (
      e.config["solver_options"][sections[e.algorithm]]["latency_weight"],
      e.config["solver_options"][sections[e.algorithm]]["fairness_weight"],
    )
    for e in experiments
  } == {
    (0, 0), (0.25, 0), (1, 0), (0, 0.25),
    (0, 1), (0.25, 0.25), (1, 1),
  }


def test_paper_experiments_round_trip_as_batch(tmp_path):
  experiments = build_e1(seeds=(1001,), algorithms=("hierarchical",))
  path = tmp_path / "batch.json"
  expected = Batch(suite="paper-e1-quality-runtime", experiments=tuple(experiments))
  expected.save(path)
  assert Batch.load(path) == expected
  assert all(
    e.config["base_solution_folder"] == f"solutions/{e.id}"
    for e in experiments
  )


def test_generated_config_builds_a_real_instance():
  experiment = build_e1(seeds=(1001,), algorithms=("hierarchical",))[0]
  data, _, graph = generate_data(
    "random", np.random.default_rng(experiment.seed), experiment.config["limits"],
  )
  assert data[None]["Nn"][None] == 10
  assert data[None]["Nf"][None] == 2
  assert graph.number_of_nodes() == 10
