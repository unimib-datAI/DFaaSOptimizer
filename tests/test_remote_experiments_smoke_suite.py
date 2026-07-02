from remote_experiments.definitions import get_suite
from remote_experiments.definitions.smoke import ALGORITHMS, build


def test_smoke_suite_is_registered():
  assert get_suite("smoke") is build


def test_build_produces_one_experiment_per_seed_and_algorithm():
  experiments = build(seeds=(1, 2), algorithms=("centralized", "faas-macro"))
  assert len(experiments) == 4


def test_build_produces_unique_ids():
  experiments = build(seeds=(1, 2), algorithms=("centralized", "faas-macro"))
  ids = [e.id for e in experiments]
  assert len(ids) == len(set(ids))


def test_build_sets_seed_in_config_and_base_solution_folder():
  experiments = build(seeds=(7,), algorithms=("centralized",))
  e = experiments[0]
  assert e.config["seed"] == 7
  assert e.config["base_solution_folder"] == f"solutions/{e.id}"


def test_build_default_covers_all_eleven_algorithms():
  experiments = build()
  assert {e.algorithm for e in experiments} == set(ALGORITHMS)
  assert len(ALGORITHMS) == 11
  assert {"hierarchical", "hierarchical-madea"} <= set(ALGORITHMS)
