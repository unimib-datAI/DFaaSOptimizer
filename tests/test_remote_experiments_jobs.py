import json

import pytest

from remote_experiments.batch import Experiment
from remote_experiments.jobs import SCRIPT_BY_ALGORITHM, experiment_to_job


def _experiment(algorithm="centralized"):
  return Experiment(
    id="e1", suite="smoke", algorithm=algorithm, seed=1,
    graph_params={}, load_params={}, config={"seed": 1, "base_solution_folder": "solutions/e1"},
  )


def test_all_ten_algorithms_are_mapped():
  expected = {
    "centralized", "faas-macro", "faas-macro-v0", "faas-madea", "hierarchical",
    "faas-diffuse", "faas-powd", "faas-br-s", "faas-br-r", "faas-br-o",
  }
  assert set(SCRIPT_BY_ALGORITHM) == expected


def test_experiment_to_job_builds_expected_command_for_variant_algorithm(tmp_path):
  job = experiment_to_job(_experiment("faas-br-r"), tmp_path)
  assert job.command == (
    "uv", "run", "decentralized_bestresponse.py", "-c", "config.json",
    "--disable_plotting", "--variant", "r",
  )


def test_experiment_to_job_builds_expected_command_for_plain_algorithm(tmp_path):
  job = experiment_to_job(_experiment("centralized"), tmp_path)
  assert job.command == (
    "uv", "run", "run_centralized_model.py", "-c", "config.json", "--disable_plotting",
  )


def test_experiment_to_job_writes_config_file(tmp_path):
  job = experiment_to_job(_experiment(), tmp_path)
  config_path = tmp_path / "e1.json"
  assert config_path.exists()
  assert json.loads(config_path.read_text())["seed"] == 1
  assert job.inputs[0].destination == "config.json"


def test_experiment_to_job_output_matches_experiment_id(tmp_path):
  job = experiment_to_job(_experiment(), tmp_path)
  assert job.outputs[0].source == "solutions/e1"
  assert job.outputs[0].destination == "e1"


def test_experiment_to_job_unknown_algorithm_raises(tmp_path):
  with pytest.raises(KeyError):
    experiment_to_job(_experiment("nope"), tmp_path)
