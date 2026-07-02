import copy
import json
from dataclasses import replace

import pytest

from remote_experiments.batch import Batch
from remote_experiments.definitions.paper import build_e0
from remote_experiments.instances import (
  instance_id,
  load_materialized_instance,
  materialize_batch,
  validate_instance,
)


def _experiments():
  return build_e0(
    seeds=(42,), algorithms=("centralized", "hierarchical"),
  )[:2]


def test_instance_identity_ignores_algorithm_configuration():
  centralized, hierarchical = _experiments()
  assert instance_id(centralized) == instance_id(hierarchical)


def test_instance_identity_ignores_algorithm_seed():
  experiment = _experiments()[0]
  config = copy.deepcopy(experiment.config)
  config["seed"] = 999
  changed = replace(experiment, seed=999, config=config)
  assert instance_id(experiment) == instance_id(changed)


def test_instance_identity_changes_with_generation_inputs():
  experiment = _experiments()[0]
  changed = copy.deepcopy(experiment)
  changed.config["limits"]["memory_capacity"] = {"min": 24, "max": 24}
  assert instance_id(experiment) != instance_id(changed)


def test_materialize_batch_deduplicates_and_writes_complete_instance(tmp_path):
  experiments = _experiments()
  suite_path = materialize_batch(
    Batch(suite="smoke", experiments=tuple(experiments)), tmp_path,
  )

  data_dirs = list((suite_path / "data").iterdir())
  assert len(data_dirs) == 1
  assert {path.name for path in data_dirs[0].iterdir()} == {
    "base_instance_data.json",
    "load_limits.json",
    "input_requests_traces.json",
    "graph.json",
    "metadata.json",
  }
  manifest = json.loads((suite_path / "manifest.json").read_text())
  assert manifest["experiments"] == {
    experiment.id: instance_id(experiment) for experiment in experiments
  }
  assert (suite_path / "README.md").exists()


def test_materialized_instance_round_trips_load_and_graph(tmp_path):
  experiment = _experiments()[0]
  suite_path = materialize_batch(
    Batch(suite="smoke", experiments=(experiment,)), tmp_path,
  )
  instance_path = suite_path / "data" / instance_id(experiment)

  base_data, traces, agents, graph = load_materialized_instance(instance_path)

  assert base_data[None]["Nn"][None] == 10
  assert len(traces[0][0]) == experiment.config["max_steps"]
  assert list(agents) == list(range(10))
  assert graph.number_of_nodes() == 10
  assert all("network_latency" in attrs for _, _, attrs in graph.edges(data=True))


def test_validate_instance_rejects_checksum_mismatch(tmp_path):
  experiment = _experiments()[0]
  suite_path = materialize_batch(
    Batch(suite="smoke", experiments=(experiment,)), tmp_path,
  )
  instance_path = suite_path / "data" / instance_id(experiment)
  graph_path = instance_path / "graph.json"
  graph_path.write_text(graph_path.read_text() + "\n")

  with pytest.raises(ValueError, match="checksum mismatch.*graph.json"):
    validate_instance(instance_path)
