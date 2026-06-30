import json

from remote_experiments.batch import Batch, Experiment


def _experiment(id="e1", seed=1):
  return Experiment(
    id=id, suite="smoke", algorithm="centralized", seed=seed,
    graph_params={"Nn": 10}, load_params={"trace_type": "sinusoidal"},
    config={"seed": seed, "base_solution_folder": f"solutions/{id}"},
  )


def test_experiment_round_trips_through_dict():
  e = _experiment()
  assert Experiment.from_dict(e.to_dict()) == e


def test_batch_save_and_load_round_trips(tmp_path):
  batch = Batch(suite="smoke", experiments=(_experiment("e1", 1), _experiment("e2", 2)))
  path = tmp_path / "batch.json"
  batch.save(path)
  loaded = Batch.load(path)
  assert loaded == batch


def test_batch_save_writes_readable_json(tmp_path):
  batch = Batch(suite="smoke", experiments=(_experiment(),))
  path = tmp_path / "batch.json"
  batch.save(path)
  raw = json.loads(path.read_text())
  assert raw["suite"] == "smoke"
  assert raw["experiments"][0]["id"] == "e1"
