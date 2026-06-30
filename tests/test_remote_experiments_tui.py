from rich.console import Console
from ray_dispatcher import Inventory, RemoteHost

from remote_experiments.batch import Batch, Experiment
from remote_experiments.manifest import Manifest
from remote_experiments.stats import summarize
from remote_experiments.tui import render


def _batch():
  return Batch(suite="smoke", experiments=(
    Experiment(id="e1", suite="smoke", algorithm="centralized", seed=1,
               graph_params={}, load_params={}, config={}),
  ))


def test_render_includes_experiment_status_and_host(tmp_path):
  manifest = Manifest(tmp_path / "m.json")
  manifest.record("e1", status="running", host="vm1")
  inventory = Inventory((RemoteHost("vm1", user="ubuntu", slots=1),))
  stats = summarize(["e1"], manifest, inventory, {"e1": "vm1"}, elapsed_s=10.0)
  layout = render(_batch(), manifest, stats)

  console = Console(record=True, width=120)
  console.print(layout)
  output = console.export_text()
  assert "e1" in output
  assert "running" in output
  assert "vm1" in output
