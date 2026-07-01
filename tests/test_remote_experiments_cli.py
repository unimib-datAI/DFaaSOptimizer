import json

import pytest
from ray_dispatcher import JobHandle, JobStatus

from remote_experiments.batch import Batch, Experiment
from remote_experiments.cli import build_parser, cmd_define, cmd_run
from remote_experiments.manifest import Manifest


def test_define_accepts_registered_suite_name():
  parser = build_parser()
  args = parser.parse_args(["define", "smoke", "-o", "/tmp/x.json"])
  assert args.suite == "smoke"


def test_define_rejects_unregistered_suite_name():
  parser = build_parser()
  with pytest.raises(SystemExit):
    parser.parse_args(["define", "does-not-exist", "-o", "/tmp/x.json"])


def test_cmd_define_writes_batch_file(tmp_path):
  out_path = tmp_path / "out" / "batch.json"
  args = build_parser().parse_args(["define", "smoke", "-o", str(out_path)])
  cmd_define(args)
  loaded = Batch.load(out_path)
  assert loaded.suite == "smoke"
  assert len(loaded.experiments) == 20  # smoke suite default: 2 seeds x 10 algorithms


def test_cmd_define_output_is_valid_json(tmp_path):
  out_path = tmp_path / "batch.json"
  cmd_define(build_parser().parse_args(["define", "smoke", "-o", str(out_path)]))
  raw = json.loads(out_path.read_text())
  assert raw["suite"] == "smoke"


def test_run_subcommand_parses_required_arguments():
  parser = build_parser()
  args = parser.parse_args(["run", "batches/foo.json", "--inventory", "inv.yaml"])
  assert args.batch_file == "batches/foo.json"
  assert args.inventory == "inv.yaml"
  assert args.results_dir == "./results"


class _FakeDispatcher:
  """Context-manager FakeDispatcher matching ray_dispatcher.Dispatcher's duck type.

  Adapted from tests/test_remote_experiments_runner.py's FakeDispatcher — same
  submit/status/cancel/running_hosts contract, plus __enter__/__exit__ since
  cmd_run uses `with Dispatcher(...) as dispatcher:`.
  """

  def __init__(self, *_args, **_kwargs):
    self._sequences = {"e1": [JobStatus.RUNNING, JobStatus.SUCCEEDED]}

  def __enter__(self):
    return self

  def __exit__(self, *exc_info):
    return False

  def submit(self, jobs):
    return [JobHandle(batch_id="b1", job_id=j.id, token=j.id) for j in jobs]

  def status(self, handle):
    seq = self._sequences[handle.job_id]
    return seq.pop(0) if len(seq) > 1 else seq[0]

  def cancel(self, handle):
    pass

  def running_hosts(self):
    return {"e1": "10.0.0.10"}


def test_cmd_run_wires_dispatcher_into_manifest(tmp_path, monkeypatch):
  experiment = Experiment(
    id="e1", suite="smoke", algorithm="centralized", seed=42,
    graph_params={}, load_params={}, config={"seed": 42},
  )
  batch = Batch(suite="smoke", experiments=(experiment,))
  batch_path = tmp_path / "b.json"
  batch.save(batch_path)

  inventory_path = tmp_path / "inventory.yaml"
  inventory_path.write_text("hosts:\n  - host: 10.0.0.10\n    user: ubuntu\n    slots: 1\n")

  monkeypatch.setattr("remote_experiments.cli.Dispatcher", _FakeDispatcher)
  monkeypatch.setattr("builtins.input", lambda prompt: "all")

  args = build_parser().parse_args([
    "run", str(batch_path), "--inventory", str(inventory_path),
  ])
  cmd_run(args)

  manifest = Manifest(batch_path.with_suffix(".manifest.json"))
  assert manifest.status("e1") == "succeeded"
  assert manifest.host("e1") == "10.0.0.10"


def test_cmd_run_reports_terminal_failures(tmp_path, monkeypatch, capsys):
  class _FailedDispatcher(_FakeDispatcher):
    def __init__(self, *_args, **_kwargs):
      self._sequences = {"e1": [JobStatus.FAILED]}

  experiment = Experiment(
    id="e1", suite="smoke", algorithm="centralized", seed=42,
    graph_params={}, load_params={}, config={"seed": 42},
  )
  batch_path = tmp_path / "b.json"
  Batch(suite="smoke", experiments=(experiment,)).save(batch_path)
  inventory_path = tmp_path / "inventory.yaml"
  inventory_path.write_text("hosts:\n  - host: 10.0.0.10\n    user: ubuntu\n    slots: 1\n")

  monkeypatch.setattr("remote_experiments.cli.Dispatcher", _FailedDispatcher)
  monkeypatch.setattr("builtins.input", lambda prompt: "all")
  args = build_parser().parse_args([
    "run", str(batch_path), "--inventory", str(inventory_path),
  ])
  cmd_run(args)

  output = capsys.readouterr().out
  assert "batch finished with 1 unsuccessful experiment" in output
  assert "batch complete" not in output
