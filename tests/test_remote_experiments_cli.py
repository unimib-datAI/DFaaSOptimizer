import json

import pytest

from remote_experiments.batch import Batch
from remote_experiments.cli import build_parser, cmd_define


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
