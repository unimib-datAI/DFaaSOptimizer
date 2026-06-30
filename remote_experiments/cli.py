"""CLI entry point: `define` builds a batch file from a suite, `run` executes/resumes it."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from ray_dispatcher import Dispatcher, Inventory, Project, SecretFile

from . import definitions  # imports register all suites as a side effect
from .batch import Batch
from .definitions import get_suite, list_suites
from .jobs import experiment_to_job
from .manifest import Manifest
from .runner import run_batch
from .selection import default_selection, parse_selection
from .tui import live_view


def cmd_define(args: argparse.Namespace) -> None:
  build = get_suite(args.suite)
  experiments = build()
  batch = Batch(suite=args.suite, experiments=tuple(experiments))
  out_path = Path(args.output)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  batch.save(out_path)
  print(f"wrote {len(experiments)} experiments to {out_path}")


def cmd_run(args: argparse.Namespace) -> None:
  batch = Batch.load(args.batch_file)
  manifest_path = Path(args.batch_file).with_suffix(".manifest.json")
  manifest = Manifest(manifest_path)
  experiment_ids = [e.id for e in batch.experiments]

  default_idx = default_selection(experiment_ids, manifest)
  print(f"{len(batch.experiments)} experiments in batch, {len(default_idx)} pending")
  for i, e in enumerate(batch.experiments):
    print(f"  [{i}] {e.id} ({manifest.status(e.id)})")
  raw = input(f"Select to run [default: {len(default_idx)} pending] (indices/ranges/'all'): ")
  selected_idx = parse_selection(raw, len(batch.experiments)) if raw.strip() else default_idx
  selected = [batch.experiments[i] for i in selected_idx]
  if not selected:
    print("nothing selected, exiting")
    return

  inventory = Inventory.from_yaml(args.inventory)
  secrets = ()
  if args.gurobi_license:
    secrets = (
      SecretFile(source=args.gurobi_license, remote_name="gurobi.lic", env_var="GRB_LICENSE_FILE"),
    )
  project = Project(
    path=str(Path(args.project_path).resolve()),
    project_id="dfaas-optimizer",
    python=args.python_version,
    uv_version=args.uv_version,
    secrets=secrets,
    exclude=(".venv/", ".git/", "solutions/", "results/", "batches/"),
  )
  config_dir = manifest_path.parent / f"{manifest_path.stem}-configs"
  jobs = [experiment_to_job(e, config_dir) for e in selected]

  with Dispatcher(inventory, project, results_dir=args.results_dir) as dispatcher:
    start = time.monotonic()
    with live_view(batch, manifest, inventory, start_time=start) as on_tick:
      completed = run_batch(dispatcher, jobs, manifest, on_tick)
  print("batch complete" if completed else "stopped — rerun the same command to resume")


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(prog="remote_experiments")
  sub = parser.add_subparsers(dest="command", required=True)

  define_p = sub.add_parser("define", help="Build a batch file from a registered suite")
  define_p.add_argument("suite", choices=list_suites())
  define_p.add_argument("-o", "--output", required=True)
  define_p.set_defaults(func=cmd_define)

  run_p = sub.add_parser("run", help="Run (or resume) a batch file through the TUI")
  run_p.add_argument("batch_file")
  run_p.add_argument("--inventory", required=True)
  run_p.add_argument("--project-path", default=".")
  run_p.add_argument("--results-dir", default="./results")
  run_p.add_argument("--gurobi-license", default=None)
  run_p.add_argument("--python-version", default="3.10.19")
  run_p.add_argument("--uv-version", default="0.11.25")
  run_p.set_defaults(func=cmd_run)

  return parser


def main(argv: list[str] | None = None) -> None:
  parser = build_parser()
  args = parser.parse_args(argv)
  args.func(args)
