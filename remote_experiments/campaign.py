"""Orchestrate the two-phase campaign: screening -> select survivors -> confirm."""

from __future__ import annotations

import json
from pathlib import Path

from .batch import Batch
from .definitions import get_suite
from .definitions.paper import SURVIVORS_PATH
from .instances import materialize_batch
from .manifest import Manifest
from .selection import default_selection
from .survivors import select_survivors

SCREENING_SUITE = "paper-a-screening"
CONFIRMATORY_SUITES = (
  "paper-e1-quality-runtime", "paper-e2-scalability", "paper-e3-topology",
  "paper-e4-robustness", "paper-e5-dynamics", "paper-e6-ablation",
  "paper-e7-tradeoffs", "paper-e8-spatial-latency",
)
STATE_PATH = Path("batches") / "campaign-state.json"


def _load_state() -> dict:
  if STATE_PATH.exists():
    return json.loads(STATE_PATH.read_text())
  return {"stage": "screening", "done_suites": []}


def _save_state(state: dict) -> None:
  STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
  STATE_PATH.write_text(json.dumps(state, indent=2))


def write_survivors(path: Path, survivors: list[str]) -> None:
  Path(path).parent.mkdir(parents=True, exist_ok=True)
  Path(path).write_text(json.dumps({"survivors": survivors}, indent=2))


def _batch_path(suite: str) -> Path:
  return Path("batches") / f"{suite}.json"


def _define_and_materialize(suite: str, instances_root: str) -> Batch:
  batch = Batch(suite=suite, experiments=tuple(get_suite(suite)()))
  path = _batch_path(suite)
  batch.save(path)
  materialize_batch(batch, instances_root)
  return batch


def _run_suite(suite: str, args, state: dict) -> None:
  # imported lazily to avoid a cli<->campaign import cycle
  from .cli import execute_batch
  batch = _define_and_materialize(suite, args.instances)
  manifest_path = _batch_path(suite).with_suffix(".manifest.json")
  manifest = Manifest(manifest_path)
  selected_idx = default_selection([e.id for e in batch.experiments], manifest)
  selected = [batch.experiments[i] for i in selected_idx]
  if selected:
    execute_batch(batch, manifest, manifest_path, selected, args)


def run_campaign(args) -> None:
  state = _load_state()

  if state["stage"] == "screening":
    _run_suite(SCREENING_SUITE, args, state)
    state["stage"] = "select"
    _save_state(state)

  if state["stage"] == "select":
    # rebuild the (deterministic) screening batch rather than reading it back
    # from disk — keeps `select` independent of `_run_suite`'s side effects.
    batch = Batch(suite=SCREENING_SUITE, experiments=tuple(get_suite(SCREENING_SUITE)()))
    survivors = select_survivors(batch, Path(args.results_dir))
    write_survivors(SURVIVORS_PATH, survivors)
    print(f"survivors: {survivors}")
    state["stage"] = "confirm"
    _save_state(state)

  if state["stage"] == "confirm":
    for suite in CONFIRMATORY_SUITES:
      if suite in state["done_suites"]:
        continue
      _run_suite(suite, args, state)
      state["done_suites"].append(suite)
      _save_state(state)
  print("campaign complete")
