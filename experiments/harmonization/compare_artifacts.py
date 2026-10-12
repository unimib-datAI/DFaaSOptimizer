"""Compare two artifact captures of capture_baseline.py: decisions must not change.

  uv run python experiments/harmonization/compare_artifacts.py BEFORE AFTER [--method NAME ...]

Solutions, checkpoints, objectives and materialized inputs must be byte-identical. Only timing
values are normalized: runtime.csv is compared by contract, termination reasons without their
durations, diagnostic CSVs without their timing columns, MACrO/MADEA logs through their parsers
without parsed times, config.json without its output root. Exit code 1 on any difference or
missing coverage. The captures are never modified.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils import run_log  # noqa: E402

log = run_log.logger(__name__)

COLUMNS = ("step", "algorithm_s", "wall_s", "parallel_phases", "workers")  # utils.step_time.COLUMNS
STRUCTURAL = ("step", "parallel_phases", "workers")
TIMING_COLUMNS = {"refinement.csv": {"seconds"}}  # diagnostic file -> its timing columns
NUMBER = r"[-+]?(?:inf|nan|\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)"
TIMING_TEXT = (  # termination reasons: only the durations
  (re.compile(rf"(total runtime[:=] ){NUMBER}"), r"\1<T>"),
  (re.compile(rf"(reached time limit: ){NUMBER} (>=|>) {NUMBER}"), r"\1<T> \2 <T>"),
  (re.compile(rf"(reached time limit \(){NUMBER}\)"), r"\1<T>)"),
)
OUTPUT_ROOT = "base_solution_folder"


def _rows(path: Path) -> list[list[str]]:
  with open(path, newline="") as f:
    return list(csv.reader(f))


def _runtime(before: Path, after: Path) -> str | None:
  rb, ra = _rows(before), _rows(after)
  if not rb or not ra:
    return None if rb == ra else "empty on one side"
  if ra[0] != rb[0] and tuple(ra[0]) != COLUMNS:
    return f"columns {ra[0]}, expected {rb[0]} or {list(COLUMNS)}"
  if len(ra) != len(rb):
    return f"{len(ra) - 1} rows, expected {len(rb) - 1}"
  if tuple(ra[0]) == tuple(rb[0]) == COLUMNS:
    index = [COLUMNS.index(c) for c in STRUCTURAL]
    for b, a in zip(rb[1:], ra[1:]):
      if [b[i] for i in index] != [a[i] for i in index]:
        return f"step/phases/workers {[a[i] for i in index]}, expected {[b[i] for i in index]}"
  return None


def normalize_termination(text: str) -> str:
  for pattern, replacement in TIMING_TEXT:
    text = pattern.sub(replacement, text)
  return text


def _without_columns(path: Path, columns: set) -> list[list[str]]:
  rows = _rows(path)
  keep = [i for i, c in enumerate(rows[0] if rows else []) if c not in columns]
  return [[row[i] for i in keep if i < len(row)] for row in rows]


def _config(path: Path) -> dict:
  config = json.loads(path.read_text())
  config.pop(OUTPUT_ROOT, None)
  return config


def _parsed_log(folder: Path, label: str):
  import pandas as pd
  from logs_postprocessing import parse_log_file
  data = json.loads((folder / "base_instance_data.json").read_text())
  logs, best = parse_log_file(str(folder), "run", pd.DataFrame(), {}, int(data["None"]["Nn"]["None"]), label)
  timing = [c for c in logs.columns if "runtime" in c or "wallclock" in c or c == "measured_total_time"]
  return logs.drop(columns=timing), best


def _log(before: Path, after: Path, label: str) -> str | None:
  """MACrO/MADEA logs through logs_postprocessing (what the analyses read); others are not parsed."""
  if not any(k in label.lower() for k in ("macro", "madea")):
    return None
  parsed = []
  for folder in (before.parent, after.parent):
    try:
      parsed.append(_parsed_log(folder, label))
    except Exception as error:  # an unparsable log must stay unparsable the same way
      parsed.append(type(error).__name__)
  (lb, bb), (la, ba) = [p if isinstance(p, tuple) else (p, None) for p in parsed]
  if isinstance(lb, str) or isinstance(la, str):
    return None if lb == la else f"parser: {la if isinstance(la, str) else 'ok'}, expected {lb if isinstance(lb, str) else 'ok'}"
  if not lb.equals(la):
    return f"parsed log differs ({len(la)} rows, expected {len(lb)})"
  if set(bb) != set(ba) or any(not bb[k].equals(ba[k]) for k in bb):
    return "parsed best solutions differ"
  return None


def compare_file(before: Path, after: Path, label: str) -> str | None:
  name = before.name
  if before.suffix == ".png":  # inventory only: plotting smoke tests check the images
    return None
  if name == "runtime.csv":
    return _runtime(before, after)
  if name == "termination_condition.csv":
    same = normalize_termination(before.read_text()) == normalize_termination(after.read_text())
    return None if same else "termination differs beyond timing"
  if name == "config.json":
    return None if _config(before) == _config(after) else "configuration differs"
  if name == "out.log":
    return _log(before, after, label)
  if name in TIMING_COLUMNS:
    same = _without_columns(before, TIMING_COLUMNS[name]) == _without_columns(after, TIMING_COLUMNS[name])
    return None if same else "differs beyond timing columns"
  return None if before.read_bytes() == after.read_bytes() else "differs"


def _inventory(folder: Path) -> set[str]:
  return {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}


def compare_run(before: Path, after: Path, name: str) -> list[str]:
  """Mismatches between two run folders (the timestamped folder a runner returns)."""
  fb, fa = _inventory(before), _inventory(after)
  problems = [f"{name}: missing {f}" for f in sorted(fb - fa)]
  problems += [f"{name}: new {f}" for f in sorted(fa - fb)]
  obj = before / "obj.csv"
  label = obj.read_text().splitlines()[0] if obj.exists() else ""
  for f in sorted(fb & fa):
    problem = compare_file(before / f, after / f, label)
    if problem:
      problems.append(f"{name}: {f}: {problem}")
  return problems


def _run_folder(folder: Path) -> Path | None:
  runs = [p for p in folder.iterdir() if p.is_dir()] if folder.is_dir() else []
  return runs[0] if len(runs) == 1 else None


def compare_artifacts(before: Path, after: Path, names: list[str] | None = None) -> list[str]:
  before, after = Path(before), Path(after)
  mb, ma = (json.loads((p / "manifest.json").read_text()) for p in (before, after))
  problems = []
  for name in names or sorted(mb["methods"]):
    statuses = [m["methods"].get(name, {}).get("status", "not captured") for m in (mb, ma)]
    if statuses != ["ok", "ok"]:
      problems.append(f"{name}: coverage: before {statuses[0]}, after {statuses[1]}")
      continue
    for instance in mb["instances"]:
      rb, ra = (_run_folder(p / name / instance) for p in (before, after))
      if rb is None or ra is None:
        problems.append(f"{name}/{instance}: coverage: run folder missing or ambiguous")
        continue
      problems += compare_run(rb, ra, f"{name}/{instance}")
  return problems


def main(argv=None) -> int:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("before", type=Path)
  parser.add_argument("after", type=Path)
  parser.add_argument("--method", action="append", help="compare only these methods (default: all captured before)")
  args = parser.parse_args(argv)
  problems = compare_artifacts(args.before, args.after, args.method)
  log.info("%s", "\n".join(problems) if problems else "artifacts match")
  return 1 if problems else 0


if __name__ == "__main__":
  with run_log.open_run_log(None, 1):
    sys.exit(main())
