"""Welfare per step of every harmonized method on the regression instances, and raw run artifacts.

Run before and after a refactor: decisions must not change.
  uv run python experiments/harmonization/capture_baseline.py tests/data/harmonization/baseline.json
  uv run python experiments/harmonization/capture_baseline.py --artifacts DIR [--method NAME ...] [--solver appsi_highs]
Artifacts are compared with compare_artifacts.py; a capture folder is never overwritten.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from copy import deepcopy
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "madea_pg_compact"))

import pandas as pd  # noqa: E402
from utils import run_log

log = run_log.logger(__name__)

DATA = ROOT / "tests" / "data" / "harmonization"
CONFIG = ROOT / "experiments" / "madea_pg_compact" / "catalog_config_overnight.json"
STEPS, SEED, WORKERS = 2, 1, 0
NONBINDING_S = 1e5          # algorithm and solver budgets: large, finite (configuration validation)
PG_PER_NODE_S = 1e3         # madea_pg.time_limit_per_node: reserve stays below half the budget
CENTRALIZED_LIMIT_S = 3600
# options the centralized solver understands; HiGHS by default: glpk does not finish the regression instances
CENTRALIZED_OPTIONS = {
  "appsi_highs": dict(time_limit=CENTRALIZED_LIMIT_S, threads=1, random_seed=0, mip_rel_gap=1e-6),
  "glpk": dict(TimeLimit=CENTRALIZED_LIMIT_S),  # tmlim, whole seconds
}
DEFAULT_SOLVER = "appsi_highs"
# registry label -> label it shares a callable with (recorded explicitly in the manifest)
ALIASES = {"hierarchical-madea-level-cycles-pg": "hierarchical-madea-pg"}


def runners():
  from methods import HARMONIZED
  return HARMONIZED


def centralized(config, parallelism=0, log_on_file=False, disable_plotting=False):
  from run_centralized_model import run
  return run(config, log_on_file=log_on_file, disable_plotting=disable_plotting)


def baseline_runners() -> dict:
  """Harmonized methods, the excluded ones by their current entry points, and centralized."""
  from methods import SEQUENTIAL_METHODS
  import decentralized_bestresponse as br
  import decentralized_potentialgame as pg
  import hierarchical_frankwolfe as hfw
  import madea2
  import madea_pg
  import one_shot_pg
  excluded = {
    "FaaS-MAPG-S": pg.run_pg_s, "FaaS-MAPG-R": pg.run_pg_r,
    "faas-br-s": br.run_br_s, "faas-br-r": br.run_br_r, "faas-br-o": br.run_br_o,
    "MADEA-PG": madea_pg.run, "one-shot-pg": one_shot_pg.run,
    "one-shot2-pg": madea2.run_one_shot_pg, "one-shot2-b-pg": madea2.run_one_shot_b_pg,
    "MAFW-PG": hfw.run_flat_pg, "MAFW-b-PG": hfw.run_flat_pg_b,
    "hierarchical-MAFW-PG": hfw.run_hierarchical_pg,
    "hierarchical-madea-pg": madea_pg.run_hierarchical,
    "hierarchical-madea-level-cycles-pg": madea_pg.run_hierarchical,
    "hierarchical-one-shot-pg": one_shot_pg.run_hierarchical,
  }
  return {**runners(), **excluded, **SEQUENTIAL_METHODS, "centralized": centralized}


def config_for(instance: Path, out: Path, steps: int = 2) -> dict:
  config = json.loads(CONFIG.read_text())
  full = json.loads((ROOT / "config_files" / "eval_full.json").read_text())["solver_options"]
  config["solver_options"] = {**{k: v for k, v in full.items() if k != "general"}, **config["solver_options"]}
  config["solver_options"]["frankwolfe"] = dict(iterations=400, kappa=10.0, smoothing=1.0, max_rounds=20)
  config["solver_options"]["plasma_welfare"] = dict(W=1., rounds_per_step=20, epsilon=1e-6,
                                                    hb_latency_rounds=1, hb_loss=0., staleness_rounds=3)
  config.update(solver_name="missing_solver", seed=1, max_steps=steps, min_run_time=0,
                max_run_time=steps, run_time_step=1, checkpoint_interval=1,
                base_solution_folder=str(out))
  config["limits"].update(instance_type="materialized", path=str(instance.resolve()))
  return config


def capture_config(instance: Path, out: Path, name: str, solver: str = DEFAULT_SOLVER) -> dict:
  """Deterministic run: fixed seed, bounded iterations/sweeps, time budgets that never bind."""
  config = config_for(instance, out, STEPS)
  config.update(verbose=2, seed=SEED)
  options = config["solver_options"]
  options["general"].update(TimeLimit=NONBINDING_S, algorithm_time_limit=NONBINDING_S)
  options["madea_pg"]["time_limit_per_node"] = PG_PER_NODE_S
  if name == "centralized":  # a real solve: only the options the solver understands
    config["solver_name"] = solver
    options["general"] = dict(CENTRALIZED_OPTIONS.get(solver, CENTRALIZED_OPTIONS["glpk"]))
  return config


def welfare(runner, instance: Path, parallelism: int = 0) -> list[float]:
  with tempfile.TemporaryDirectory() as tmp:
    folder = Path(runner(config_for(instance, Path(tmp)), parallelism, log_on_file=True, disable_plotting=True))
    return [float(v) for v in pd.read_csv(folder / "obj.csv").iloc[:, -1]]


def instances() -> list[Path]:
  return sorted(p for p in DATA.iterdir() if (p / "metadata.json").exists())


def _callable_name(fn) -> str:
  if isinstance(fn, partial):
    return f"partial({_callable_name(fn.func)}, {fn.keywords})"
  return f"{fn.__module__}.{fn.__qualname__}"


def _revision() -> dict:
  git = lambda *args: subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True).stdout.strip()
  return dict(commit=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain", "--untracked-files=no")))


def capture_artifacts(destination: Path, names: list[str] | None = None, solver: str = DEFAULT_SOLVER) -> None:
  """Run each method on every regression instance; keep its run folder under
  <destination>/<method>/<instance>/ and record coverage in <destination>/manifest.json."""
  destination = Path(destination)
  available = baseline_runners()
  names = list(names or available)
  unknown = sorted(set(names) - set(available))
  if unknown:
    raise ValueError(f"unknown methods {unknown}; available: {sorted(available)}")
  taken = [n for n in names if (destination / n).exists()]
  if taken:
    raise FileExistsError(f"{destination}: already captured {taken} (captures are immutable)")
  path = destination / "manifest.json"
  manifest = (json.loads(path.read_text()) if path.exists() else
              dict(solver=solver, seed=SEED, workers=WORKERS, steps=STEPS,
                   instances=[i.name for i in instances()], methods={}))
  for name in names:
    entry = dict(callable=_callable_name(available[name]), alias_of=ALIASES.get(name), solver=solver,
                 revision=_revision(), status="ok", configs={})
    try:
      for instance in instances():
        config = capture_config(instance, destination / name / instance.name, name, solver)
        entry["configs"][instance.name] = deepcopy(config)
        available[name](config, WORKERS, log_on_file=True, disable_plotting=True)
    except Exception as error:  # incomplete coverage, never success
      kind = "unavailable" if name == "centralized" else "failed"
      entry["status"] = f"{kind}: {type(error).__name__}: {error}"
    manifest["methods"][name] = entry
    destination.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1, default=str))
    log.info('%s: %s', name, entry['status'])


def main(output: Path) -> None:
  result = {name: {i.name: welfare(run, i) for i in instances()} for name, run in runners().items()}
  output.write_text(json.dumps(result, indent=1))
  log.info("%s", json.dumps(result, indent=1))


if __name__ == "__main__":
  with run_log.open_run_log(None, 1):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", nargs="?", type=Path, help="welfare JSON of the harmonized methods")
    parser.add_argument("--artifacts", type=Path, help="capture raw run folders here instead")
    parser.add_argument("--method", action="append", help="capture only these methods (default: all)")
    parser.add_argument("--solver", default=DEFAULT_SOLVER, help="solver of the centralized model")
    args = parser.parse_args()
    if args.artifacts is None and args.output is None:
      parser.error("give a welfare JSON path or --artifacts DIR")
    if args.artifacts is None:
      main(args.output)
    else:
      capture_artifacts(args.artifacts, args.method, args.solver)
      statuses = json.loads((args.artifacts / "manifest.json").read_text())["methods"]
      sys.exit(0 if all(statuses[n]["status"] == "ok" for n in (args.method or statuses)) else 1)
