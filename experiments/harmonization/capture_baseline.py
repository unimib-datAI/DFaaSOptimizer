"""Welfare per step of every harmonized method on the regression instances.

Run before and after the harmonization: decisions must not change.
  uv run python experiments/harmonization/capture_baseline.py tests/data/harmonization/baseline.json
"""
import json
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "madea_pg_compact"))

import pandas as pd  # noqa: E402

DATA = ROOT / "tests" / "data" / "harmonization"
CONFIG = ROOT / "experiments" / "madea_pg_compact" / "catalog_config_overnight.json"


def runners():
  from methods import HARMONIZED
  return HARMONIZED


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


def welfare(runner, instance: Path, parallelism: int = 0) -> list[float]:
  with tempfile.TemporaryDirectory() as tmp:
    folder = Path(runner(config_for(instance, Path(tmp)), parallelism, log_on_file=True, disable_plotting=True))
    return [float(v) for v in pd.read_csv(folder / "obj.csv").iloc[:, -1]]


def main(output: Path) -> None:
  instances = sorted(p for p in DATA.iterdir() if (p / "metadata.json").exists())
  result = {name: {i.name: welfare(run, i) for i in instances} for name, run in runners().items()}
  output.write_text(json.dumps(result, indent=1))
  print(json.dumps(result, indent=1))


if __name__ == "__main__":
  main(Path(sys.argv[1]))
