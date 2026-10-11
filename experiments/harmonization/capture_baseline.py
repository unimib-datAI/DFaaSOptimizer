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
  try:
    from methods import HARMONIZED
    return HARMONIZED
  except ImportError:  # before Task 7: the same names, resolved by hand
    from functools import partial
    import decentralized_auction, decentralized_diffusion, decentralized_dual, decentralized_frankwolfe
    import decentralized_gcaa, decentralized_powerd, hierarchical_frankwolfe, madea2, run_faasmacro, run_faasmadea
    from hierarchical_auction import madea_cycles_runner, madea_level_cycles_runner, madea_runner, runner
    from hierarchical_auction.iterative_engine import IterativeHierarchicalAuctionEngine
    from plasma import welfare
    from plasma.runner import run as plasma_run
    return {
      "MADEA": run_faasmadea.run, "MADEA-one-shot": decentralized_auction.run,
      "MADEA2": madea2.run, "MADEA2-b": madea2.run_b,
      "MADEA2-one-shot": madea2.run_one_shot, "MADEA2-b-one-shot": madea2.run_one_shot_b,
      "FaaS-MAFW": decentralized_frankwolfe.run, "FaaS-MAFW-b": hierarchical_frankwolfe.run_mafw_b,
      "hierarchical-MAFW": partial(hierarchical_frankwolfe.run, hierarchy=True, refine_welfare=False),
      "hierarchical-madea": madea_level_cycles_runner.run,
      "hierarchical-madea-cycles": madea_cycles_runner.run,
      "hierarchical-madea-legacy": madea_runner.run,
      "hierarchical-one-shot": partial(runner.run, engine_class=IterativeHierarchicalAuctionEngine),
      "hierarchical": runner.run,
      "PLASMA": plasma_run, "PLASMA-Welfare": welfare.run,
      "faas-macro": run_faasmacro.run, "faas-macro-v0": partial(run_faasmacro.run, v0=True),
      "faas-diffuse": decentralized_diffusion.run, "faas-powd": decentralized_powerd.run,
      "faas-gcaa": decentralized_gcaa.run, "faas-dual": decentralized_dual.run,
    }


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
