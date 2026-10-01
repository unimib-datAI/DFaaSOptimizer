"""One-shot auction followed by bounded, decentralized PG improvements."""
from copy import deepcopy
import argparse

from decentralized_auction import parse_arguments
from madea_pg import refinement_options
from utils.common import load_configuration


def run(config, parallelism=0, log_on_file=False, disable_plotting=False):
  from decentralized_auction import run as run_one_shot
  config = deepcopy(config)
  options = config.setdefault("solver_options", {}).setdefault("madea_pg", {})
  if "time_limit" not in options and "time_limit_per_node" not in options:
    options["time_limit_per_node"] = .25
  refinement_options(config)
  config["algorithm"] = "one-shot-pg"
  return run_one_shot(config, parallelism, log_on_file, disable_plotting, refine_welfare=True)


def run_hierarchical(config, parallelism=0, log_on_file=False, disable_plotting=False):
  from hierarchical_auction.runner import run as run_hierarchy
  from hierarchical_auction.iterative_engine import IterativeHierarchicalAuctionEngine
  config = deepcopy(config)
  options = config.setdefault("solver_options", {}).setdefault("madea_pg", {})
  if "time_limit" not in options and "time_limit_per_node" not in options:
    options["time_limit_per_node"] = .25
  refinement_options(config)
  config["algorithm"] = "hierarchical-one-shot-pg"
  return run_hierarchy(
    config, parallelism, log_on_file, disable_plotting,
    engine_class=IterativeHierarchicalAuctionEngine, refine_welfare=True,
  )


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("-c", "--config", default="config_files/manual_config.json")
  parser.add_argument("-j", "--parallelism", type=int, default=0)
  parser.add_argument("--disable_plotting", action="store_true")
  parser.add_argument("--variant", choices=("one-shot", "hierarchical"), default="one-shot")
  args = parser.parse_args()
  runner = run if args.variant == "one-shot" else run_hierarchical
  runner(load_configuration(args.config), args.parallelism, disable_plotting=args.disable_plotting)
