from __future__ import annotations

import argparse

from utils.common import load_configuration

from plasma.runner import run


def parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Run PLASMA standalone (debugging entry point)",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument("-c", "--config", type=str, default="manual_config.json")
  parser.add_argument("-j", "--parallelism", type=int, default=0)
  parser.add_argument("--disable_plotting", default=False,
                      action="store_true")
  return parser.parse_known_args()[0]


if __name__ == "__main__":
  args = parse_arguments()
  run(
    load_configuration(args.config), args.parallelism,
    disable_plotting=args.disable_plotting,
  )
