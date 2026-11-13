"""Complete MADEA cycles followed by converged auctions at each hierarchy level."""

from typing import Any

from hierarchical_auction.iterative_engine import IterativeHierarchicalAuctionEngine
from hierarchical_auction.madea_cycles_runner import _run
from hierarchical_auction.madea_runner import parse_arguments
from utils.common import load_configuration


def run(
    config: dict[str, Any],
    parallelism: int = -1,
    log_on_file: bool = False,
    disable_plotting: bool = False,
    *, detailed_init: bool = False,
  ) -> str:
  return _run(
    config, parallelism, log_on_file, disable_plotting,
    engine_class=IterativeHierarchicalAuctionEngine,
    result_name="hierarchical-madea-detailed" if detailed_init else "HierarchicalMADeALevelCycles",
    detailed_init=detailed_init,
  )


if __name__ == "__main__":
  args = parse_arguments()
  run(
    load_configuration(args.config), parallelism=args.parallelism,
    disable_plotting=args.disable_plotting,
  )
