"""Registry of the decentralized methods (spec: method harmonization).

HARMONIZED: methods whose design is parallel; they solve local problems by DP
on the shared worker pool and report the modelled algorithm time in
runtime.csv. Every runner has the signature
run(config, parallelism, log_on_file=False, disable_plotting=False).
EXCLUDED: methods kept in the code base but not offered to campaigns, with the reason.
"""
from functools import partial

import decentralized_auction
import decentralized_diffusion
import decentralized_dual
import decentralized_frankwolfe
import decentralized_gcaa
import decentralized_powerd
import hierarchical_frankwolfe
import madea2
import run_faasmacro
import run_faasmadea
from hierarchical_auction import madea_cycles_runner, madea_level_cycles_runner, madea_runner, runner
from hierarchical_auction.iterative_engine import IterativeHierarchicalAuctionEngine
from plasma import welfare
from plasma.runner import run as plasma_run

HARMONIZED = {
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

SEQUENTIAL = "nodes act one at a time in a network-wide order (Gauss-Seidel better-response sweeps)"
PG_REFINED = "parallel phase followed by a potential-game refinement that passes a turn through all nodes"
EXCLUDED = {
  "FaaS-MAPG-S": SEQUENTIAL, "FaaS-MAPG-R": SEQUENTIAL,
  "faas-br-s": SEQUENTIAL, "faas-br-r": SEQUENTIAL, "faas-br-o": SEQUENTIAL,
  "MADEA-PG": PG_REFINED, "one-shot-pg": PG_REFINED, "one-shot2-pg": PG_REFINED,
  "one-shot2-b-pg": PG_REFINED, "MAFW-PG": PG_REFINED, "MAFW-b-PG": PG_REFINED,
  "hierarchical-MAFW-PG": PG_REFINED, "hierarchical-madea-pg": PG_REFINED,
  "hierarchical-one-shot-pg": PG_REFINED, "hierarchical-madea-level-cycles-pg": PG_REFINED,
}
