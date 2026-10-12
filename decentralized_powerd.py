from run_faasmacro import (
  parallel_solver_run,
)
from run_faasmadea import (
  VAR_TYPE,
)
from decentralized_diffusion import assignment_step
from utils.centralized import (
  ping_pong_forbidden_hosts
)
from utils.common import load_configuration

from functools import partial
from typing import Tuple
import pandas as pd
import numpy as np
import argparse
from utils import run_frame, step_time


@step_time.parallel
def sample_assignments(
    omega: np.array,
    blackboard: np.array,
    data: dict,
    neighborhood: np.array,
    rho: np.array,
    powerd_options: dict,
    latency: np.array,
    fairness: np.array,
    force_memory_bids: bool,
    rng: np.random.Generator,
    current_y: np.array = None,
    tolerance: float = 1e-6,
  ) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
  """Power-of-d-choices counterpart of decentralized_diffusion.define_assignments.

  For each overloaded (i, f), instead of scanning the whole candidate
  neighbourhood greedily (FaaS-MADiG), repeatedly probe a random sample of ``d``
  candidate sellers (uniform, without replacement) and pick the best by
  ``criterion``: "score" -> max s_{ij}^f = beta - w_lat*L - w_fair*phi;
  "capacity" -> max advertised residual capacity. Ties break on the lower node
  id for reproducibility. The score is always stored in the ``utility`` column,
  on which the reused evaluate_assignments later sorts the seller side.
  """
  try:
    d = int(powerd_options["d"])
  except (TypeError, ValueError) as exc:
    raise ValueError("powerd d must be an integer >= 1") from exc
  if d < 1:
    raise ValueError("powerd d must be >= 1")
  criterion = powerd_options.get("criterion", "score")
  if criterion not in {"score", "capacity"}:
    raise ValueError("powerd criterion must be one of: score, capacity")
  unit_bids = powerd_options.get("unit_bids", False)
  if current_y is None:
    current_y = np.zeros((omega.shape[0], omega.shape[0], omega.shape[1]))
  # no_ping_pong: exclude sender nodes of f from being hosts of f (covers both
  # the capacity and memory seller path, both derived from potential_sellers).
  forbidden = ping_pong_forbidden_hosts(omega, current_y, tolerance)
  potential_buyers, functions_to_share = np.nonzero(omega)
  bids = {"i": [], "j": [], "f": [], "d": [], "utility": []}
  memory_bids = {"i": [], "j": [], "f": []}
  for i, f in step_time.timed(zip(potential_buyers, functions_to_share), key=lambda item: item[0]):
    i = int(i)
    f = int(f)
    potential_sellers = set(np.nonzero(neighborhood[i, :])[0])
    potential_sellers -= {int(j) for j in np.nonzero(forbidden[:, f])[0]}
    potential_capacity_sellers = potential_sellers.intersection(
      set(np.where(blackboard[:, f] >= 1)[0])
    )
    memory_requirement = data[None]["memory_requirement"][f + 1]
    potential_memory_sellers = {
      int(j) for j in potential_sellers if rho[int(j)] >= memory_requirement
    }
    score = {}
    candidates = []
    for j in potential_capacity_sellers:
      j = int(j)
      s = (
        data[None]["beta"][(i + 1, j + 1, f + 1)]
        - powerd_options["latency_weight"] * latency[i, j]
        - powerd_options["fairness_weight"] * fairness[i, f]
      )
      if s > - data[None]["gamma"][(i + 1, f + 1)]:
        score[j] = s
        candidates.append(j)
    candidates.sort()
    # buyer-local view of advertised capacity, decremented as we bid, so we
    # never request more from a seller than the buyer has observed locally
    remaining = {j: blackboard[j, f] for j in candidates}
    assigned = 0
    while assigned < omega[i, f] and len(candidates) > 0:
      sample_size = min(d, len(candidates))
      sample = rng.choice(candidates, size=sample_size, replace=False)
      if criterion == "capacity":
        j_star = int(max(sample, key=lambda k: (remaining[int(k)], -int(k))))
      else:
        j_star = int(max(sample, key=lambda k: (score[int(k)], -int(k))))
      if unit_bids:
        if omega[i, f] - assigned < 1:
          break
        q = 1
      else:
        q = VAR_TYPE(min(remaining[j_star], omega[i, f] - assigned))
      if q <= 0:
        # VAR_TYPE truncates to int: q hits 0 when either j_star is nearly full
        # (drop it and try others) or the residual demand is below one integer
        # unit (nothing more can be offloaded) -- without this the loop spins
        # forever appending zero-quantity bids.
        if remaining[j_star] < 1:
          candidates.remove(j_star)
          continue
        break
      bids["i"].append(i)
      bids["f"].append(f)
      bids["j"].append(j_star)
      bids["d"].append(q)
      bids["utility"].append(score[j_star])
      assigned += q
      remaining[j_star] -= q
      if remaining[j_star] < 1:
        candidates.remove(j_star)
    if assigned < omega[i, f] or force_memory_bids:
      for j in sorted(score, key=lambda k: (-score[k], k)):
        if j in potential_memory_sellers:
          memory_bids["i"].append(i)
          memory_bids["j"].append(int(j))
          memory_bids["f"].append(f)
    if assigned < omega[i, f] or force_memory_bids:
      for j in potential_memory_sellers - potential_capacity_sellers:
        memory_bids["i"].append(i)
        memory_bids["j"].append(int(j))
        memory_bids["f"].append(f)
  return pd.DataFrame(bids), pd.DataFrame(memory_bids), len(potential_buyers)


def parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Run FaaS-MAPoD",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument(
    "-c", "--config", help="Configuration file", type=str,
    default="config_files/manual_config.json",
  )
  parser.add_argument(
    "-j", "--parallelism",
    help="Number of parallel processes (-1: auto, 0: sequential)",
    type=int, default=-1,
  )
  parser.add_argument(
    "--disable_plotting",
    help="True to disable automatic plot generation for each experiment",
    default=False, action="store_true",
  )
  return parser.parse_known_args()[0]


@parallel_solver_run
def run(
    config: dict,
    parallelism: int,
    log_on_file: bool = False,
    disable_plotting: bool = False,
  ) -> str:
  solver_options = config["solver_options"]
  options = dict(solver_options["powerd"])
  options.setdefault("d", 2)
  options.setdefault("criterion", "score")
  options.setdefault("latency_weight", 0.0)
  options.setdefault("fairness_weight", 0.0)
  options.setdefault(
    "unit_bids", solver_options.get("auction", {}).get("unit_bids", False)
  )
  rng = np.random.default_rng(config["seed"])  # one generator for the whole run
  return run_frame.run(
    config, parallelism, log_on_file=log_on_file, disable_plotting=disable_plotting,
    method="FaaS-MAPoD", outputs=run_frame.LSP_OUTPUTS,
    solve_step=partial(assignment_step, options=options,
                       assign=partial(sample_assignments, rng=rng)),
  )


if __name__ == "__main__":
  args = parse_arguments()
  config = load_configuration(args.config)
  run(config, args.parallelism, disable_plotting=args.disable_plotting)
