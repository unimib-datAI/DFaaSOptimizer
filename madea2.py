"""MADEA2 and one-shot2: MADEA / one-shot with replica decisions imported from FaaS-MAFW.

The auction loops, messages and stopping rules are those of MADEA
(``run_faasmadea``) and of one-shot (``decentralized_auction``), which are not
modified.  Three local rules change, with the same number of messages:

* memory bids carry two more numbers, the uncovered quantity ``d`` and the
  buyer's advantage over the Cloud ``v`` (beta + gamma minus the latency and
  fairness terms);
* a seller starts new replicas greedily by requested value per MB
  (``start_replicas_by_value``) instead of splitting its free memory in
  proportion to the number of memory bids per function;
* when replicas are started inside ``evaluate_bids`` (MADEA only), pending
  bids are served by value per MB of a new replica instead of by bid alone.

Optionally (``solver_options.auction.congestion_weight > 0``) a seller also
raises its price in proportion to the demand it was offered but could not
serve, relative to its capacity; this uses only bids it already received.

``run`` and ``run_one_shot`` execute the unmodified MADEA / one-shot runners
with these three rules passed explicitly (``rules=_RULES``).
"""
from copy import deepcopy
from typing import Tuple

import numpy as np
import pandas as pd

import decentralized_auction as _one_shot
import run_faasmadea as _madea
# the shared bodies, bound once: replacing MADEA's rules never changes MADEA2's
from run_faasmadea import define_bids as _madea_bids, evaluate_bids as _madea_evaluation
from utils import step_time


def define_bids(
    omega: np.array,
    blackboard: np.array,
    p: np.array,
    data: dict,
    neighborhood: np.array,
    rho: np.array,
    auction_options: dict,
    latency: np.array,
    fairness: np.array,
    force_memory_bids: bool
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """MADEA's bids; memory bids also carry the uncovered quantity ("d") and the
  buyer's advantage over the Cloud ("v"): two more numbers in the same message,
  used only by the value-per-MB replica allocation (start_additional_replicas)."""
  # "beta" (default): the memory bid reveals the buyer's private advantage
  # beta + gamma.  "bid": it carries only an auction bid built like the
  # capacity bids, b = p_j + epsilon + (u_j - u_next), so the seller learns
  # no more than in MADEA (see docs/decentralized-algorithms/information-model.md)
  value_mode = auction_options.get("memory_bid_value", "beta")
  if value_mode not in {"beta", "bid"}:
    raise ValueError("memory_bid_value must be 'beta' or 'bid'")
  bid_cache = {}

  def private_value(i, j, f):
    return (
      data[None]["beta"][(i+1,j+1,f+1)] + data[None]["gamma"][(i+1,f+1)]
      - auction_options["latency_weight"] * latency[i,j]
      - auction_options["fairness_weight"] * fairness[i,f]
    )

  def memory_bid_value(i, j, f):
    if value_mode == "beta":
      return private_value(i, j, f)
    if (i, f) not in bid_cache:
      sellers = sorted(set(np.nonzero(neighborhood[i,:])[0]) & set(np.nonzero(rho)[0]))
      utility = {k: private_value(i, k, f) - p[k,f] for k in sellers}
      order = sorted(sellers, key=lambda k: -utility[k])
      bid_cache[(i, f)] = {
        k: p[k,f] + auction_options["epsilon"]
           + (utility[k] - utility[order[idx+1]] if idx + 1 < len(order) else 0.0)
        for idx, k in enumerate(order)
      }
    return bid_cache[(i, f)].get(j, p[j,f] + auction_options["epsilon"])

  return _madea_bids(
    omega, blackboard, p, data, neighborhood, rho, auction_options, latency, fairness,
    force_memory_bids, memory_bid_value=memory_bid_value,
  )


def evaluate_bids(
    bids: pd.DataFrame,
    blackboard: np.array,
    data: dict,
    previous_y: np.array = None,
    ell: np.array = None,
    p: np.array = None,
    total_capacity: np.array = None,
    u0: np.array = None,
    auction_options: dict = None,
    initial_rho: np.array = None,
    r: np.array = None,
    tentatively_start_replicas: bool = False,
    it: int = 0,
    may_replace_existing_assignments: bool = True,
    residual_capacity: np.array = None
  ) -> np.array:
  """MADEA's evaluation with replicas started by value per MB and, optionally,
  congestion pricing (``auction.congestion_weight``)."""
  options = auction_options if auction_options is not None else {}
  return _madea_evaluation(
    bids, blackboard, data, previous_y, ell, p, total_capacity, u0, auction_options,
    initial_rho, r, tentatively_start_replicas, it, may_replace_existing_assignments,
    residual_capacity,
    replica_value_per_mb=options.get("replica_allocation", "value_per_mb") == "value_per_mb",
    congestion_weight=float(options.get("congestion_weight", 0.0)),
  )


@step_time.parallel
def start_replicas_by_value(
    memory_bids: pd.DataFrame,
    r: np.array,
    data: dict,
    rho: np.array
  ) -> Tuple[np.array, np.array]:
  """Greedy replica start by requested value per MB (seller-local).

  For each seller j, every memory bid (i, f) asks for ``d`` req/s worth
  ``v`` per req/s (advantage over the Cloud).  The seller repeatedly starts
  the smallest productive replica batch whose served requests (taken from
  the highest-value bids still uncovered) bring the largest value per MB,
  pricing the whole batch and its usable capacity, until the free memory or
  the requested demand runs out.  Bids from different buyers are not
  coordinated, so the same uncovered demand may be requested from several
  sellers; LSPr releases replicas that stay idle.
  """
  additional_replicas = np.zeros(r.shape)
  residual_memory = deepcopy(rho)
  if len(memory_bids) == 0:
    return additional_replicas, residual_memory
  for j, bids_for_j in step_time.timed(memory_bids.groupby("j"), key=lambda item: item[0]):
    free = float(rho[j])
    if free <= 0:
      continue
    queues = {}
    for f, group in bids_for_j.groupby("f"):
      requests = sorted(
        ((float(v), float(d)) for v, d in zip(group["v"], group["d"]) if d > 0 and v > 0),
        reverse=True,
      )
      if requests:
        queues[int(f)] = [list(item) for item in requests]
    while queues:
      best = None
      for f, requests in queues.items():
        ram_f = data[None]["memory_requirement"][f+1]
        capacity = data[None]["max_utilization"][f+1] / data[None]["demand"][(j+1,f+1)]
        count, capacity = _madea.next_replica_batch(r[j,f] + additional_replicas[j,f], capacity)
        memory = count * ram_f
        if memory > free:
          continue
        value, left = 0.0, capacity
        for v, d in requests:
          take = min(d, left)
          value += v * take
          left -= take
          if left <= 0:
            break
        if value > 0 and (best is None or value / memory > best[0]):
          best = (value / memory, f, capacity, count, memory)
      if best is None:
        break
      _, f, capacity, count, memory = best
      additional_replicas[j, f] += count
      free -= memory
      residual_memory[j] -= memory
      left = capacity
      requests = queues[f]
      while requests and left > 0:
        take = min(requests[0][1], left)
        requests[0][1] -= take
        left -= take
        if requests[0][1] <= 1e-9:
          requests.pop(0)
      if not requests:
        del queues[f]
  return additional_replicas, residual_memory


def start_additional_replicas(memory_bids, r, data, rho, *args, **kwargs):
  """Drop-in replacement used by the MADEA2 / one-shot2 runners."""
  return start_replicas_by_value(memory_bids, r, data, rho)


_RULES = {
  "define_bids": define_bids,
  "evaluate_bids": evaluate_bids,
  "start_additional_replicas": start_additional_replicas,
}


def run(config, parallelism, log_on_file=False, disable_plotting=False):
  """MADEA2: the MADEA runner with the value-per-MB replica rules."""
  return _madea.run(config, parallelism, log_on_file, disable_plotting, rules=_RULES)


def run_one_shot(config, parallelism, log_on_file=False, disable_plotting=False,
                 *, refine_welfare=False):
  """One-shot2 (and one-shot2-PG): the one-shot runner with the same rules."""
  return _one_shot.run(config, parallelism, log_on_file, disable_plotting,
                       refine_welfare=refine_welfare, rules=_RULES)


def run_one_shot_pg(config, parallelism, log_on_file=False, disable_plotting=False):
  return run_one_shot(config, parallelism, log_on_file, disable_plotting, refine_welfare=True)


def _bid_only(config):
  """Same configuration with memory bids that carry the auction bid, not beta."""
  config = deepcopy(config)
  config.setdefault("solver_options", {}).setdefault("auction", {})["memory_bid_value"] = "bid"
  return config


def run_b(config, parallelism, log_on_file=False, disable_plotting=False):
  """MADEA2-b: MADEA2 without revealing beta to the sellers."""
  return run(_bid_only(config), parallelism, log_on_file, disable_plotting)


def run_one_shot_b(config, parallelism, log_on_file=False, disable_plotting=False):
  return run_one_shot(_bid_only(config), parallelism, log_on_file, disable_plotting)


def run_one_shot_b_pg(config, parallelism, log_on_file=False, disable_plotting=False):
  return run_one_shot(_bid_only(config), parallelism, log_on_file, disable_plotting, refine_welfare=True)
