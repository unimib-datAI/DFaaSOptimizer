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
with these three functions swapped in for the duration of the call.
"""
from contextlib import contextmanager
from copy import deepcopy
from typing import Tuple

import numpy as np
import pandas as pd

import decentralized_auction as _one_shot
import run_faasmadea as _madea
from run_faasmadea import VAR_TYPE, ensure_memory_sellers


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
  use_internal_value = auction_options.get("use_internal_value", False)
  # loop over agents and functions
  potential_buyers, functions_to_share = np.nonzero(omega)
  bids = {
    "i": [], "j": [], "f": [], "d": [], "b": [], "utility": []
  }
  # memory bids also carry the uncovered quantity ("d") and the buyer's
  # advantage over the Cloud ("v"): two more numbers in the same message,
  # used only by the value-per-MB replica allocation (see
  # start_additional_replicas)
  memory_bids = {
    "i": [], "j": [], "f": [], "d": [], "v": []
  }

  def memory_bid_value(i, j, f):
    return (
      data[None]["beta"][(i+1,j+1,f+1)] + data[None]["gamma"][(i+1,f+1)]
      - auction_options["latency_weight"] * latency[i,j]
      - auction_options["fairness_weight"] * fairness[i,f]
    )

  for i,f in zip(potential_buyers, functions_to_share):
    # identify potential sellers
    potential_sellers = set(np.nonzero(neighborhood[i,:])[0])
    # -- capacity sellers are neighbors with residual computing capacity 
    # for function f
    potential_capacity_sellers = potential_sellers.intersection(
      set(np.where(blackboard[:,f]>=1)[0])
    )
    # -- memory sellers are neighbors with residual memory capacity to 
    # instantiate new replicas
    potential_memory_sellers = potential_sellers.intersection(
      set(np.nonzero(rho)[0])
    )
    # -- loop over potential sellers
    utility = []
    candidate_sellers = []
    candidate_values = []
    for j in potential_capacity_sellers:
      # -- compute utility
      value = data[None]["beta"][(i+1,j+1,f+1)]
      if use_internal_value:
        # Match the welfare normalization, including its zero-load convention.
        load = data[None]["incoming_load"][(i+1,f+1)] or 1
        value = (value + data[None]["gamma"][(i+1,f+1)]) / load
      ut = (  
        value -
        p[j,f] - 
        auction_options["latency_weight"] * latency[i,j] - 
        auction_options["fairness_weight"] * fairness[i,f]
      )
      reservation_utility = 0.0 if use_internal_value else -data[None]["gamma"][(i+1,f+1)]
      if ut > reservation_utility:
        utility.append(ut)
        candidate_sellers.append(j)
        candidate_values.append(value)
    # compute weights and define bids
    assigned = 0
    if len(utility) > 0:
      utility = np.array(utility)
      sellers_order = np.argsort(utility)[::-1]
      idx = 0
      while idx < len(sellers_order) and assigned < omega[i,f]:
        j = candidate_sellers[sellers_order[idx]]
        delta = 0.0
        if idx < len(sellers_order) - 1:
          delta = utility[sellers_order[idx]] - utility[sellers_order[idx+1]]
        b = p[j,f] + auction_options["epsilon"] + delta
        if use_internal_value:
          b = min(b, candidate_values[sellers_order[idx]])
        if auction_options["unit_bids"]:
          d = 1
          while (d < int(min(blackboard[j,f], omega[i,f])) + 1) and (
              assigned < omega[i,f]
            ):
            bids["i"].append(i)
            bids["f"].append(f)
            bids["j"].append(j)
            bids["d"].append(1)
            bids["b"].append(b)
            bids["utility"].append(utility[sellers_order[idx]])
            assigned += 1
            d += 1
        else:
          d = VAR_TYPE(min(blackboard[j,f], (omega[i,f] - assigned)))
          bids["i"].append(i)
          bids["f"].append(f)
          bids["j"].append(j)
          bids["d"].append(d)
          bids["b"].append(b)
          bids["utility"].append(utility[sellers_order[idx]])
          assigned += d
        idx += 1
      # -- if you could not bid for everything, ask also for new replicas
      if assigned < omega[i,f]:
        for idx in sellers_order:
          j = candidate_sellers[idx]
          if j in potential_memory_sellers:
            memory_bids["i"].append(i)
            memory_bids["j"].append(j)
            memory_bids["f"].append(f)
            memory_bids["d"].append(float(omega[i,f] - assigned))
            memory_bids["v"].append(memory_bid_value(i, j, f))
    # if you could not bid for all, ask for new replicas
    if assigned < omega[i,f] or force_memory_bids:
      for j in potential_memory_sellers - potential_capacity_sellers:
        memory_bids["i"].append(i)
        memory_bids["j"].append(j)
        memory_bids["f"].append(f)
        memory_bids["d"].append(float(max(omega[i,f] - assigned, 0.0)))
        memory_bids["v"].append(memory_bid_value(i, j, f))
  return pd.DataFrame(bids), pd.DataFrame(memory_bids), len(potential_buyers)


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
  Nn = data[None]["Nn"][None]
  Nf = data[None]["Nf"][None]
  last_y = np.zeros((Nn,Nn,Nf)) if previous_y is None else previous_y.copy()
  if ell is None:
    ell = np.zeros((Nn,Nf))
  if p is None:
    p = np.zeros((Nn,Nf))
  if total_capacity is None:
    total_capacity = np.ones((Nn,Nf))
  if u0 is None:
    u0 = np.zeros((Nn,Nf))
  if auction_options is None:
    auction_options = {"eta": 0.0, "zeta": 0.0}
  if initial_rho is None:
    initial_rho = np.zeros((Nn,))
  if r is None:
    r = np.zeros((Nn,Nf))
  # eta may be a per-iteration schedule (e.g. [0.5, 0.3, 0.15]); clamp to the
  # last value once the schedule is exhausted, accept a plain scalar too
  eta = auction_options["eta"]
  eta = eta[min(it, len(eta) - 1)] if isinstance(eta, (list, tuple)) else eta
  value_per_mb = auction_options.get("replica_allocation", "value_per_mb") == "value_per_mb"
  congestion_weight = float(auction_options.get("congestion_weight", 0.0))

  def replica_capacity(j, f):
    return data[None]["max_utilization"][f+1] / data[None]["demand"][(j+1,f+1)]
  # no_ping_pong (validate_centralized_solution): a node must not both send 
  # and receive the same function across the accumulated y. Track realized 
  # roles and refuse any bid that would give a node a second, conflicting 
  # role. Mirrors the guard in decentralized_auction.evaluate_bids.
  sending = last_y.sum(axis = 1) > 1e-10
  receiving = last_y.sum(axis = 0) > 1e-10
  # loop over agents and functions
  potential_sellers, functions_to_share = [None, None]
  potential_sellers, functions_to_share = np.nonzero(blackboard)
  if tentatively_start_replicas:
    potential_sellers, functions_to_share = ensure_memory_sellers(
      potential_sellers,
      functions_to_share,
      np.nonzero(initial_rho)[0],
      Nf
    )
  y = np.zeros((Nn,Nn,Nf))
  additional_replicas = np.zeros((Nn,Nf))
  rho = deepcopy(initial_rho)
  unique_potential_sellers = np.unique(potential_sellers)
  for j in unique_potential_sellers:
    # extract bids for the current node
    all_bids_for_j = bids[(bids["j"] == j)].sort_values(
      by = "b", ascending = False
    )
    bid_rows = all_bids_for_j.to_records(index=False)
    remaining_capacities = None
    if may_replace_existing_assignments:
      remaining_capacities = residual_capacity[j,:].astype(int).copy()
    else:
      remaining_capacities = blackboard[j,:].astype(int).copy()
    all_min_b = all_bids_for_j.groupby("f")["b"].max()
    pending_bids = []
    # Fill free capacity first; keep each bid's unserved quantity for its function.
    for bid in bid_rows:
      i = int(bid["i"])
      f = int(bid["f"])
      if receiving[i,f] or sending[j,f]:
        continue
      q = min(remaining_capacities[f], bid["d"])
      if q > 0:
        y[i,j,f] += q
        remaining_capacities[f] -= q
        all_min_b[f] = min(all_min_b[f], bid["b"])
        sending[i,f] = True
        receiving[j,f] = True
      if bid["d"] > q:
        pending_bids.append((i, f, bid["d"] - q, bid["b"]))

    if tentatively_start_replicas and value_per_mb:
      # replicas started here compete for the same memory: serve first the
      # pending bids that bring the most value per MB of a new replica
      pending_bids.sort(key=lambda t: -t[3] * replica_capacity(j, t[1]) /
                        data[None]["memory_requirement"][t[1]+1])
    for i, f, remaining_d, b in pending_bids:
      if receiving[i,f] or sending[j,f]:
        continue
      if tentatively_start_replicas:
        current_a = int(additional_replicas[j,f])
        max_a = current_a + int(rho[j]/data[None]["memory_requirement"][f+1])
        # Reuse capacity from a replica already started, even if memory is exhausted.
        a = max(current_a, 1)
        while a <= max_a and remaining_d > 0:
          capacity = (r[j,f] + a) * (
            data[None]["max_utilization"][f+1] / data[None]["demand"][(j+1,f+1)]
          ) - ell[j,f] - y[:,j,f].sum()
          q = min(remaining_d, VAR_TYPE(max(0.0, capacity)))
          if q > 0:
            y[i,j,f] += q
            remaining_d -= q
            all_min_b[f] = min(all_min_b[f], b)
            sending[i,f] = True
            receiving[j,f] = True
            if current_a < a:
              rho[j] -= (a - current_a) * data[None]["memory_requirement"][f+1]
              additional_replicas[j,f] = a
              current_a = a
          a += 1
      if remaining_d <= 0 or not may_replace_existing_assignments:
        continue
      if b <= p[j,f]:
        continue
      # Only prior-round assignments are replaceable; a bid may span incumbents.
      for previous_buyer in np.nonzero(last_y[:,j,f])[0]:
        if previous_buyer == i:
          continue
        q = min(remaining_d, last_y[previous_buyer,j,f])
        y[previous_buyer,j,f] -= q
        y[i,j,f] += q
        last_y[previous_buyer,j,f] -= q
        remaining_d -= q
        all_min_b[f] = min(all_min_b[f], b)
        sending[i,f] = True
        receiving[j,f] = True
        sending[previous_buyer,f] = (
          previous_y[previous_buyer,:,f] + y[previous_buyer,:,f]
        ).sum() > 1e-10
        if remaining_d <= 0:
          break
    # compute utilization and update prices
    for f,b in all_min_b.items():
      u = (ell[j,f] + y[:,j,f].sum()) / total_capacity[j,f]
      p[j,f] = b + eta * (u - u0[j,f])
      if congestion_weight > 0:
        # demand the seller saw but could not serve, relative to its
        # capacity: local information, no extra message
        requested = all_bids_for_j.loc[all_bids_for_j["f"] == f, "d"].sum()
        excess = max(0.0, float(requested - y[:,j,f].sum()))
        p[j,f] += congestion_weight * excess / max(float(total_capacity[j,f]), 1.0)
    for f in set(functions_to_share) - set(all_min_b.index):
      p[j,f] *= (1 - auction_options["zeta"])
  return y, p, additional_replicas, len(unique_potential_sellers)


def start_replicas_by_value(
    memory_bids: pd.DataFrame,
    r: np.array,
    data: dict,
    rho: np.array
  ) -> Tuple[np.array, np.array]:
  """Greedy replica start by requested value per MB (seller-local).

  For each seller j, every memory bid (i, f) asks for ``d`` req/s worth
  ``v`` per req/s (advantage over the Cloud).  A new replica of f serves
  ``U_f / D_jf`` req/s and costs ``ram_f`` MB: the seller repeatedly starts
  the replica whose served requests (taken from the highest-value bids
  still uncovered) bring the largest value per MB, until the free memory or
  the requested demand runs out.  Bids from different buyers are not
  coordinated, so the same uncovered demand may be requested from several
  sellers; LSPr releases replicas that stay idle.
  """
  additional_replicas = np.zeros(r.shape)
  residual_memory = deepcopy(rho)
  if len(memory_bids) == 0:
    return additional_replicas, residual_memory
  for j, bids_for_j in memory_bids.groupby("j"):
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
        if ram_f > free:
          continue
        capacity = data[None]["max_utilization"][f+1] / data[None]["demand"][(j+1,f+1)]
        value, left = 0.0, capacity
        for v, d in requests:
          take = min(d, left)
          value += v * take
          left -= take
          if left <= 0:
            break
        if value > 0 and (best is None or value / ram_f > best[0]):
          best = (value / ram_f, f, capacity, ram_f)
      if best is None:
        break
      _, f, capacity, ram_f = best
      additional_replicas[j, f] += 1
      free -= ram_f
      residual_memory[j] -= ram_f
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


@contextmanager
def _madea2_rules(module):
  saved = {name: getattr(module, name) for name in _RULES}
  try:
    for name, function in _RULES.items():
      setattr(module, name, function)
    yield
  finally:
    for name, function in saved.items():
      setattr(module, name, function)


def run(config, parallelism, log_on_file=False, disable_plotting=False):
  """MADEA2: the MADEA runner with the value-per-MB replica rules."""
  with _madea2_rules(_madea):
    return _madea.run(config, parallelism, log_on_file, disable_plotting)


def run_one_shot(config, parallelism, log_on_file=False, disable_plotting=False,
                 *, refine_welfare=False):
  """One-shot2 (and one-shot2-PG): the one-shot runner with the same rules."""
  with _madea2_rules(_one_shot):
    return _one_shot.run(config, parallelism, log_on_file, disable_plotting,
                         refine_welfare=refine_welfare)


def run_one_shot_pg(config, parallelism, log_on_file=False, disable_plotting=False):
  return run_one_shot(config, parallelism, log_on_file, disable_plotting, refine_welfare=True)
