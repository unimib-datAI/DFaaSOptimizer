# FaaS-MALD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add FaaS-MALD, a Lagrangian-dual decentralized coordinator with a per-timestep duality-gap certificate, as a new standalone module plus tests and a LaTeX technical note.

**Architecture:** One new module `decentralized_dual.py` mirrors `decentralized_diffusion.py`'s runner skeleton; the only new logic is a projected dual-subgradient inner loop (`pair_scores`, `buyer_price_response`, `dual_coordination_round`) whose primal recovery reuses `decentralized_diffusion.evaluate_assignments`. Spec: `docs/superpowers/specs/2026-07-02-faas-mald-dual-coordination-design.md`.

**Tech Stack:** Python 3.10, numpy, pandas, networkx, Pyomo+Gurobi (via reused helpers), scipy (tests only), pytest, LaTeX (latexmk).

## Global Constraints

- **NEVER modify any existing file.** Only create: `decentralized_dual.py`, `tests/test_dual_helpers.py`, `tests/test_dual_e2e.py`, `faas-mald-note/*`.
- **Model policy:** every task is executed by a subagent with **`model: sonnet` at most**; `model: haiku` is allowed for simpler tasks. Each task below states its model.
- Code style: 2-space indentation, same import layout and docstring style as `decentralized_diffusion.py` / `decentralized_powerd.py`.
- Run tests with `uv run pytest <file> -v` from the repo root.
- Commits: one per task, message style matching repo history (short imperative subject).

## File Structure

- `decentralized_dual.py` — new coordinator: pure helpers (`pair_scores`, `buyer_price_response`, `dual_coordination_round`) + `run()` + CLI.
- `tests/test_dual_helpers.py` — unit tests for the pure helpers, incl. LP certificate check vs `scipy.optimize.linprog`.
- `tests/test_dual_e2e.py` — wiring + end-to-end test (skips without Gurobi), mirrors `tests/test_powerd_e2e.py`.
- `faas-mald-note/` — LaTeX technical note (`faas-mald.tex`, `main.tex`, `references.bib`, `README.md`, `.gitignore`).

---

### Task 1: Pure buyer-side helpers (`pair_scores`, `buyer_price_response`)

**Model:** sonnet

**Files:**
- Create: `decentralized_dual.py` (helpers only in this task)
- Test: `tests/test_dual_helpers.py`

**Interfaces:**
- Consumes: nothing from earlier tasks; only numpy/pandas and the standard `data[None]` dict format (`Nn`, `Nf`, `beta[(i+1,j+1,f+1)]`, `gamma[(i+1,f+1)]`).
- Produces:
  - `pair_scores(data: dict, neighborhood: np.array, latency, fairness: np.array, dual_options: dict) -> Tuple[np.array, np.array]` returning `(s, elig)` with shapes `(Nn,Nn,Nf)`; `s[i,j,f]` is the unpriced score, `elig` a bool mask (neighbor and `s > -gamma`); `s` is `-inf` where ineligible.
  - `buyer_price_response(omega: np.array, capacity: np.array, lam: np.array, s: np.array, elig: np.array) -> Tuple[pd.DataFrame, np.array, float]` returning `(bids, demand, buyer_dual_term)`; `bids` has columns `i, j, f, d, utility` (utility = unpriced score, for `evaluate_assignments`); `demand` is the uncapped argmax response `(Nn,Nf)` indexed by seller; `buyer_dual_term = Σ_{i,f} ω_{if}·max(0, max_j (s_{ijf} − λ_{jf}))`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dual_helpers.py`:

```python
import numpy as np
import pandas as pd

from decentralized_dual import buyer_price_response, pair_scores


def make_data(Nn=3, Nf=1, beta=None, gamma=0.05):
  data = {None: {
    "Nn": {None: Nn}, "Nf": {None: Nf},
    "beta": {}, "gamma": {},
    "memory_requirement": {f + 1: 2 for f in range(Nf)},
    "demand": {(j + 1, f + 1): 1.0 for j in range(Nn) for f in range(Nf)},
    "max_utilization": {f + 1: 0.7 for f in range(Nf)},
  }}
  for i in range(Nn):
    for f in range(Nf):
      data[None]["gamma"][(i + 1, f + 1)] = gamma
      for j in range(Nn):
        b = 1.0 if beta is None else beta[i][j][f]
        data[None]["beta"][(i + 1, j + 1, f + 1)] = b
  return data


def full_neighborhood(Nn):
  return np.ones((Nn, Nn)) - np.eye(Nn)


DUAL_OPTIONS = {"latency_weight": 0.0, "fairness_weight": 0.0}


def test_pair_scores_masks_non_neighbors_and_dominated_pairs():
  Nn, Nf = 3, 1
  beta = [[[1.0]] * Nn for _ in range(Nn)]
  beta[0][2][0] = -1.0  # worse than rejecting (gamma=0.05): ineligible
  data = make_data(Nn, Nf, beta=beta)
  neighborhood = full_neighborhood(Nn)
  neighborhood[0, 1] = 0  # not a neighbor
  s, elig = pair_scores(
    data, neighborhood, np.zeros((Nn, Nn)), np.zeros((Nn, Nf)), DUAL_OPTIONS
  )
  assert not elig[0, 1, 0] and not elig[0, 2, 0] and not elig[0, 0, 0]
  assert elig[1, 0, 0] and s[1, 0, 0] == 1.0
  assert s[0, 1, 0] == -np.inf


def test_buyer_response_zero_prices_picks_best_score_seller():
  Nn, Nf = 3, 1
  beta = [[[1.0]] * Nn for _ in range(Nn)]
  beta[0][1][0] = 2.0  # node 0 prefers node 1
  data = make_data(Nn, Nf, beta=beta)
  s, elig = pair_scores(
    data, full_neighborhood(Nn), np.zeros((Nn, Nn)), np.zeros((Nn, Nf)),
    DUAL_OPTIONS,
  )
  omega = np.zeros((Nn, Nf)); omega[0, 0] = 5.0
  capacity = np.full((Nn, Nf), 3.0)
  bids, demand, dual_term = buyer_price_response(
    omega, capacity, np.zeros((Nn, Nf)), s, elig
  )
  # uncapped argmax response: all 5 units demanded from node 1
  assert demand[1, 0] == 5.0 and demand[2, 0] == 0.0
  assert dual_term == 5.0 * 2.0
  # capped bids waterfill: 3 to node 1, 2 to node 2
  first = bids.iloc[0]
  assert (first.j, first.d) == (1, 3.0)
  assert bids["d"].sum() == 5.0


def test_buyer_response_price_shifts_demand():
  Nn, Nf = 3, 1
  beta = [[[1.0]] * Nn for _ in range(Nn)]
  beta[0][1][0] = 2.0
  data = make_data(Nn, Nf, beta=beta)
  s, elig = pair_scores(
    data, full_neighborhood(Nn), np.zeros((Nn, Nn)), np.zeros((Nn, Nf)),
    DUAL_OPTIONS,
  )
  omega = np.zeros((Nn, Nf)); omega[0, 0] = 5.0
  lam = np.zeros((Nn, Nf)); lam[1, 0] = 1.5  # price out node 1 (2-1.5 < 1)
  bids, demand, dual_term = buyer_price_response(
    omega, np.full((Nn, Nf), 10.0), lam, s, elig
  )
  assert demand[2, 0] == 5.0 and demand[1, 0] == 0.0
  assert dual_term == 5.0 * 1.0


def test_buyer_response_all_priced_out_yields_empty():
  Nn, Nf = 2, 1
  data = make_data(Nn, Nf)
  s, elig = pair_scores(
    data, full_neighborhood(Nn), np.zeros((Nn, Nn)), np.zeros((Nn, Nf)),
    DUAL_OPTIONS,
  )
  omega = np.zeros((Nn, Nf)); omega[0, 0] = 4.0
  lam = np.full((Nn, Nf), 10.0)
  bids, demand, dual_term = buyer_price_response(
    omega, np.full((Nn, Nf), 10.0), lam, s, elig
  )
  assert len(bids) == 0 and demand.sum() == 0.0 and dual_term == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_dual_helpers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'decentralized_dual'` (or ImportError).

- [ ] **Step 3: Implement the helpers**

Create `decentralized_dual.py`:

```python
from typing import Tuple

import numpy as np
import pandas as pd


def pair_scores(
    data: dict,
    neighborhood: np.array,
    latency: np.array,
    fairness: np.array,
    dual_options: dict,
  ) -> Tuple[np.array, np.array]:
  """Unpriced pair scores and eligibility for the coordination LP.

  s[i,j,f] = beta_{ijf} - w_lat * L_{ij} - w_fair * phi_{if}, defined only for
  neighbors j of i with s > -gamma_{if} (offloading beats rejection), the same
  score and filter used by define_assignments/best_response_sweep. Ineligible
  entries hold -inf.
  """
  Nn = data[None]["Nn"][None]
  Nf = data[None]["Nf"][None]
  s = np.full((Nn, Nn, Nf), -np.inf)
  elig = np.zeros((Nn, Nn, Nf), dtype=bool)
  for i in range(Nn):
    for j in np.nonzero(neighborhood[i, :])[0]:
      j = int(j)
      for f in range(Nf):
        val = (
          data[None]["beta"][(i + 1, j + 1, f + 1)]
          - dual_options["latency_weight"] * latency[i, j]
          - dual_options["fairness_weight"] * fairness[i, f]
        )
        if val > -data[None]["gamma"][(i + 1, f + 1)]:
          s[i, j, f] = val
          elig[i, j, f] = True
  return s, elig


def buyer_price_response(
    omega: np.array,
    capacity: np.array,
    lam: np.array,
    s: np.array,
    elig: np.array,
  ) -> Tuple[pd.DataFrame, np.array, float]:
  """One synchronous buyer step at seller prices ``lam``.

  Subgradient side: each buyer places its whole residual demand omega[i,f] on
  the argmax seller by price-adjusted score (if positive), giving the demand
  matrix D used in the projected subgradient g = D - C, and the buyer part of
  the dual value sum_{i,f} omega * max(0, max_j (s - lam)).

  Primal-recovery side: bids waterfill the ranked positive-adjusted-score
  sellers, capping each request at the advertised capacity; ``utility`` holds
  the unpriced score, on which evaluate_assignments later sorts.
  """
  Nn, Nf = omega.shape
  demand = np.zeros((Nn, Nf))
  dual_term = 0.0
  bids = {"i": [], "j": [], "f": [], "d": [], "utility": []}
  for i, f in zip(*np.nonzero(omega > 0)):
    i, f = int(i), int(f)
    sellers = np.nonzero(elig[i, :, f])[0]
    if len(sellers) == 0:
      continue
    adj = s[i, sellers, f] - lam[sellers, f]
    order = np.lexsort((sellers, -adj))
    best = float(adj[order[0]])
    if best > 0:
      j_star = int(sellers[order[0]])
      demand[j_star, f] += omega[i, f]
      dual_term += float(omega[i, f]) * best
    left = float(omega[i, f])
    for idx in order:
      if left <= 0 or adj[idx] <= 0:
        break
      j = int(sellers[idx])
      q = min(left, float(capacity[j, f]))
      if q <= 0:
        continue
      bids["i"].append(i)
      bids["j"].append(j)
      bids["f"].append(f)
      bids["d"].append(q)
      bids["utility"].append(float(s[i, j, f]))
      left -= q
  return pd.DataFrame(bids), demand, dual_term
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_dual_helpers.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add decentralized_dual.py tests/test_dual_helpers.py
git commit -m "add FaaS-MALD buyer-side dual helpers"
```

---

### Task 2: Dual subgradient loop with certificate (`dual_coordination_round`)

**Model:** sonnet

**Files:**
- Modify: `decentralized_dual.py` (append; created in Task 1 — this is our new file, allowed)
- Test: `tests/test_dual_helpers.py` (append)

**Interfaces:**
- Consumes: `pair_scores`, `buyer_price_response` (Task 1); `decentralized_diffusion.evaluate_assignments(bids, residual_capacity, data, ell, r, initial_rho, tentatively_start_replicas, last_y=None, diffusion_options=None, latency=None, fairness=None) -> (y, additional_replicas, n_sellers)` (existing, reused as-is).
- Produces: `dual_coordination_round(omega, residual_capacity, data, neighborhood, rho, dual_options, latency, fairness, force_memory_bids, ell, r) -> Tuple[np.array, np.array, pd.DataFrame, dict, int]` returning `(y_increment, additional_replicas, memory_bids, gap_info, n_active)`. `gap_info` keys: `"LB"`, `"UB"`, `"gap"`, `"inner_iterations"`, `"lam"` (final prices), `"lb_history"` (list). `dual_options` keys (all consumed here): `alpha0`, `step_rule` ("sqrt"|"polyak"), `theta`, `max_inner_iterations`, `gap_tolerance`, `latency_weight`, `fairness_weight`.

The certificate loop always calls `evaluate_assignments` with `last_y=None` and `tentatively_start_replicas=False`, so LB and UB bound the same LP (fresh assignment of `omega` to `residual_capacity`); the incumbent-replacement path of `evaluate_assignments` is deliberately unused. After the loop, one final call on the best bids uses `tentatively_start_replicas=(len(memory_bids) == 0)`, mirroring FaaS-MADiG.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dual_helpers.py`:

```python
from scipy.optimize import linprog

from decentralized_dual import dual_coordination_round


def coordination_lp_optimum(omega, capacity, s, elig):
  """Brute-force the coordination LP with scipy (test oracle)."""
  Nn, Nf = omega.shape
  pairs = [(i, j, f) for i in range(Nn) for j in range(Nn) for f in range(Nf)
           if elig[i, j, f]]
  if not pairs:
    return 0.0
  c = [-s[i, j, f] for (i, j, f) in pairs]
  A, b = [], []
  for i in range(Nn):
    for f in range(Nf):
      A.append([1.0 if (p[0], p[2]) == (i, f) else 0.0 for p in pairs])
      b.append(float(omega[i, f]))
  for j in range(Nn):
    for f in range(Nf):
      A.append([1.0 if (p[1], p[2]) == (j, f) else 0.0 for p in pairs])
      b.append(float(capacity[j, f]))
  res = linprog(c, A_ub=A, b_ub=b, bounds=[(0, None)] * len(pairs))
  assert res.success
  return -res.fun


def dual_round_setup(Nn=4, Nf=2, seed=7):
  rng = np.random.default_rng(seed)
  beta = [[[float(rng.uniform(0.5, 2.0)) for _ in range(Nf)]
           for _ in range(Nn)] for _ in range(Nn)]
  data = make_data(Nn, Nf, beta=beta)
  neighborhood = full_neighborhood(Nn)
  omega = rng.uniform(0.0, 4.0, size=(Nn, Nf))
  capacity = rng.uniform(0.0, 3.0, size=(Nn, Nf))
  return data, neighborhood, omega, capacity


DUAL_ROUND_OPTIONS = {
  "alpha0": 0.5, "step_rule": "sqrt", "theta": 1.0,
  "max_inner_iterations": 300, "gap_tolerance": 0.01,
  "latency_weight": 0.0, "fairness_weight": 0.0,
}


def run_round(data, neighborhood, omega, capacity, options=None):
  Nn = data[None]["Nn"][None]
  Nf = data[None]["Nf"][None]
  return dual_coordination_round(
    omega, capacity, data, neighborhood,
    rho=np.zeros(Nn), dual_options=options or DUAL_ROUND_OPTIONS,
    latency=np.zeros((Nn, Nn)), fairness=np.zeros((Nn, Nf)),
    force_memory_bids=False, ell=np.zeros((Nn, Nf)),
    r=np.zeros((Nn, Nf)),
  )


def test_certificate_brackets_lp_optimum_and_gap_closes():
  data, neighborhood, omega, capacity = dual_round_setup()
  y_inc, _, _, gap_info, _ = run_round(data, neighborhood, omega, capacity)
  s, elig = pair_scores(
    data, neighborhood, np.zeros_like(neighborhood),
    np.zeros(omega.shape), DUAL_ROUND_OPTIONS,
  )
  opt = coordination_lp_optimum(omega, capacity, s, elig)
  assert gap_info["UB"] >= opt - 1e-6
  assert gap_info["LB"] <= opt + 1e-6
  assert gap_info["gap"] <= 0.05
  # weak duality: reported LB is the value of the returned feasible y
  val = float((np.where(elig, s, 0.0) * y_inc).sum())
  assert abs(val - gap_info["LB"]) <= 1e-6


def test_best_lb_is_monotone_nondecreasing():
  data, neighborhood, omega, capacity = dual_round_setup(seed=11)
  _, _, _, gap_info, _ = run_round(data, neighborhood, omega, capacity)
  hist = gap_info["lb_history"]
  assert all(b >= a - 1e-12 for a, b in zip(hist, hist[1:]))


def test_recovered_increment_is_feasible():
  data, neighborhood, omega, capacity = dual_round_setup(seed=3)
  y_inc, _, _, _, _ = run_round(data, neighborhood, omega, capacity)
  assert (y_inc >= -1e-9).all()
  assert (y_inc.sum(axis=1) <= omega + 1e-6).all()
  assert (y_inc.sum(axis=0) <= capacity + 1e-6).all()


def test_price_rises_on_oversubscribed_seller():
  Nn, Nf = 3, 1
  beta = [[[1.0]] * Nn for _ in range(Nn)]
  beta[0][2][0] = 5.0
  beta[1][2][0] = 5.0  # both buyers want node 2
  data = make_data(Nn, Nf, beta=beta)
  omega = np.zeros((Nn, Nf)); omega[0, 0] = 4.0; omega[1, 0] = 4.0
  capacity = np.zeros((Nn, Nf)); capacity[2, 0] = 2.0; capacity[0, 0] = 6.0
  capacity[1, 0] = 6.0
  _, _, _, gap_info, _ = run_round(data, full_neighborhood(Nn), omega, capacity)
  assert gap_info["lam"][2, 0] > 0.0


def test_no_demand_returns_zero_gap_and_empty_outputs():
  data, neighborhood, _, capacity = dual_round_setup()
  omega = np.zeros((4, 2))
  y_inc, add_r, memory_bids, gap_info, n_active = run_round(
    data, neighborhood, omega, capacity
  )
  assert y_inc.sum() == 0.0 and add_r.sum() == 0.0
  assert len(memory_bids) == 0 and n_active == 0
  assert gap_info["LB"] == 0.0 and gap_info["UB"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_dual_helpers.py -v -k "certificate or monotone or recovered or oversubscribed or no_demand"`
Expected: FAIL with `ImportError: cannot import name 'dual_coordination_round'`.

- [ ] **Step 3: Implement `dual_coordination_round`**

Append to `decentralized_dual.py` (add `from decentralized_diffusion import evaluate_assignments` to the imports):

```python
def dual_coordination_round(
    omega: np.array,
    residual_capacity: np.array,
    data: dict,
    neighborhood: np.array,
    rho: np.array,
    dual_options: dict,
    latency: np.array,
    fairness: np.array,
    force_memory_bids: bool,
    ell: np.array,
    r: np.array,
  ) -> Tuple[np.array, np.array, pd.DataFrame, dict, int]:
  """Projected dual subgradient loop with primal recovery and gap certificate.

  Relaxes the seller capacity constraints of the coordination LP with prices
  lam >= 0. Each inner iteration: closed-form buyer responses at current
  prices (buyer_price_response), dual value L(lam) = lam.C + buyer terms as a
  valid upper bound (weak duality), feasible primal via the reused
  evaluate_assignments (lower bound; last_y=None and no tentative replicas so
  LB and UB bound the same LP), then lam <- max(0, lam + alpha_k (D - C)).
  Stops on relative gap <= gap_tolerance or max_inner_iterations. A final
  evaluate_assignments call on the best bids may tentatively start replicas
  (mirroring FaaS-MADiG) when no memory bids were generated.
  """
  Nn = data[None]["Nn"][None]
  Nf = data[None]["Nf"][None]
  s, elig = pair_scores(data, neighborhood, latency, fairness, dual_options)
  C = np.array(residual_capacity, dtype=float)
  lam = np.zeros((Nn, Nf))
  s_val = np.where(elig, s, 0.0)
  best_lb, best_ub = 0.0, np.inf
  best_bids = pd.DataFrame({"i": [], "j": [], "f": [], "d": [], "utility": []})
  lb_history = []
  n_active = int((omega > 0).any(axis=1).sum())
  gap = 0.0
  k = 0
  max_inner = int(dual_options["max_inner_iterations"])
  for k in range(1, max_inner + 1):
    bids, demand, dual_term = buyer_price_response(omega, C, lam, s, elig)
    best_ub = min(best_ub, float((lam * C).sum() + dual_term))
    if len(bids) > 0:
      y_try, _, _ = evaluate_assignments(
        bids, residual_capacity, data, ell, r, rho,
        tentatively_start_replicas=False, last_y=None,
        diffusion_options=dual_options, latency=latency, fairness=fairness,
      )
      lb = float((s_val * y_try).sum())
      if lb > best_lb:
        best_lb, best_bids = lb, bids
    lb_history.append(best_lb)
    gap = (best_ub - best_lb) / max(1.0, abs(best_ub))
    if gap <= dual_options["gap_tolerance"]:
      break
    if demand.sum() <= 0 and len(bids) == 0:
      break
    g = demand - C
    if dual_options["step_rule"] == "polyak":
      denom = float((g * g).sum())
      alpha = (
        0.0 if denom <= 0.0
        else dual_options["theta"] * max(0.0, best_ub - best_lb) / denom
      )
    else:
      alpha = dual_options["alpha0"] / np.sqrt(k)
    lam = np.maximum(0.0, lam + alpha * g)
  memory_bids = {"i": [], "j": [], "f": []}
  placed = (
    np.zeros((Nn, Nf)) if len(best_bids) == 0
    else evaluate_assignments(
      best_bids, residual_capacity, data, ell, r, rho,
      tentatively_start_replicas=False, last_y=None,
      diffusion_options=dual_options, latency=latency, fairness=fairness,
    )[0].sum(axis=1)
  )
  for i, f in zip(*np.nonzero(omega > 0)):
    i, f = int(i), int(f)
    if placed[i, f] < omega[i, f] or force_memory_bids:
      memory_requirement = data[None]["memory_requirement"][f + 1]
      for j in np.nonzero(neighborhood[i, :])[0]:
        if rho[int(j)] >= memory_requirement:
          memory_bids["i"].append(i)
          memory_bids["j"].append(int(j))
          memory_bids["f"].append(f)
  memory_bids = pd.DataFrame(memory_bids)
  y_increment = np.zeros((Nn, Nn, Nf))
  additional_replicas = np.zeros((Nn, Nf))
  if len(best_bids) > 0:
    y_increment, additional_replicas, _ = evaluate_assignments(
      best_bids, residual_capacity, data, ell, r, rho,
      tentatively_start_replicas=(len(memory_bids) == 0), last_y=None,
      diffusion_options=dual_options, latency=latency, fairness=fairness,
    )
  gap_info = {
    "LB": best_lb, "UB": best_ub, "gap": gap,
    "inner_iterations": k, "lam": lam, "lb_history": lb_history,
  }
  if not np.isfinite(gap_info["UB"]):
    gap_info["UB"] = 0.0
    gap_info["gap"] = 0.0
  return y_increment, additional_replicas, memory_bids, gap_info, n_active
```

Note for the implementer: `best_ub` stays `inf` only if the loop never runs (`max_inner_iterations == 0`); with no demand the first iteration yields `dual_term == 0` and `lam == 0`, so `UB == 0`. Keep the final normalization exactly as written so `test_no_demand_returns_zero_gap_and_empty_outputs` passes.

- [ ] **Step 4: Run all helper tests**

Run: `uv run pytest tests/test_dual_helpers.py -v`
Expected: 9 PASS (4 from Task 1 + 5 new).

- [ ] **Step 5: Commit**

```bash
git add decentralized_dual.py tests/test_dual_helpers.py
git commit -m "add FaaS-MALD dual subgradient round with gap certificate"
```

---

### Task 3: Runner, CLI, and end-to-end tests

**Model:** sonnet

**Files:**
- Modify: `decentralized_dual.py` (append `parse_arguments`, `run`, `__main__`)
- Test: `tests/test_dual_e2e.py`

**Interfaces:**
- Consumes: `dual_coordination_round` (Task 2); the same reused helpers `decentralized_diffusion.py` imports (`run_centralized_model`, `run_faasmacro`, `run_faasmadea`, `utils.*`, `models.sp`, `postprocessing.load_solution`).
- Produces: `run(config: dict, parallelism: int, log_on_file: bool = False, disable_plotting: bool = False) -> str` (returns the solution folder), CLI `uv run decentralized_dual.py -c <config> [-j N] [--disable_plotting]`. Output column/method name: `FaaS-MALD`. Options key: `solver_options["dual"]` with defaults `{"alpha0": 1.0, "step_rule": "sqrt", "theta": 1.0, "max_inner_iterations": 50, "gap_tolerance": 0.01, "latency_weight": 0.0, "fairness_weight": 0.0}` applied via `setdefault`, so existing configs run unchanged.

**Implementation instruction (read first):** open `decentralized_diffusion.py` and copy its `run()`, `parse_arguments()`, and `__main__` block verbatim into `decentralized_dual.py`, then apply ONLY the deltas below. Do not restructure. The imports to add at the top of `decentralized_dual.py` are exactly the ones `decentralized_diffusion.py` uses (minus `VAR_TYPE`), plus `from decentralized_diffusion import evaluate_assignments` already added in Task 2.

Deltas versus `decentralized_diffusion.run`:

1. Argparse description: `"Run FaaS-MALD"`.
2. Options block replaces the `diffusion_options` block:

```python
  dual_options = dict(solver_options.get("dual", {}))
  dual_options.setdefault("alpha0", 1.0)
  dual_options.setdefault("step_rule", "sqrt")
  dual_options.setdefault("theta", 1.0)
  dual_options.setdefault("max_inner_iterations", 50)
  dual_options.setdefault("gap_tolerance", 0.01)
  dual_options.setdefault("latency_weight", 0.0)
  dual_options.setdefault("fairness_weight", 0.0)
```

3. The `define_assignments` + `evaluate_assignments` block inside the inner `while not stop_searching` loop (diffusion lines building `bids`/`memory_bids` and then `diffusion_y`) is replaced by:

```python
      s = datetime.now()
      y_inc, additional_replicas, memory_bids, gap_info, n_active = (
        dual_coordination_round(
          omega, residual_capacity, sp_data, neighborhood, coordination_rho,
          dual_options, latency, fairness,
          force_memory_bids=(
            (coordination_rho > 0).any()
            and len(n_accepted_queue) >= n_accepted_queue.maxlen
            and all(x == n_accepted_queue[0] for x in n_accepted_queue)
          ),
          ell=ell, r=sp_r,
        )
      )
      rt = (datetime.now() - s).total_seconds()
      total_runtime += (rt / n_active) if n_active else rt
      rmp_omega = y.sum(axis=1)
      allocation_changed = (np.abs(y_inc) > tolerance).any()
      if allocation_changed:
        y += y_inc
        y[np.abs(y) < tolerance] = 0.0
        rmp_omega = y.sum(axis=1)
        fairness += (rmp_omega > tolerance)
        n_accepted_queue.append(rmp_omega.sum())
        bad_nodes = check_ls_pr_feasibility_from_fixed_y(sp_data, y)
        if bad_nodes:
          raise RuntimeError(
            f"LSPr infeasible from fixed y assignments: {bad_nodes}"
          )
        spr_sol, spr_obj, spr_tc, spr_runtime = compute_social_welfare(
          spr, sp_data, agents, solver_name, general_solver_options,
          y, rmp_omega, parallelism
        )
        total_runtime += spr_runtime
        sp_x, _, _, _, sp_r, sp_rho = spr_sol
        omega = sp_omega - rmp_omega
        omega[np.abs(omega) < tolerance] = 0.0
```

   (`additional_replicas` initialization to zeros is no longer needed — the round returns it.)
4. `check_stopping_criteria` receives `None` in place of `bids` (as `decentralized_bestresponse.py` does), and after it add the same no-progress guard pattern:

```python
      if not stop_searching and not allocation_changed and not (
          additional_replicas > tolerance
        ).any():
        stop_searching = True
        why_stop_searching = "no dual progress"
```

5. Termination string gains the certificate (this is the spec's logging requirement):

```python
        tc_dict["LSPr"].append(
          f"{why_stop_searching} "
          f"(it: {it}; gap: {gap_info['gap']:.6f}; "
          f"LB: {gap_info['LB']:.6f}; UB: {gap_info['UB']:.6f}; "
          f"inner: {gap_info['inner_iterations']}; "
          f"best it: {best_it_so_far}; "
          f"best centralized it: {best_centralized_it}; "
          f"total runtime: {total_runtime})"
        )
```

6. Output column: `pd.DataFrame(obj_dict["LSPr_final"], columns=["FaaS-MALD"])`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dual_e2e.py` (mirror of `tests/test_powerd_e2e.py`; the config is identical except the options key):

```python
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pytest

from decentralized_dual import run as run_dual


def _require_gurobi() -> None:
  solver = pyo.SolverFactory("gurobi")
  if not solver.available(exception_flag=False):
    pytest.skip("Gurobi solver is not available")


def _dual_e2e_config(base_solution_folder: Path) -> dict:
  return {
    "base_solution_folder": str(base_solution_folder),
    "seed": 21,
    "limits": {
      "Nn": {"min": 10, "max": 10},
      "Nf": {"min": 1, "max": 1},
      "neighborhood": {"type": "planar", "degree": 3},
      "weights": {
        "alpha": {"min": 1.0, "max": 1.0},
        "beta_multiplier": {"min": 1.5, "max": 2.0},
        "gamma": {"min": 0.05, "max": 0.1},
        "delta_multiplier": {"min": 0.1, "max": 0.2},
      },
      "demand": {"values": [1.0]},
      "memory_capacity": {"values": [12] * 10},
      "memory_requirement": {"values": [2]},
      "max_utilization": {"min": 0.7, "max": 0.7},
      "load": {
        "trace_type": "clipped",
        "min": {"min": 2.0, "max": 2.0},
        "max": {"min": 3.0, "max": 3.0},
      },
    },
    "solver_name": "gurobi",
    "solver_options": {
      "general": {"TimeLimit": 60, "OutputFlag": 0},
      "dual": {"alpha0": 1.0, "step_rule": "sqrt",
               "max_inner_iterations": 30, "gap_tolerance": 0.01,
               "latency_weight": 0.0, "fairness_weight": 0.0},
    },
    "max_iterations": 2,
    "patience": 1,
    "max_steps": 8,
    "min_run_time": 1,
    "max_run_time": 1,
    "run_time_step": 1,
    "checkpoint_interval": 1,
    "tolerance": 1e-6,
    "verbose": 0,
  }


def test_dual_runner_produces_expected_artifacts_with_gap(tmp_path):
  _require_gurobi()
  folder = run_dual(
    _dual_e2e_config(tmp_path), parallelism=0, disable_plotting=True
  )

  obj = pd.read_csv(Path(folder, "obj.csv"))
  assert len(obj) >= 1
  assert "FaaS-MALD" in obj.columns
  assert np.isfinite(pd.to_numeric(obj["FaaS-MALD"], errors="coerce")).all()

  runtime = pd.read_csv(Path(folder, "runtime.csv"))
  assert "tot" in runtime.columns
  assert (runtime["tot"] >= 0).all()

  tc = pd.read_csv(Path(folder, "termination_condition.csv"))
  assert tc.iloc[:, -1].astype(str).str.contains("gap:").all()

  assert Path(folder, "LSPc_solution.csv").exists()


def test_dual_runner_is_reproducible_for_same_seed(tmp_path):
  _require_gurobi()
  folder_a = run_dual(
    _dual_e2e_config(tmp_path / "a"), parallelism=0, disable_plotting=True
  )
  folder_b = run_dual(
    _dual_e2e_config(tmp_path / "b"), parallelism=0, disable_plotting=True
  )

  obj_a = pd.read_csv(Path(folder_a, "obj.csv"))["FaaS-MALD"].to_numpy()
  obj_b = pd.read_csv(Path(folder_b, "obj.csv"))["FaaS-MALD"].to_numpy()

  assert obj_a.shape[0] >= 1
  assert obj_a.shape == obj_b.shape
  assert np.allclose(obj_a, obj_b)


def test_dual_defaults_applied_without_dual_options_key(tmp_path):
  _require_gurobi()
  config = _dual_e2e_config(tmp_path)
  del config["solver_options"]["dual"]
  folder = run_dual(config, parallelism=0, disable_plotting=True)
  assert Path(folder, "obj.csv").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_dual_e2e.py -v`
Expected: FAIL with `ImportError: cannot import name 'run'` (or SKIP if Gurobi missing — in that case verify the import error by `uv run python -c "from decentralized_dual import run"`).

- [ ] **Step 3: Implement `run`, `parse_arguments`, `__main__`**

Copy from `decentralized_diffusion.py` and apply deltas 1–6 above. The `__main__` block:

```python
if __name__ == "__main__":
  args = parse_arguments()
  config = load_configuration(args.config)
  run(config, args.parallelism, disable_plotting=args.disable_plotting)
```

- [ ] **Step 4: Run the full new test suite**

Run: `uv run pytest tests/test_dual_helpers.py tests/test_dual_e2e.py -v`
Expected: all PASS (e2e SKIP only if Gurobi unavailable). Also run the sibling suites to prove nothing existing changed: `uv run pytest tests/test_diffusion_helpers.py tests/test_powerd_helpers.py -v` — all PASS, and `git status` shows only the two new files.

- [ ] **Step 5: Commit**

```bash
git add decentralized_dual.py tests/test_dual_e2e.py
git commit -m "add FaaS-MALD runner and e2e tests"
```

---

### Task 4: LaTeX technical note (`faas-mald-note/`)

**Model:** sonnet (the `.gitignore` and compile check are haiku-grade, but the note's math must match the shipped code, so keep one sonnet task)

**Files:**
- Create: `faas-mald-note/faas-mald.tex`, `faas-mald-note/main.tex`, `faas-mald-note/references.bib`, `faas-mald-note/README.md`, `faas-mald-note/.gitignore`

**Interfaces:**
- Consumes: the shipped `decentralized_dual.py` (Tasks 1–3) — the pseudocode and stated properties MUST match it; `faas-madig-note/` as the structural template (read `faas-madig-note/README.md`, `faas-madig-note/main.tex`, and `faas-madig-note/faas-madig.tex` first and follow their conventions exactly, including the fenced notation recap).
- Produces: a compilable standalone note (`latexmk -pdf main.tex` in `faas-mald-note/`).

- [ ] **Step 1: Read the template note**

Read `faas-madig-note/README.md`, `faas-madig-note/main.tex`, `faas-madig-note/faas-madig.tex`, and `faas-madig-note/.gitignore`. Copy `main.tex` and `.gitignore` adapting only the input filename (`faas-mald`).

- [ ] **Step 2: Write `faas-mald.tex`**

A `\section{FaaS-MALD: Lagrangian-dual coordination with an optimality certificate}` in the notation and style of `Decentralized_FaaS_coordination.pdf`, mirroring faas-madig.tex's structure, with these mandatory subsections and content:

1. *Self-contained notation recap* — fenced by `% BEGIN self-contained notation recap` / `% END ...` comments, reproducing the same table rows and capacity equations the MADiG note reproduces.
2. *The coordination problem* — the transportation LP exactly as in the spec:

```latex
\begin{align}
\max_{y \ge 0}\;& \sum_{i,f}\sum_{j \in N(i)} s_{ij}^{f}\, y_{ij}^{f} \\
\text{s.t. }& \textstyle\sum_{j} y_{ij}^{f} \le \omega_i^f, \qquad
             \textstyle\sum_{i} y_{ij}^{f} \le C_j^f,
\end{align}
```

with $s_{ij}^f = \beta_{ij}^f - w_{\mathrm{lat}} L_{ij} - w_{\mathrm{fair}}\varphi_i^f$ and the eligibility filter $s_{ij}^f > -\gamma_i^f$.
3. *Lagrangian relaxation and algorithm* — the dual function

```latex
\mathcal{L}(\lambda) = \sum_{j,f} \lambda_j^f C_j^f
  + \sum_{i,f} \omega_i^f \max\Bigl(0, \max_{j \in N(i)} \bigl(s_{ij}^f - \lambda_j^f\bigr)\Bigr),
```

the closed-form buyer response, the projected subgradient update $\lambda \leftarrow [\lambda + \alpha_k (D - C)]^+$ with $\alpha_k = \alpha_0/\sqrt{k}$ (and the Polyak option), and primal recovery by greedy capacity fill of the price-ranked bids (cite that it reuses the MADiG evaluation mechanism). Include an `algorithm`/`algorithmic` float whose steps match `dual_coordination_round` one-to-one.
4. *Properties* — (i) weak duality: every reported $(\mathrm{LB}, \mathrm{UB})$ pair is a valid per-timestep suboptimality certificate regardless of early stopping; (ii) with $\alpha_k = \alpha_0/\sqrt{k}$ the dual iterates converge to the dual optimum and, the LP having zero duality gap, $\mathrm{UB} \to$ LP optimum; (iii) per-iteration complexity $O(\sum_i |N(i)| \cdot |F|)$ buyer-side, $O(|N|\cdot|F|)$ seller-side, message complexity $O(|E|\cdot|F|)$, 1-hop only.
5. *Positioning with respect to the literature* — dual decomposition and subgradient methods (Bertsekas; Nedić & Ozdaglar 2009 on primal recovery/averaging), auction algorithms (Bertsekas ε-auction, as the MADEA baseline), price-based offloading in edge computing. Every claim carries a `\citep{}` backed by `references.bib`.

- [ ] **Step 3: Write `references.bib` and `README.md`**

`references.bib`: real, verifiable entries for at least — Bertsekas *Nonlinear Programming* (or *Parallel and Distributed Computation*, Bertsekas & Tsitsiklis 1989); Nedić & Ozdaglar 2009 "Approximate primal solutions and rate analysis for dual subgradient methods" (SIAM J. Optim. 19(4)); Bertsekas 1988 auction algorithm (Annals of OR); at least one recent price-/market-based edge offloading survey or paper already cited by the sibling notes' `references.bib` (reuse entries from `faas-madig-note/references.bib` where applicable). `README.md`: same structure as `faas-madig-note/README.md`, adapted (deliverable file name, algorithm one-liner, compile instructions).

- [ ] **Step 4: Compile check**

Run: `cd faas-mald-note && latexmk -pdf main.tex && latexmk -c`
Expected: `main.pdf` produced with no errors. If `latexmk` is not installed, run `command -v pdflatex` and use `pdflatex main.tex` twice + `bibtex main` if available; if no TeX toolchain exists on the machine, state that explicitly in the task report instead of claiming the note compiles.

- [ ] **Step 5: Cross-check note vs code**

Re-read `dual_coordination_round` and confirm the algorithm float matches it step-by-step (buyer response → UB update → primal recovery → LB update → gap test → subgradient step). Fix any mismatch in the note (never in the code).

- [ ] **Step 6: Commit**

```bash
git add faas-mald-note/
git commit -m "add FaaS-MALD technical note"
```

---

## Self-Review (done at plan-writing time)

- Spec coverage: coordination LP + dual loop (Tasks 1–2), certificate logging and runner/CLI/config defaults (Task 3), testing section (Tasks 1–3 tests), LaTeX note (Task 4), execution policy (Global Constraints + per-task Model lines), non-goals respected (no existing file modified, no remote_experiments integration). Benchmark comparison is explicitly out of scope in the spec.
- No placeholders: every code step carries complete code; Task 3 defines its deltas against a file the implementer is instructed to copy verbatim first.
- Type consistency: `dual_coordination_round` signature identical in Task 2 (definition), Task 2 tests (`run_round`), and Task 3 (call site); `gap_info` keys used in Task 3's termination string all produced in Task 2.
