# FaaS-MADiG — Distributed greedy-diffusion heuristic

**Date:** 2026-06-25
**Status:** Approved (brainstorming) — pending spec review
**Author:** brainstormed with Claude Code

## 1. Goal

Add a new **distributed coordination heuristic**, named **FaaS-MADiG**
(*Multi-Agent Distributed Greedy*), to be compared against the two existing
auction-based methods for the DiFRALB problem (distributed Function Replica
Allocation and Load Balancing in the Edge-Cloud continuum):

- **FaaS-MADeA** — flat decentralized auction (`run_faasmadea.py`). The
  older `decentralized_auction.py` module remains useful as historical/helper
  context, but it is not the production baseline dispatched by `run.py`.
- **HierarchicalAuction** — hierarchical multi-structure auction
  (`hierarchical_auction/`).

FaaS-MADiG is an **apples-to-apples ablation of the market mechanism**: it
keeps everything the production flat auction does except the price/bidding
signal, replacing seller selection with the same greedy ordering evaluated on a
price-free score. This isolates the scientific question: *does the auction's
price signal actually improve the solution over a plain greedy diffusion that
uses the same information?*

## 2. Scope decisions (settled during brainstorming)

- **Local planning is reused unchanged.** Each node still solves the local MILP
  subproblem `LSP`/`LSPr` (via `solve_subproblem` / `compute_social_welfare`),
  which fixes replicas `r[n,f]`, locally served load `x[n,f]`, and produces the
  residual load `omega[i,f]` to offload. Only the inter-node coordination is
  replaced.
- **New standalone file**, `decentralized_diffusion.py`, mirroring the current
  production FaaS-MADeA runner `run_faasmadea.py`. This follows the established
  codebase pattern (each method has its own runner reusing shared helpers), but
  avoids implementing the ablation against the older `decentralized_auction.py`
  behavior.
- **Method name:** `FaaS-MADiG`; CLI method key: `faas-diffuse`.
- **Alternatives (power-of-d-choices, distributed best-response) are recorded**
  for future work in
  `docs/superpowers/specs/2026-06-25-distributed-heuristic-alternatives.md`.

## 3. The precise ablation (what changes vs. the auction)

The production auction iteration loop (`run_faasmadea.run`) does, per iteration:

1. `compute_residual_capacity(sp_x, y, sp_r, sp_data)` → `capacity`,
   `residual_capacity`, `ell`; then `blackboard = max(0, capacity - sp_x)`.
2. `define_bids(...)` → `bids`, `memory_bids`
3. if bids: `evaluate_bids(...)` → `auction_y`, **updated prices `p`**;
   then `y += auction_y`; recompute `rmp_omega`; bump `fairness`;
   `compute_social_welfare(spr, ...)` (restricted problem) → updated
   `sp_x, sp_r, sp_rho`; update `omega = sp_omega - rmp_omega`.
   else: `start_additional_replicas(...)` → add replicas.
4. `combine_solutions` + `compute_centralized_objective`; track best.
5. `check_stopping_criteria`; save on stop.

FaaS-MADiG keeps **steps 1, 4, 5 and the non-bid branch of step 3 identical**.
It replaces only the two market functions in steps 2–3:

### 3.1 `define_bids` → `define_assignments`

Start from `run_faasmadea.define_bids`, not the older
`decentralized_auction.define_bids`. Keep its production behavior, with only
the price/bid signal removed:

- The buyer utility loses the price term:
  `score(i,j,f) = beta[i,j,f] − latency_weight·latency[i,j] − fairness_weight·fairness[i,f]`
  (the auction had an extra `− p[j,f]`). Candidate sellers are still neighbors
  with residual computing capacity `blackboard[j,f] >= 1`, kept using the same
  convenience threshold as FaaS-MADeA:
  `score > -data[None]["gamma"][(i+1,f+1)]`.
- Candidate sellers are sorted by descending `score`, matching the current
  greedy allocation structure of `run_faasmadea.define_bids`. The older
  softmax-proportional split in `decentralized_auction.py` is **not** used for
  this ablation because it would change both pricing and allocation behavior.
- Demand quantity `d` is computed exactly as in `run_faasmadea.define_bids`:
  if `solver_options["diffusion"]["unit_bids"]` is enabled, keep the same
  unit-request behavior; otherwise use
  `d = min(blackboard[j,f], omega[i,f] - assigned)`.
- The bid price `b` is **not computed** (no `epsilon`, no `delta`). **Reuse the
  existing `utility` column** that `define_bids` already produces (it stores the
  per-pair score) instead of adding a new `score` column — this minimizes churn,
  and `evaluate_assignments` will sort on `utility`. The `b` column is dropped;
  no downstream logic may interpret any column as a price.
- The partial-assignment and memory-request branches are unchanged:
  when assigned load is smaller than `omega[i,f]`, populate `memory_bids`
  exactly as `run_faasmadea.define_bids` does, including `force_memory_bids`.

### 3.2 `evaluate_bids` → `evaluate_assignments`

Sellers fill their residual capacity with a **pure greedy fill** — the
production capacity loop kept verbatim, every price-dependent piece removed:

- Candidate requests for seller `(j,f)` are sorted by **`utility` descending**
  (the auction sorted by bid price `b` descending). Pandas' stable sort would
  otherwise break ties by insertion order, so use an **explicit secondary key**
  on the buyer index for reproducibility:
  `sort_values(by=["utility", "i"], ascending=[False, True])`. The buyer that
  values seller `j` most for function `f` is served first.
- The capacity-filling loop (`q = min(remaining_capacity, d)`, accumulate into
  `y`, decrement `remaining_capacity`) is unchanged, and it still respects
  `residual_capacity[j,f]` (the array passed in this slot by the runner, see §3
  step 1 — `define_assignments` uses `blackboard`, the seller fill uses
  `residual_capacity`).
- **Keep `tentatively_start_replicas`** (the `initial_rho`/`r` tentative-replica
  branch): it is already **price-free** — it decides on utilization
  (`u <= max_utilization`), not on price — so it stays for parity with the
  baseline and feeds `additional_replicas`.
- **Drop the price machinery entirely** — this is the ablation:
  - no `min_b` tracking;
  - no final price update (`p[j,f] = min_b + eta·(u - u0)` and the
    `p[j,f] *= (1 - zeta)` decay) — `eta`, `zeta`, `u0` are not used;
  - **no previous-assignment replacement swap.** The production swap is gated by
    `b_arr[next_bid_idx] > p[j,f]`; with prices removed (`p ≡ 0`) that test is
    almost always true, causing assignment thrashing/oscillation. So the
    `last_y`-based swap block is **removed**, not re-derived on score. A
    non-regression unit test documents that, for the intended baseline,
    omitting the swap does not change accepted load relative to the greedy fill.
- `evaluate_assignments` returns `y`, `additional_replicas`, and `n_auctions`
  (`len(potential_sellers)`); **never a price array** (the production
  `evaluate_bids` returns `p` in slot 2 — that slot is dropped here).

Net effect: the only behavioral difference between FaaS-MADeA and FaaS-MADiG is
the absence of the price signal (cross-iteration congestion feedback +
seller-side price arbitration). Note the buyer-side greedy-by-utility ordering
already exists in production `define_bids`; what the auction adds on top — and
what this ablation removes — is precisely the price coordination. Replica
scaling (`start_additional_replicas`, tentative replicas), the
restricted-problem re-solve, the fairness mechanism, the objective, and all
stopping criteria are kept aligned with `run_faasmadea.py`.

## 4. New module: `decentralized_diffusion.py`

```
decentralized_diffusion.py
├── parse_arguments()              # same CLI as run_faasmadea (-c, -j, --disable_plotting)
├── define_assignments(...)        # NEW — see §3.1
├── evaluate_assignments(...)      # NEW — see §3.2
└── run(config, parallelism,       # copy of run_faasmadea.run with
        log_on_file=False,         #   steps 2–3 swapped and price logic removed
        disable_plotting=False)
```

Imported **unchanged** from `run_faasmadea` where possible:
`compute_residual_capacity`, `neigh_dict_to_matrix`, `start_additional_replicas`,
`check_stopping_criteria`, `check_ls_pr_feasibility_from_fixed_y`, and any
small helper already used by the production runner.

Imported (as the auction does) from `run_centralized_model`, `run_faasmacro`,
`utils.*`, `models.sp`: `init_problem`, `get_current_load`, `update_data`,
`init_complete_solution`, `join_complete_solution`, `save_checkpoint`,
`save_solution`, `plot_history`, `solve_subproblem`, `compute_social_welfare`,
`combine_solutions`, `compute_centralized_objective`, `decode_solutions`,
`LSP`, `LSPr`.

The `run()` signature matches `run_faasmadea.run` /
`hierarchical_auction.runner.run`: `(config, parallelism, log_on_file=False,
disable_plotting=False)`.

### Config: `solver_options["diffusion"]`

The auction reads `solver_options["auction"]` with keys including `epsilon`,
`eta`, `zeta`, `latency_weight`, `fairness_weight`, and possibly `unit_bids`.
FaaS-MADiG reads `solver_options["diffusion"]`:

```json
"diffusion": {
  "latency_weight": 0.0,
  "fairness_weight": 0.0,
  "unit_bids": false
}
```

`epsilon`, `eta`, `zeta`, `u0` are dropped (price-only parameters). If
`unit_bids` is omitted, default it to the auction config's value when present,
otherwise `false`, so benchmark configs can keep both methods aligned.

## 5. Wiring into `run.py`

- Add import: `from decentralized_diffusion import run as run_diffusion`.
- Add `"faas-diffuse"` to the `--methods` choices list.
- Add `"faas-diffuse": []` to the `solution_folders` structure and preserve it
  when loading existing `experiments.json` files that may not yet contain the
  key.
- Derive a `run_d` boolean from `--methods` alongside `run_a`/`run_h`, including
  the same resume/skip checks used for `faas-madea` and `hierarchical`.
- Add a dispatch block mirroring the auction one:
  ```python
  if run_d:
      d_folder = run_diffusion(
          config, sp_parallelism,
          log_on_file=log_on_file, disable_plotting=disable_plotting,
      )
      set_solution_folder(solution_folders, "faas-diffuse", experiment_idx, d_folder)
  ```
- Extend the `mkey`/`mname` mapping (around `run.py:277-290`):
  `mkey = "LSPc"` (same artifact family as the auctions), and
  `mname = "FaaS-MADiG"` for `method == "faas-diffuse"`.
- Add a color entry in `method_colors` so plots distinguish all four methods.
- Extend the old-instance reuse preference list if needed so experiments
  generated by FaaS-MADiG can also be reused as a source of load traces.

## 6. Output artifacts & comparison compatibility

Because `run()` is a copy of the auction loop, FaaS-MADiG produces the **same
artifacts** automatically:

- `LSP` / `LSPc` solution folders + checkpoints (`save_checkpoint`, `save_solution`).
- `obj_dict["LSPr_final"]` and a termination string in the auction's format:
  `"{why_stopping} (it: ...; obj. deviation: None; best it: ...; best centralized it: ...; total runtime: ...)"`.
  `run.py` already parses both the "obj. deviation" and the hierarchical formats
  (`run.py:158-166`), so no parser change is needed.
- `obj.csv` must use column name `FaaS-MADiG` so `run.py` and
  `compare_results.py` can keep method-specific columns distinct.
- **Runtime accounting (fairness of comparison).** `run_faasmadea.run` does not
  measure wall-clock: it accumulates a *per-agent average* via
  `total_runtime += rt / n_auctions` for the `define`/`evaluate` phases (plus
  `spr_runtime` for the restricted-problem solves), modelling distributed
  execution across agents. FaaS-MADiG **must keep the identical accounting** so
  the runtime comparison stays apples-to-apples. Add a guard for
  `n_auctions == 0` (latent divide-by-zero in the baseline) so empty-bid
  iterations don't raise.
- **Persist `runtime.csv` explicitly.** `run_faasmadea.py` does *not* save
  `runtime.csv` (only `obj.csv` + the termination CSV), forcing `run.py` to
  reconstruct runtime from logs. Recent commits (`a6646b8`, `84a2f1d`) added
  explicit `runtime.csv` saving to the faasmacro and hierarchical runners
  *"to avoid fragile log parsing"* — FaaS-MADiG follows that newer, robust
  direction and **saves `runtime.csv`** rather than relying on log parsing.

`compare_results.py` needs small compatibility updates before it can ingest
FaaS-MADiG robustly:

- add `FaaS-MADiG` to color/palette dictionaries used by box and violin plots;
- optionally extend the default `--models` list when FaaS-MADiG should be part
  of standard comparisons;
- verify all deviation helpers handle any model whose name starts with `FaaS-`.

### Three/four-way benchmark config

Extend `config_files/planar_comparison.json` to add the `"diffusion"` section
under `solver_options` so the existing planar campaign can run all methods:

```
python run.py -c config_files/planar_comparison.json \
  --methods centralized faas-macro faas-madea hierarchical faas-diffuse \
  --n_experiments 3 --loop_over Nn
```

## 7. Testing

Follow the existing `tests/` patterns (e.g. `test_run_faasmacro_helpers.py`,
`test_hierarchical_*`):

- **Unit — `define_assignments`:**
  - score uses `beta − latency_weight·latency − fairness_weight·fairness`, no price term;
  - convenience filtering matches `run_faasmadea.define_bids`
    (`score > -data[None]["gamma"][(i+1,f+1)]`);
  - assignments follow descending score order and respect the production
    `unit_bids`/non-unit quantity behavior;
  - no-capacity and partial-assignment cases populate `memory_bids` identically
    to `run_faasmadea.define_bids`, including `force_memory_bids`.
- **Unit — `evaluate_assignments`:**
  - total assigned to seller `(j,f)` never exceeds `residual_capacity[j,f]`;
  - higher-`utility` buyers are served first; tie-break is the explicit
    secondary sort on buyer index `i` (assert a constructed tie resolves to the
    lower `i`, not insertion order);
  - returns `y`, `additional_replicas`, `n_auctions` and **no price array**;
    running twice on the same input yields identical `y` (determinism);
  - `tentatively_start_replicas` branch still produces `additional_replicas`
    from `initial_rho`/`r` on the utilization condition;
  - **non-regression for the dropped swap:** on an input where the production
    `evaluate_bids` would have triggered the `last_y` replacement, assert the
    pure greedy fill yields the same total accepted load for the intended
    baseline (documents that omitting the swap is behavior-preserving here).
- **Unit — wiring/postprocessing:**
  - `run.parse_arguments` accepts `faas-diffuse`;
  - `run.results_postprocessing` maps `faas-diffuse` to `LSPc` /
    `FaaS-MADiG`;
  - `compare_results.py` plotting helpers accept `FaaS-MADiG` without palette
    errors.
- **Smoke / e2e:** a tiny instance (small `Nn`, `Nf`, few steps, short
  `TimeLimit`) run end-to-end via `decentralized_diffusion.run`, asserting the
  expected output files exist — including `obj.csv` (column `FaaS-MADiG`), the
  termination CSV, and a well-formed `runtime.csv`. Gate on solver availability
  like the other e2e tests.

## 8. Out of scope (recorded for future work)

- Power-of-d-choices randomized baseline (Option B).
- Distributed best-response / Gauss-Seidel relaxation (Option C).

See `2026-06-25-distributed-heuristic-alternatives.md`.

## 9. Open questions / assumptions

- **Convergence without prices.** Greedy diffusion lacks the price signal that
  damps oscillation. The main thrashing risk — the `last_y` replacement swap
  firing every iteration once prices are zero — is eliminated by dropping that
  swap (§3.2). Residual capacity is monotonically consumed within the iteration
  loop, and `check_stopping_criteria` (no capacity / all assigned / max
  iterations / time limit) bounds iterations. If empirical oscillation still
  appears, a simple guard (stop when `y` is unchanged between iterations) can be
  added — noted as a contingency, not built up front (YAGNI).
- **Method name** `FaaS-MADiG` is final unless changed during spec review.
