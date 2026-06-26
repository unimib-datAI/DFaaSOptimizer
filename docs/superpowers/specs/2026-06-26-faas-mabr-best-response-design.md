# FaaS-MABR — Distributed best-response (Gauss-Seidel) heuristics (+ LaTeX note)

**Date:** 2026-06-26
**Status:** Approved (brainstorming) — pending spec review
**Author:** brainstormed with Claude Code

This spec covers, in two parts:
- **Part I** — three distributed best-response methods (Option C from
  `2026-06-25-distributed-heuristic-alternatives.md`) and their implementation.
- **Part II** — a paper-ready LaTeX note for the FaaS-MABR family, mirroring
  `faas-madig-note/` and `faas-mapod-note/`.

---

# Part I — Algorithms: FaaS-MABR-S / -R / -O

## 1. Goal

Add the **FaaS-MABR** family (*Multi-Agent Best-Response*), three solver-light
**Gauss-Seidel** distributed heuristics for DiFRALB/DeFRALB, comparable head-to-head
with LMM, FaaS-MACrO, FaaS-MADeA, HierarchicalAuction, FaaS-MADiG, FaaS-MAPoD.
Where FaaS-MADeA (auction), FaaS-MADiG (greedy diffusion), and FaaS-MAPoD
(power-of-d) all compute every buyer's request **simultaneously** against a
snapshot and then resolve conflicts (a Jacobi-style round + seller clearing),
the FaaS-MABR family replaces that with a **sequential** sweep: nodes act in an
order, each claims capacity on a **shared residual-capacity ledger** that is
decremented immediately, so a node best-responds to what earlier nodes have
already taken. This is the Gauss-Seidel relaxation of the same coordination
problem. The research question: *does sequential best-response (immediate-update)
beat simultaneous diffusion (snapshot), and is anticipatory local
re-optimization worth its cost?*

## 2. The three methods (two orthogonal axes)

The family spans two axes — **node order** and **response type**:

| Method | CLI key | obj.csv col | order | response |
|--------|---------|-------------|-------|----------|
| **FaaS-MABR-S** | `faas-br-s` | `FaaS-MABR-S` | fixed (ascending node index) | greedy |
| **FaaS-MABR-R** | `faas-br-r` | `FaaS-MABR-R` | randomized per sweep (seeded) | greedy |
| **FaaS-MABR-O** | `faas-br-o` | `FaaS-MABR-O` | fixed (ascending node index) | re-optimization |

- **greedy response:** the node places its residual demand `omega_i^f` greedily,
  in descending score order, onto the current ledger (same score
  `s_{ij}^f = beta_{ij}^f − w_lat·L_{ij} − w_fair·phi_i^f` and convenience
  threshold `s > −gamma_i^f` as FaaS-MADiG).
- **re-optimization response (FaaS-MABR-O):** before placing, the node re-solves
  its local subproblem with `omega` **capped** at the capacity its neighbours
  currently still advertise, so it can shift demand to local execution / Cloud
  when neighbours are congested (an anticipatory best response), then places the
  capped `omega` greedily on the ledger.
- **fixed vs randomized order:** `-S`/`-O` sweep nodes in ascending index
  (deterministic, reproducible); `-R` draws a fresh node permutation each sweep
  from a per-run RNG (seeded from `config["seed"]`; variance via
  `--n_experiments`, exactly as FaaS-MAPoD).

(The fourth combination, randomized-order re-optimization, is intentionally
**not** exposed — YAGNI; it can be added later if needed.)

## 3. Architecture

One new module `decentralized_bestresponse.py`, **DRY internally, three methods
at the user level**:

- a shared sweep function `best_response_sweep(...)` parameterized by `order`
  and `response`;
- three thin public entry points `run_br_s`, `run_br_r`, `run_br_o`, each reading
  its own `solver_options` block and writing its own `obj.csv` column;
- a single internal `_run(config, parallelism, *, order, response, method_name,
  rng_seed, ...)` that holds the control-period loop (the three entry points are
  one-liners delegating to it).

**Reused unchanged by import** (from `run_faasmadea` / `run_faasmacro` /
`run_centralized_model` / `decentralized_diffusion`): `compute_residual_capacity`,
`check_stopping_criteria`, `neigh_dict_to_matrix`, `start_additional_replicas`,
`check_ls_pr_feasibility_from_fixed_y`, `VAR_TYPE`, `combine_solutions`,
`compute_social_welfare`, `decode_solutions`, `solve_subproblem`,
`init_problem`/`update_data`/`get_current_load`/`save_solution`/…, `LSP`,
`LSPr`, `LSP_fixedr`.

**Not reused:** `evaluate_assignments` / `define_assignments` (the sequential
sweep has no separate seller-clearing phase — sequential updates produce no
cross-buyer conflicts to resolve).

**Control-period loop reuse.** The `_run` loop is the same per-runner loop as
`decentralized_diffusion.run` (`for t: LSP → while not stop_searching: coordinate
→ re-solve LSPr → update omega → check stop → save`). Only the **coordinate**
step changes: one call to `best_response_sweep` (one sequential sweep) replaces
the `define_assignments`+`evaluate_assignments` pair. The inter-sweep `LSPr`
re-solve is what makes this the Gauss-Seidel relaxation (each sweep best-responds
to the state left by the previous sweep's re-solve). The existing
`check_stopping_criteria` detects the fixed point (a sweep that places no new
load) and `max_iterations`.

### 3.1 `best_response_sweep`

```
best_response_sweep(omega, residual_ledger, data, neighborhood, rho,
                    br_options, latency, fairness, force_memory_bids,
                    *, order, response, rng=None,
                    reopt_ctx=None) -> (y_increment, memory_bids, n_active)
```
- `residual_ledger`: a **mutable copy** of the current true residual capacity
  (from `compute_residual_capacity`), `shape (Nn, Nf)`, decremented in place as
  nodes claim — this is the shared ledger that gives Gauss-Seidel its
  immediate-update behaviour.
- Returns `y_increment` `(Nn, Nn, Nf)` (this sweep's placements), `memory_bids`
  DataFrame `[i,j,f]` (price-free replica-expansion requests, identical to
  FaaS-MADiG), and `n_active` (number of nodes that acted this sweep, for runtime
  amortization).

**Sweep body.** Determine node visiting order: `range(Nn)` if `order=="fixed"`,
else `rng.permutation(Nn)`. For each node `i` in that order, for each `f` with
`omega[i,f] > 0`:
1. (**FaaS-MABR-O only**) re-optimize: cap `omega[i,f]` at the accessible
   neighbour capacity `sum_{j in N_i} residual_ledger[j,f]` and re-solve node
   `i`'s local subproblem via `reopt_ctx` (§3.2), replacing `omega[i,f]` (and
   `x_i^f`, `z_i^f`) with the re-optimized split. `-S`/`-R` skip this step.
2. Admit candidates `j in N_i` with `residual_ledger[j,f] >= 1` and
   `s_{ij}^f > −gamma_i^f`; sort by descending `s_{ij}^f` (tie-break lower `j`).
3. Place greedily: for each candidate `j` in order, `q = min(residual_ledger[j,f],
   omega[i,f] − placed)`; `y_increment[i,j,f] += q`; **`residual_ledger[j,f] −= q`**
   (immediate decrement → later nodes see less); `placed += q`; stop when
   `omega[i,f]` met or candidates exhausted.
4. If `placed < omega[i,f]` (or `force_memory_bids`), emit price-free
   replica-expansion `memory_bids` to neighbours with `rho_j > 0` (both
   still-serving capacity sellers and memory-only neighbours), **at parity with
   FaaS-MADiG** (the two-block emission).

### 3.2 Re-optimization context (FaaS-MABR-O)

`reopt_ctx` bundles what a single-node re-solve needs (`sp_data`, `agents`,
the solver name/options, the model factory). For node `i` it solves
`LSP_capped` (§4) with `omega_ub[f] = sum_{j in N_i} residual_ledger[j,f]`, taking
the rest of node `i`'s parameters from `sp_data`. It returns the re-optimized
`x_i`, `omega_i`, `z_i` for the node, which overwrite the node's row before the
greedy placement of step 3. Single-agent solves reuse the same solver-invocation
path `solve_subproblem` uses, restricted to agent `i`.

## 4. Model addition: `LSP_capped` (additive, `models/sp.py`)

Add a new class **only**; do not modify existing classes:

```python
class LSP_capped(LSP):
  # adds an upper bound on horizontal offloading per function
  #   omega[f] <= omega_ub[f]
  # where omega_ub is a Param fed from accessible neighbour residual capacity.
```

It subclasses `LSP`, declares `omega_ub` as a `Param`, and adds one
`Constraint` `omega[f] <= omega_ub[f]`. Everything else (objective, traffic-loss,
utilization, residual-capacity constraints) is inherited unchanged. Only
FaaS-MABR-O instantiates it; FaaS-MADiG/MAPoD/MADeA/MACrO and FaaS-MABR-S/-R are
untouched.

## 5. Config: `solver_options.{br_s, br_r, br_o}`

Three blocks (one per method), all sharing the latency/fairness weights:
```json
"br_s": { "latency_weight": 0.0, "fairness_weight": 0.0 },
"br_r": { "latency_weight": 0.0, "fairness_weight": 0.0 },
"br_o": { "latency_weight": 0.0, "fairness_weight": 0.0 }
```
`-R` additionally uses `config["seed"]` for its per-sweep permutation RNG
(`np.random.default_rng(seed)`, created once). `_run` copies its options block
with `dict(...)` before applying defaults, so the input config is never mutated
(as in FaaS-MAPoD).

## 6. Wiring (`run.py`, `compare_results.py`, planar config — additive only)

Same additive pattern as FaaS-MADiG/MAPoD, ×3: import `run_br_s`/`run_br_r`/
`run_br_o`; add `faas-br-s`/`faas-br-r`/`faas-br-o` to `--methods` choices and to
the `solution_folders` template; three `run_*` flags (init + resume-check +
`except ValueError`); extend the run-guard; three dispatch blocks; extend the
`mname` ternary (`FaaS-MABR-S/-R/-O`, all resolving `mkey = "LSPc"`); add three
`method_colors`; add the three names to `compare_results` default model set and
both palettes; add the three blocks to `config_files/planar_comparison.json`.
Existing methods' code paths stay byte-for-byte unchanged.

## 7. Output, evaluation, comparison

Artifacts identical to FaaS-MADiG/MAPoD (so `compare_results.py` ingests them
unchanged except palette/default-set keys). `-R` variance surfaces across
`--n_experiments` (each experiment: own seed → own instance and own sweep order).
Extend the planar campaign:
```
python run.py -c config_files/planar_comparison.json \
  --methods centralized faas-madea hierarchical faas-diffuse faas-powd \
            faas-br-s faas-br-r faas-br-o \
  --n_experiments 3 --loop_over Nn
```

## 8. Testing

- **Unit — `best_response_sweep` (greedy, no solver):**
  - sequential ledger: a constructed instance where node 0 (acting first) claims
    a shared seller's capacity, leaving node 1 (later) strictly less than a
    simultaneous round would — i.e. the ledger decrement is observable and
    order-dependent;
  - `order="fixed"` determinism (two calls → identical `y_increment`);
  - `order="random"` reproducibility with two independent `default_rng(seed)`;
  - admission threshold `> −gamma` filters candidates; descending-score order +
    lower-`j` tie-break;
  - no-capacity → `memory_bids` (parity two-block emission, asserted against the
    FaaS-MADiG memory-bid shape).
- **Unit — `LSP_capped`:** with a tight `omega_ub`, the solved `omega[f]` respects
  the cap and the freed demand moves to `x`/`z` (constructed small model);
  without a cap (or a loose one) it matches plain `LSP`.
- **Unit — reopt step:** on a constructed node whose neighbours are nearly full,
  FaaS-MABR-O caps `omega` and rebalances to Cloud/local versus FaaS-MABR-S on
  the same instance.
- **Unit — wiring:** the three CLI keys accepted; `run` exposes `run_br_s/r/o`;
  planar config has the three blocks; `compare_results` default set + palette
  include the three names.
- **Smoke/e2e (Gurobi-gated):** one tiny run per method asserting `obj.csv`
  (correct column), `runtime.csv`, `termination_condition.csv`, `LSPc_solution.csv`;
  `-S`/`-O` reproducible (same seed → identical `obj.csv`).

## 9. Out of scope (v1)

- Randomized-order re-optimization (the 4th axis combination).
- `--fix_r` for FaaS-MABR-O beyond what `LSP_fixedr` already gives the greedy
  variants (the cap and fixed-r interaction is deferred).
- Any change to FaaS-MADeA/MACrO/MADiG/MAPoD/hierarchical code paths.

---

# Part II — LaTeX note: `faas-bestresponse-note/`

Mirrors `faas-madig-note/` / `faas-mapod-note/` in structure and conventions.

## 10. Folder structure
```
faas-bestresponse-note/
├── faas-mabr.tex     % the \section to \input/paste into the paper
├── main.tex          % standalone preview wrapper (article + natbib)
├── references.bib    % cited works
├── README.md
└── .gitignore
```

## 11. Content of `faas-mabr.tex`

In the paper's notation; no abstract/intro; self-contained removable notation
recap (reused from the sibling notes). Sections:
1. **Hook:** FaaS-MABR is the *sequential* (Gauss-Seidel) counterpart of the
   *simultaneous* (Jacobi) coordination of FaaS-MADiG/MAPoD; nodes best-respond
   on a shared ledger updated in place.
2. **Notation recap** (reused, removable).
3. **Sequential best-response sweep** — the shared ledger, the visiting order,
   the greedy claim with immediate decrement; Eqs. for the score and admission
   (same as FaaS-MADiG). State the Jacobi↔Gauss-Seidel relation explicitly.
4. **The three variants** — fixed vs randomized order; greedy vs re-optimization
   (the `omega`-capped local re-solve, with the `LSP_capped` constraint written
   out).
5. **Pseudocode** — one Algorithm for the sweep (parameterized by order/response).
6. **What is kept / changed** vs FaaS-MADiG (kept: local plan, score, threshold,
   replica expansion, restricted re-solve; changed: simultaneous→sequential, no
   seller-clearing phase, optional anticipatory re-optimization).
7. **Positioning (honest):** Gauss-Seidel relaxation / sequential best-response
   dynamics — a classical distributed-optimization scheme, not novel; value is
   the *sequential-vs-simultaneous* contrast and the re-optimization ablation.
   Cite **Bertsekas & Tsitsiklis, *Parallel and Distributed Computation: Numerical
   Methods* (1989)** (Gauss-Seidel / Jacobi relaxation), the diffusion/auction
   siblings (Cybenko, FaaS-MADeA), and best-response/better-response dynamics.

## 12. Build & insertion
`latexmk -pdf main.tex` for a preview; `\input{faas-mabr}` into the paper,
removing the notation recap and converting cross-refs. Cited PDFs verified +
downloaded into `faas-bestresponse-note/cited_papers/` (same audit workflow as
the sibling notes; reuse the shared verified set where it overlaps).

---

# Open questions / assumptions

- Names `FaaS-MABR-S/-R/-O` and CLI `faas-br-s/-r/-o` are final unless changed in
  review.
- Re-optimization solves one `LSP_capped` per node per sweep — accepted as the
  method's (higher) cost; runtime is amortized per active node as in the
  siblings, and only the MILP solves are charged in full.
- The shared ledger is the **true** residual capacity (`compute_residual_capacity`)
  decremented in place, not the advertised blackboard; admission still uses the
  score/threshold.
- Stopping reuses `check_stopping_criteria` (fixed point = a sweep that places no
  new load, plus the existing omega/`max_iterations` guards).
