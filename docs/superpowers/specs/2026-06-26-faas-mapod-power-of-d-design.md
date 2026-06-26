# FaaS-MAPoD — Power-of-d-choices distributed heuristic (+ LaTeX note)

**Date:** 2026-06-26
**Status:** Reviewed — ready for implementation
**Author:** brainstormed with Claude Code

This spec covers, in two parts:
- **Part I** — the FaaS-MAPoD algorithm (Option B from
  `2026-06-25-distributed-heuristic-alternatives.md`) and its implementation.
- **Part II** — a paper-ready LaTeX note for FaaS-MAPoD, mirroring
  `faas-madig-note/`.

---

# Part I — Algorithm: FaaS-MAPoD

## 1. Goal

Add **FaaS-MAPoD** (*Multi-Agent Power-of-d*), a randomized, partial-visibility
distributed heuristic for DiFRALB/DeFRALB, as a sixth comparable method
alongside LMM, FaaS-MACrO, FaaS-MADeA, HierarchicalAuction, and FaaS-MADiG.
FaaS-MAPoD is the **"random / partial-visibility" endpoint** of the
*market → greedy → random* spectrum: like FaaS-MADiG it uses no prices, but
instead of scanning the full one-hop neighbourhood it probes only `d` randomly
sampled neighbours per offloading step (power-of-d-choices). It isolates the
research question: *how much does full neighbourhood visibility (FaaS-MADiG)
buy over probing a random sample of `d`?*

## 2. Scope decisions (settled during brainstorming)

- **Local planning reused unchanged.** Same as FaaS-MADiG: each node solves the
  local MILP `LSP`/`LSPr` (or `LSP_fixedr` when `--fix_r` provides an
  `opt_solution_folder`), fixing `r`, `x`, residual `omega`; only the buyer-side
  request generation differs.
- **Two selection criteria, config-selectable** (`criterion ∈ {"score","capacity"}`):
  - `score` — pick the sampled neighbour with the highest
    `s_{ij}^f = beta[i,j,f] − w_lat·L_{ij} − w_fair·phi_i^f` (identical to
    FaaS-MADiG's score, but evaluated only on the `d`-sample → ablates
    *visibility*).
  - `capacity` — pick the sampled neighbour with the largest advertised residual
    capacity `C_j^f`. This is a buyer-side capacity-sampling variant inspired by
    shortest-queue power-of-d, not a fully capacity-ordered clearing rule:
    cross-buyer seller conflicts remain resolved by FaaS-MADiG's score-ordered
    seller rule.
- **Stochasticity handled by the existing pipeline.** The sampling RNG is seeded
  once per run from `config["seed"]`, so each run is reproducible. Variance is
  captured by the existing `--n_experiments` aggregation in `run.py` /
  `compare_results.py` (box/violin) — **no new repetition/averaging machinery**.
- **New standalone file** `decentralized_powerd.py`, mirroring
  `decentralized_diffusion.py` (established per-method-runner pattern; the
  new-file-over-shared-loop choice was already adjudicated for FaaS-MADiG).
- **Method name** `FaaS-MAPoD`; CLI key `faas-powd`; `obj.csv` column
  `FaaS-MAPoD`; artifact family `mkey = LSPc`. (Name adjustable in spec review.)

## 3. What changes vs. FaaS-MADiG (and what is reused)

FaaS-MAPoD reuses **everything** of FaaS-MADiG except the buyer-side request
generation:

- **Reused unchanged by import from `decentralized_diffusion`:**
  `evaluate_assignments` (seller-side greedy clearing — resolves cross-buyer
  conflicts deterministically by `(utility, i)`). Because the seller clearing
  is reused verbatim, conflict resolution is identical to FaaS-MADiG.
- **Reused unchanged by import from `run_faasmadea`:**
  `compute_residual_capacity`, `check_stopping_criteria`, `neigh_dict_to_matrix`,
  `start_additional_replicas`, `ensure_memory_sellers`,
  `check_ls_pr_feasibility_from_fixed_y`, `VAR_TYPE`.
- **Reused unchanged from the current FaaS-MADiG runner path:**
  `load_solution`, `encode_solution`, `LSP`, `LSP_fixedr`, and `LSPr`, so
  `--fix_r` / `opt_solution_folder` behaves exactly as in FaaS-MADiG.
- **New function** `sample_assignments(...)` replaces
  `define_assignments`. **New `run(...)`** is a copy of
  `decentralized_diffusion.run` with `define_assignments` → `sample_assignments`
  and an RNG created once from the seed.

### 3.1 `sample_assignments` (randomized buyer side)

```
sample_assignments(omega, blackboard, data, neighborhood, rho,
                   powerd_options, latency, fairness, force_memory_bids,
                   rng) -> (pd.DataFrame, pd.DataFrame, int)
```
Returns the same shapes as `define_assignments`: a bids DataFrame with columns
`["i","j","f","d","utility"]`, a memory-bids DataFrame `["i","j","f"]`, and the
number of potential buyers. (The `utility` column is **always** populated with
the score `s_{ij}^f`, regardless of `criterion`, so the reused
`evaluate_assignments` can sort sellers by `(utility, i)`.)

For each `(i,f)` with `omega_i^f > 0`:
1. Candidate sellers = one-hop neighbours with `blackboard[j,f] ≥ 1` **and**
   `s_{ij}^f > −gamma_i^f` (same convenience threshold as FaaS-MADiG/MADeA).
2. **Power-of-d loop** (let `criterion = powerd_options["criterion"]`,
   `d = powerd_options["d"]`). For each `(i,f)`, maintain a local
   `remaining_blackboard[j]` copy initialized from `blackboard[j,f]`; this avoids
   emitting more bids to a seller than the buyer has observed locally.
   - **Batched (default, `unit_bids=false`):** while `assigned < omega_i^f` and
     candidates remain: draw a sample `S` of `min(d, |candidates|)` candidates
     **uniformly at random without replacement** via `rng`; pick `j*` =
     `argmax_{j∈S} (s_{ij}^f, -j)` (if `criterion="score"`) or
     `argmax_{j∈S} (remaining_blackboard[j], -j)` (if
     `criterion="capacity"`); let
     `q=min(remaining_blackboard[j*], omega_i^f−assigned)`; emit a bid
     `(i, j*, f, d=q, utility=s_{i,j*}^f)`; decrement
     `remaining_blackboard[j*]` by `q`; remove `j*` from candidates;
     `assigned += q`.
   - **Per-unit (`unit_bids=true`, classic power-of-d):** repeat for each unit of
     `omega_i^f`: draw a fresh sample of `min(d, |candidates|)`, pick `j*` by the
     same criterion and tie-break, emit a unit bid `(i, j*, f, d=1, utility=s)`;
     decrement `remaining_blackboard[j*]`; remove `j*` from the candidate pool
     only when its advertised capacity is locally exhausted.
3. If `assigned < omega_i^f` (or `force_memory_bids`), emit **price-free
   replica-expansion bids** to neighbours with `rho_j > 0` — both same-node
   capacity sellers that ran out of capacity and memory-only neighbours,
   identical to FaaS-MADiG (two emission blocks). This keeps the difference
   from FaaS-MADiG to buyer visibility alone.

Degeneracy check: when `d ≥ |candidates|`, the sample is the whole candidate set
and `criterion="score"` reduces to FaaS-MADiG's greedy buyer rule for all
instances without exact score ties. With exact ties, MAPoD must remain
deterministic via the explicit lower-node-id tie-break; tests should avoid
claiming byte-for-byte equality with FaaS-MADiG unless the constructed scores
are unique.

### 3.2 `run(...)`

Copy of `decentralized_diffusion.run` with two changes:
- create the RNG once: `rng = np.random.default_rng(config["seed"])` (before the
  time loop), and pass it to `sample_assignments`;
- read `powerd_options = solver_options["powerd"]` (with
  `unit_bids` defaulting from `solver_options["auction"]["unit_bids"]` when
  present, `d` to `2`, `criterion` to `"score"`). Copy the options with
  `dict(...)` before applying defaults so the input config is not mutated.
Everything else (residual-capacity computation, the reused
`evaluate_assignments`, fairness update, restricted re-solve, `combine_solutions`
+ objective, stopping criteria, `obj.csv`/`runtime.csv`/`termination_condition.csv`
saving, and `opt_solution_folder` / `LSP_fixedr` support) is identical to
FaaS-MADiG. `obj.csv` column = `FaaS-MAPoD`. Runtime accounting
(`rt/n_auctions` + `spr_runtime`, with the zero-guard) is kept for a fair
comparison. Signature: `run(config, parallelism, log_on_file=False,
disable_plotting=False) -> str`.

## 4. Config: `solver_options["powerd"]`

```json
"powerd": {
  "d": 2,
  "criterion": "score",
  "latency_weight": 0.0,
  "fairness_weight": 0.0,
  "unit_bids": false
}
```
`criterion ∈ {"score","capacity"}`. The convenience filter and the seller-side
clearing always use the score populated in the `utility` column. When
`criterion="capacity"`, the weights do not affect the buyer-side ranking among
the sampled candidates, but they still affect candidate admissibility through
the convenience threshold and seller-side conflict resolution through
`evaluate_assignments`.

## 5. Wiring into `run.py` (additive only)

Same additive pattern as FaaS-MADiG: import
`from decentralized_powerd import run as run_powerd`; add `"faas-powd"` to
`--methods` choices and to `solution_folders`; add a `run_p` flag (init +
resume-check + `except ValueError` branch); extend the run-guard `if`; add a
dispatch block; extend the `mname` ternary with
`"FaaS-MAPoD" if method == "faas-powd"` (`mkey` already resolves to `LSPc`); add
a 7th `method_colors` entry. Existing methods' code paths stay byte-for-byte
unchanged.

## 6. Output, evaluation, comparison

Artifacts identical to FaaS-MADiG. Extend `compare_results.py` by adding
`FaaS-MAPoD` to the default model set and palette. Stochasticity surfaces
across `--n_experiments` (each experiment uses its own seed → its own instance
and its own sampling draw). Extend `config_files/planar_comparison.json` with
the `powerd` block so the planar campaign can run the full spectrum:
```
python run.py -c config_files/planar_comparison.json \
  --methods centralized faas-macro faas-madea hierarchical faas-diffuse faas-powd \
  --n_experiments 3 --loop_over Nn
```

## 7. Testing

- **Unit — `sample_assignments`** (no solver): determinism with two independent
  `np.random.default_rng(seed)` instances (not two calls on the same advanced
  RNG state); the sample size never exceeds `d`; `criterion="score"` vs
  `"capacity"` pick different sellers on a constructed instance where capacity
  and score disagree; capacity is consumed from the buyer-local
  `remaining_blackboard`; convenience threshold `> −gamma` filters candidates;
  no-capacity case emits `memory_bids`; **degeneracy**: with
  `d ≥ |candidates|`, `criterion="score"`, and unique scores, output equals
  FaaS-MADiG's `define_assignments` on the same input.
- **Reuse** the existing `evaluate_assignments` unit tests (no new seller-side
  tests needed; it is imported unchanged).
- **Unit — wiring:** `run.parse_arguments` accepts `faas-powd`; `run` exposes
  `run_powerd`; `compare_results` default model set and palette include
  `FaaS-MAPoD`; `decentralized_powerd.run` uses `LSP_fixedr` when
  `opt_solution_folder` is present and does not mutate `solver_options["powerd"]`.
- **Smoke / e2e** (Gurobi-gated): tiny instance via `decentralized_powerd.run`,
  asserting `obj.csv` (col `FaaS-MAPoD`), `runtime.csv`, `termination_condition.csv`,
  `LSPc_solution.csv` exist; assert reproducibility (same config seed → identical
  `obj.csv`).

## 8. Out of scope (v1)

- Option C (distributed best-response) — see the alternatives note.
- Fully capacity-ordered seller clearing for `criterion="capacity"`; v1 keeps
  score-ordered FaaS-MADiG seller clearing for apples-to-apples reuse.
- Per-instance K-seed averaging (the `--n_experiments` aggregation is used
  instead).

---

# Part II — LaTeX note: `faas-mapod-note/`

Mirrors `faas-madig-note/` exactly in structure and conventions.

## 9. Folder structure
```
faas-mapod-note/
├── faas-mapod.tex   % the \section to \input/paste into the paper
├── main.tex         % standalone preview wrapper (article + natbib bibliography)
├── references.bib   % cited works (subset shared with the MADiG note + Mitzenmacher)
├── README.md        % compile + insertion instructions
└── .gitignore       % LaTeX build artifacts (*.aux, *.bbl, main.pdf, …)
```

## 10. Content of `faas-mapod.tex` (concentrated on algorithm + rationale)

In the paper's notation; no abstract/intro; self-contained, removable notation
recap (same block as the MADiG note, reused so the note reads on its own).

1. **Hook (1–2 sentences):** FaaS-MAPoD is the randomized, partial-visibility
   sibling of FaaS-MADiG — same price-free score and local plan, but each
   offloading step probes only `d` randomly sampled neighbours.
2. **Notation and capacity model** — the same removable recap as the MADiG note
   (Eqs. 1,2,12,13 + Table 2/3 subset), so the note is self-contained.
3. **Power-of-d coordination** — define the sampling step and the two criteria
   (`score` Eq. mirroring the MADiG score; `capacity` buyer-side largest-spare-
   capacity sampling); state the batched vs per-unit forms; note the degeneracy
   `d ≥ |A_i^f|` ⇒ FaaS-MADiG for unique scores. Reuse FaaS-MADiG's seller-side
   greedy clearing (cite the sibling note/section).
4. **Two pseudocodes** (algorithmicx, paper style): *buyer power-of-d sampling*
   and a one-line note that the seller clearing is FaaS-MADiG's
   `evaluate_assignments`.
5. **What is removed / kept** vs FaaS-MADeA and vs FaaS-MADiG (removed: prices
   AND full-neighbourhood visibility; kept: local plan, residual/memory
   advertising, convenience threshold, fairness, replica expansion).
6. **Rationale and positioning** — honest: FaaS-MAPoD is essentially the
   **power-of-d-choices** rule of Mitzenmacher applied to FRALB offloading; it is
   *even less novel* than FaaS-MADiG and is used as the random/partial-visibility
   endpoint of the spectrum. Reproducibility via per-experiment seeding; variance
   via `--n_experiments`; communication footprint smaller still (only `d` probes
   per step).

## 11. Positioning / references (`references.bib`)

Reuse the verified entries from `faas-madig-note/references.bib`:
`mitzenmacher2001` (the **direct** ancestor — promote from "contrast" to "basis"),
`cybenko1989`, `willebeek1993`, `leechoi2021`, plus `bertsekas1988` if the
auction relation is mentioned. Cite FaaS-MADeA and FaaS-MADiG as sibling methods.
The positioning subsection states plainly that the buyer rule **is**
power-of-d-choices; novelty is not claimed; value is comparative (the random
endpoint and a partial-visibility ablation of FaaS-MADiG).

> Reuse the citation-verification workflow already applied to the MADiG note:
> the cited works are the same verified set (existence + metadata checked via
> CrossRef; PDFs in `faas-madig-note/cited_papers/`), so no re-verification is
> needed for the shared references.

## 12. Build & insertion
`latexmk -pdf main.tex` for a preview; insert by `\input{faas-mapod}` into the
paper, converting plain-text cross-refs to `\ref{}` and merging the (shared)
`.bib` entries. Remove the notation-recap block on insertion.

---

# Open questions / assumptions

- **Method name** `FaaS-MAPoD` / CLI `faas-powd` are final unless changed in
  review.
- **Default `d = 2`** (classic power-of-two-choices); configurable.
- **Seller clearing stays score-ordered** even when `criterion="capacity"`
  (so `evaluate_assignments` is reused unchanged). This is no longer an open
  implementation ambiguity: v1 deliberately randomizes only the
  probe/selection step, while cross-buyer conflict resolution remains the
  deterministic score order of FaaS-MADiG.
