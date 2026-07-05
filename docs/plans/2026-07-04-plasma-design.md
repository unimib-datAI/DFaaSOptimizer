# PLASMA Design (repo-aligned)

**Supersedes:** `/Users/micheleciavotta/Downloads/PLASMA_SPEC.md` (standalone draft).
**Status:** design specification, ready for implementation.
**Relation to existing work:** PLASMA is a new fully-decentralized method for the
same FRALB/DiFRALB problem solved by `run_centralized_model.py` (LMM),
`run_faasmacro.py` (FaaS-MACrO) and `run_faasmadea.py` (FaaS-MADeA), evaluated
side by side with them exactly like `decentralized_gcaa.py`,
`decentralized_bestresponse.py`, `decentralized_diffusion.py`,
`decentralized_dual.py`, `decentralized_potentialgame.py`,
`decentralized_powerd.py` and `hierarchical_auction/` already are.

The algorithmic content (Physarum routing Layer A, discrete Simulated
Bifurcation Layer B, Hamiltonian, protocol, acceptance criteria) is unchanged
from the original spec, §§2–5 and §9–10 there. This document fixes the
conflicts identified against this repo's conventions and rules:

1. Python version pin.
2. New dependencies where the repo already has an equivalent tool.
3. Package/CLI/test layout inconsistent with existing conventions.
4. The asynchronous variant (PLASMA-A) is **removed entirely**. PLASMA ships
   only the synchronous, round-barrier execution described as PLASMA-S in the
   original spec's §4.4. There is a single execution mode; `execution.mode`
   as a config axis does not exist.

## 1. Constraints resolved

### 1.1 Python version

`pyproject.toml` pins `requires-python = ">=3.10,<3.11"`. That file is not to
be modified for this feature. PLASMA targets **Python 3.10**: no
`match` on complex patterns beyond what 3.10 supports, no 3.11-only stdlib
(`tomllib`, `ExceptionGroup`). Use `from __future__ import annotations` where
helpful. Drop the original spec's "Python 3.11+" line.

### 1.2 Reuse existing tools, do not modify them; new functions are fine

Per the pattern already used by `hierarchical_auction/madea_runner.py` ("the
implementation may import existing functions but must not modify the
FaaS-MADeA functions or the existing hierarchical runner"), PLASMA:

- **Must not edit** any existing file (`models/*`, `run_centralized_model.py`,
  `run_faasmacro.py`, `run_faasmadea.py`, `heuristic_coordinator.py`,
  `generators/*`, `postprocessing.py`, `logs_postprocessing.py`,
  `compare_results.py`, `run.py` internals other than the additive
  registration described in §3).
- **May freely import and call** existing functions/classes from those
  modules.
- **May add new functions/modules** wherever the existing surface doesn't
  cover what PLASMA needs (the SB integrator, the Hamiltonian, the heartbeat
  protocol, the routing dynamics have no existing analogue and are written
  from scratch as originally specified).

Concretely, replace the three new-dependency baselines from the original
spec with thin wrappers over what already exists:

| Original spec (`plasma/baselines/*`) | New dependency | Replacement (reuse, no new dependency) |
|---|---|---|
| `milp.py` via PuLP/CBC | `pulp` | New module `plasma/baselines/milp_baseline.py` that imports `init_problem` from `run_centralized_model.py`, `update_data` from `generators/generate_data.py`, `get_current_load` from `utils/centralized.py`, and the Pyomo model in `models/sp.py`/`models/model.py`, re-solved every `K` s on observed rates. Same Gurobi/GLPK solver path as every other baseline in the repo. |
| `greedy.py` | — | New module `plasma/baselines/greedy_baseline.py` that instantiates `heuristic_coordinator.GreedyCoordinator` (already implements a local, non-coordinated greedy allocator) instead of writing a new greedy solver. |
| `sim/traffic.py`, `sim/scenarios.py` | — | Reuse `generators/generate_data.py`, `generators/load_generator.py`, `generators/generate_load.py` for integer arrival traces (existing `clipped`/`sinusoidal` trace types) and for topology generation (`generate_neighborhood` already supports ER via `p`, G(n,m) via `m`, k-regular via `k`, and Euclidean planar — same `neighborhood` config as every other method). Barabási–Albert is **not** supported and is not added: milestones use G(n,m) or k-regular instead. Anything genuinely missing (pure Poisson traces, MMPP bursts) is written as new generator functions in `plasma/sim/` — never by editing `generators/*`. |
| `sim/network.py` via `simpy` | `simpy` | New module `plasma/sim/clock.py`: a `heapq`-based discrete-event scheduler (stdlib `heapq` + `dataclasses`) driving a single shared round barrier (period `W`; SB pass every `k_sb` rounds) and delivering heartbeat/data-plane messages with configurable latency/loss. `simpy`'s extra features (resources, processes-as-generators, per-node independent clocks) are not needed — there is one global clock. |
| `cli.py` via Typer | `typer` | No new CLI framework. PLASMA is invoked the same way as every other method: through `run.py --methods ...` (see §3) plus, for direct debugging, a small `argparse` entry point in `plasma/cli.py`, matching `run_centralized_model.py`/`run_faasmacro.py`. |

Net new third-party dependency: **none**. Everything PLASMA needs (numpy,
networkx, pandas, pyyaml) is already in `pyproject.toml`.

### 1.3 Package layout and tests

Keep `plasma/` as its own subpackage — `hierarchical_auction/` already
establishes that a self-contained algorithm package is an accepted shape in
this repo, so this is not a new convention, just a second instance of one.
Layout, trimmed to what §1.2 didn't already reuse:

```
plasma/
  core/
    types.py          # NodeId, FnId, frozen dataclasses for config & messages
    node.py            # PlasmaNode: owns Layer A + Layer B state; step(dt) API
    routing.py          # Layer A: sampling, gates, conductance updates
    sbm.py              # Layer B: encoding, local Hamiltonian, dSB integrator, hysteresis, repair
    protocol.py         # heartbeat encode/decode, staleness handling
  sim/
    clock.py            # heapq-based discrete-event scheduler, single shared round barrier (period = W)
  baselines/
    milp_baseline.py     # wraps run_centralized_model.init_problem + models/sp.py
    greedy_baseline.py    # wraps heuristic_coordinator.GreedyCoordinator
    madea_iface.py        # adapter so run_faasmadea can be plugged in for comparison
  eval/
    regret.py            # dynamic/static oracle regret, adaptation lag (genuinely new metrics)
  cli.py                 # argparse entry point for standalone debugging runs
```

Drop the original spec's `sim/traffic.py`, `sim/scenarios.py`, `sim/network.py`
and `eval/metrics.py`/`eval/plots.py` as separate modules: traces/topologies
come from `generators/`, headline metrics/plots come from
`postprocessing.py`/`logs_postprocessing.py`/`compare_results.py` (extended,
not forked — see §3). `eval/regret.py` holds only the metrics that don't
exist anywhere in the repo yet (cumulative regret vs. dynamic/static oracle,
adaptation lag).

Tests move out of the package and into the existing flat `tests/` directory,
one file per concern, matching the `test_<module>_<aspect>.py` convention
used everywhere else (`test_gcaa_helpers.py`, `test_gcaa_e2e.py`,
`test_gcaa_wiring.py`, ...):

```
tests/test_plasma_routing.py       # Layer A: gates, conductance updates, LP convergence
tests/test_plasma_sbm.py           # Layer B: dSB vs brute force, hysteresis, RAM repair
tests/test_plasma_protocol.py      # heartbeat, staleness, locality invariant
tests/test_plasma_clock.py         # round barrier scheduling, phase-locked-commit scenario
tests/test_plasma_baselines.py     # milp_baseline/greedy_baseline wiring, madea_iface
tests/test_plasma_e2e.py           # full pipeline vs MILP on a small graph
```

## 2. Only the synchronous variant exists

The original spec's §4.4 described two execution modes, PLASMA-A (async,
independent jittered per-node clocks, no barrier) and PLASMA-S (sync, shared
round barrier, period `W`). **PLASMA-A is dropped entirely.** Every other
decentralized method already in this repo (FaaS-MACrO, FaaS-MADeA, GCAA,
best-response, diffusion, dual, potential-game, power-d, hierarchical) runs
on synchronized rounds, so the sync formulation is the one directly
comparable to all of them on the repo's own terms; the async variant added
implementation surface (independent clocks, re-jittered ticks, phase-lock as
an "accidental" pathology) without a comparable baseline to evaluate it
against.

Consequences for the rest of this design:

- `plasma/sim/clock.py` implements **only** the shared round barrier (period
  `W`; SB pass every `k_sb` rounds, default 10). There is no per-node clock,
  no tick jitter, no `execution.mode` config key.
- The "Jacobi-oscillation countermeasure" (randomized commit with probability
  `p_commit`, default 0.5) from §4.4 is mandatory, not mode-gated — under a
  shared barrier, simultaneous neighbor commits are always the norm, so this
  is simply how Layer B commits, unconditionally. Hysteresis (§4.3 of the
  original spec) stays exactly as specified.
- `test_phase_locked_commit` becomes: with `p_commit = 1` (deliberately
  disabling the randomized-commit countermeasure) the adversarial scenario
  must exhibit replica thrashing; with default `p_commit = 0.5` plus
  hysteresis, the same scenario must settle within 20 slow ticks. There is no
  "jitter off" variant to test since there is no jitter.
- Every acceptance criterion in the test plan (§8 of the original spec:
  `test_dead_neighbor`, `test_oscillation_hysteresis`,
  `test_phase_locked_commit`, LP-convergence, dSB-vs-brute-force) is evaluated
  once, under the single execution model.
- Every cross-method comparison in M5/M6 (§4 below) reports a single PLASMA
  row/curve against MILP/greedy/FaaS-MADeA, at matched message budget (tune
  `W` vs. each baseline's own round period, per §6.1 of the original spec).

## 3. Integration with `run.py` and `compare_results.py`

Following the `hierarchical-madea` precedent (`run.py` line 37:
`"hierarchical-madea": ("LSPc", "HierarchicalMADeA")`), register one new
method, additively, in `run.py`'s dispatch tables (method list, result-label
mapping, folder bookkeeping) — the same touch points already extended for
`hierarchical-madea`, none of the existing entries changed:

```python
"plasma": ("LSPc", "Plasma"),
```

(The first element of a `METHOD_RESULT_MODELS` entry is the result-model key
consumed by `load_models_results` at `run.py:317` — `"LSPc"` for every
decentralized method — not a config class name. PLASMA writes LSPc-format
result logs so it plugs into the existing loaders unchanged.)

"Additively" means the same ~7 touch points `hierarchical-madea` needed, all
new lines next to existing ones, none changed: the runner import (cf. line 7),
the `--methods` choices list (line ~79), the `METHOD_RESULT_MODELS` table,
the solution-folder bookkeeping dict (line ~901), the resume/skip logic
(line ~970), the per-experiment run flag (line ~1030), and the invocation +
folder registration (lines ~1143–1150).

Invocation mirrors every other method:

```bash
python run.py -c config_files/plasma_comparison.json \
  --methods centralized faas-macro plasma \
  --n_experiments 3 --loop_over Nn
```

`compare_results.py` needs no changes: it already compares an arbitrary
`--models` list by result label, so `Plasma` slots in next to
`LoadManagementModel`/`FaaS-MACrO`/`HierarchicalMADeA`.

`postprocessing.py`/`logs_postprocessing.py` gain new parsing functions only
for what doesn't exist (regret vs. oracle, adaptation lag, msgs/node/s) —
added as new functions in `plasma/eval/regret.py` plus, if genuinely needed,
new (not modified) functions alongside the existing `parse_*_log_file`
functions in `logs_postprocessing.py`.

## 4. Milestones (repo-aligned)

1. **M1 — Skeleton + scheduler.** `plasma/core/types.py`, `plasma/sim/clock.py`
   (shared round barrier only), empty `PlasmaNode` that rejects everything.
   Metrics pipeline (reusing `postprocessing.py`) works end-to-end on a
   3-node line generated via `generators/generate_data.py`.
2. **M2 — Layer A alone**, replicas fixed by config, on the round barrier. LP
   convergence criteria on a 10-node ER graph (`neighborhood: {"p": ...}`)
   from existing topology generation, stationary traffic from
   `load_generator.py`'s existing `clipped` trace type (a pure-Poisson trace,
   if needed for the paper, is a new function in `plasma/sim/`, per §1.2).
3. **M3 — Baselines wired.** `milp_baseline.py` over `models/sp.py`,
   `greedy_baseline.py` over `HeuristicCoordinator`/`GreedyCoordinator`,
   `madea_iface.py` adapter. Gap metric vs LP relaxation.
4. **M4 — Layer B alone**, synthetic `benefit` fields, dSB vs brute force
   (≤12 spins/node).
5. **M5 — Coupling + hysteresis.** Oscillation test, plus the mandatory
   randomized-commit countermeasure (`p_commit`) test from §2. Full pipeline
   vs MILP on a 20-node G(n,m) graph (`neighborhood: {"m": ...}`; BA is not
   supported by `generate_neighborhood` and is not added): target gap ≤ 10%
   stationary.
6. **M6 — Non-stationarity + failures.** MMPP-style bursts (via existing
   trace types or a small new generator if needed), node kill/revive,
   adaptation-lag metric, comparison table PLASMA / greedy / stale-MILP /
   FaaS-MADeA.
7. **M7 — Rare-function mode + parameter sweeps.** Reuse `run.py`'s existing
   `loop_over`/sweep machinery instead of a new sweep runner.

## 5. LaTeX note

The repo documents each method as a `faas-<name>-note/` LaTeX package
(`faas-magcaa-note/`, `faas-mald-note/`, `faas-mapg-note/`,
`faas-mapod-note/`, `faas-bestresponse-note/`, `faas-madig-note/`), each with
a `main.tex` wrapper (`\input{faas-<name>}`) plus a `faas-<name>.tex` body and
`references.bib`. Create `faas-plasma-note/` the same way:

```
faas-plasma-note/
  main.tex          # \input{faas-plasma}
  faas-plasma.tex    # body: problem statement, Layer A/B dynamics, Hamiltonian,
                      #        round-barrier formulation, acceptance criteria,
                      #        privacy comparison vs FaaS-MADeA's bid exchange
  references.bib
```

Populate incrementally starting at M3 (once there's a baseline gap number to
report) and keep it current through M7, replacing the original spec's
`NOTES.md` (§10) — this repo's convention for paper-facing notes is the LaTeX
package, not a markdown scratch file.

## 6. What is unchanged from the original spec

Sections 2–5 (problem model, Layer A dynamics, Layer B Hamiltonian/dSB,
communication protocol), the open-parameter list (§9 there, minus
`tick_jitter` which no longer applies), and the test plan's algorithmic
content (§8 there: locality invariant, capacity gate, RAM repair,
dSB-vs-brute-force, LP convergence, dead-neighbor decay, oscillation
hysteresis, message budget, phase-locked commit) all carry over verbatim —
only their file locations move per §1.3 above, and every test now runs under
the single round-barrier execution model per §2 (no async variant to test
against).
