# VNS-R and standalone VNS-RX

VNS-R is a separately named fixed-x VNS with replica-aware insertion/rerouting
and atomic relocation plus insertion. The original `centralized_vns` is unchanged.
Unit/full-flow candidates, two-flow exchange and seeded shaking remain available.
VNS-RX is an independent CPU refinement of any compatible feasible integer x/y
allocation, including selfish-distributed. It does not require VNS-R to run first.

Use the existing solver environment and CPU/Metal/CUDA dependencies documented
in [the original VNS README](../centralized_vns/README.md). From the solver root:

```bash
.venv/bin/python -m centralized_vns_r /path/to/instance \
  --output /path/to/new-run --backend metal --pool-size 64 --candidates 512 \
  --iterations 100000 --time-limit 15 --max-device-mib 2048 \
  --refine-x --x-iterations 100000 --x-time-limit 15
```

The fixed stage saves `fixed_x/solution.npz` and `fixed_x/result.json` before
refinement starts. The variable stage saves `variable_x/solution.npz` and
`variable_x/result.json`. Both results, their arrays, local guarantees and timing
are preserved; `comparison.json` records their welfare. Omit `--refine-x` to run
only VNS-R. Output paths must be new. Inputs and old results are never overwritten.

## Plateau stopping

VNS-R now enables a rolling plateau window by default: `--patience 200`
iterations and `--plateau-tolerance 1e-6`. It stops only when **every trajectory**
has archive improvement `(best_now - best_then) / max(1, abs(best_then))` no
greater than the tolerance across that window. Below unit magnitude this is an
absolute tolerance. The initial archive counts as the window's starting point.
Small gains accumulate across the window; a flat global leader alone cannot stop
other trajectories that are still making significant progress. Shaking losses
do not lower the archived best.

`--patience 0` disables plateau stopping to reproduce the earlier budget-only
behavior; `--plateau-tolerance 0` requires exactly unchanged archives. Positive
patience must be at least `3 * shake_after + 1` (10 with the CLI defaults), allowing
one fully stalled neighborhood/shaking cycle. The default window allows 20 such
cycles, but does not certify that every trajectory actually performed a shaking.
These defaults are not calibrated for a particular instance or GPU.

Time and iteration limits still apply and take precedence over a simultaneous
plateau. `result.json` records `termination_reason` (`plateau`, `time_limit`, or
`iteration_limit`), `iterations_completed`, and the `plateau` settings, including
for Python callers: `solve(p, options, patience=200, plateau_tolerance=1e-6)`.
Stopping is shared by CPU/Metal/CUDA and uses the archive values already returned
each iteration, with O(pool_size * patience) host memory (about 0.4 MiB for 256
trajectories and the default window), without extra GPU transfers.

This is a stagnation heuristic, not an optimality certificate. Compare enabled
and disabled runs with the same seed and budget before selecting experimental
thresholds. The independent x stage retains its existing stop after a complete
proposal pass without an accepted improvement, or its own time/attempt limit.

## Apply x refinement to another algorithm

No local DP or fixed-x VNS is run by this command:

```bash
.venv/bin/python -m centralized_vns_r.refine /path/to/materialized-instance \
  --solution /path/to/selfish-distributed-output \
  --output /path/to/new-selfish-refined --step 0 --solution-row 0 \
  --iterations 100000 --time-limit 15 --seed 4850
```

`--solution` accepts either:

- NPZ with `x` and dense `y[source,destination,function]`, or sparse `y` with
  explicit `(source,destination,function)` `edges`. Optional `z` and `r` or
  `replicas` are verified if supplied. A VNS `solution.npz` works directly.
- Native experiment folder with `LSPc_solution.csv`, `LSPc_replicas.csv`, and
  `LSPc_detailed_fwd_solution.csv`; `.csv.gz` works too. Both ordinary flow column
  names and the `_tot` schema used by the existing decoder are supported.

`--step` selects the instance's request-trace step. `--solution-row` independently
selects a zero-based data row from the CSV output (not its header). Supply the
matching step and row. Known adjacent NPZ `result.json` provenance is checked
against instance hashes/step; CSV and bare NPZ imports cannot prove historical
lineage and must be paired with the correct instance by the caller.

All supplied decisions must be nonnegative finite integers, allowing at most
1e-6 rounding noise on import. Fractional allocations outside that tolerance are
rejected. Reconstructed conservation, permitted edges, sender/receiver roles and
memory are independently checked. Imported replica allocations may reserve more
than necessary but must be physically feasible; search uses minimal replicas.

## Guarantee and search scope

At stage entry, capture for each node n:

```
floor[n] = sum_f alpha[n,f] * x_initial[n,f] / max(load[n,f], 1)
```

Every accepted transaction must preserve all floors within absolute float64
tolerance 1e-12 and increase global welfare by more than 1e-12. Floors never
follow intermediate x changes. In the integrated pipeline x_initial is the DP
plan; standalone, it is the imported solution's x. This is the x-only guarantee
of selfish-centralized, not a guarantee on total per-node forwarding utility.
Using the input plan avoids rerunning DP with potentially different tied optima.

The stage samples local admissions/localization, single-function release and
paired function reallocation, including replica thresholds. A receiver insertion
repair may make an otherwise neutral local change globally useful. Each proposed
transaction is fully checked. The input incumbent remains available if nothing
improves. `iterations` counts proposal/repair attempts, not sweeps. Deadline checks
occur between attempts; a single feasibility check can overrun the budget.

VNS-R supports CPU, real Metal and CUDA. VNS-RX currently uses CPU explicitly;
there is no hidden CPU fallback of the fixed-stage GPU backend. Metal uses GPU
integer candidate generation and CPU float64 boundary tables/selection. Its dense
load and replica-limit tables can exceed `--max-device-mib` on extreme inputs.
That cap covers estimated persistent device arrays, not host copies or runtime.

The two stages have separate time/iteration budgets, including engine setup but
excluding instance loading and initial DP. An RX result includes extra compute;
compare methods at equal total budgets when claiming quality advantages.
New neighborhoods and x changes alter search trajectories. CPU/backend parity
is checked within VNS-R, not between original VNS and VNS-R. CUDASIM does not
prove real CUDA compilation or speedup. No remote job is launched by these tools.
