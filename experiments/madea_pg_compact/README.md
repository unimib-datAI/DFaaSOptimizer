# MADEA-PG compact proposal experiments

Use the existing virtual environment from the repository root. Gurobi requires a working local license; it only solves node-local initialization models.

```sh
MPLCONFIGDIR=/tmp/dfaas-mpl XDG_CACHE_HOME=/tmp/dfaas-test-cache .venv/bin/python benchmark_madea_pg_compact.py --help
MPLCONFIGDIR=/tmp/dfaas-mpl XDG_CACHE_HOME=/tmp/dfaas-test-cache .venv/bin/python benchmark_madea_pg_compact.py --seconds 1800 --output solutions/madea-pg-compact-planar-reproduce
.venv/bin/python experiments/madea_pg_compact/summarize.py solutions/madea-pg-compact-planar-reproduce
.venv/bin/python experiments/madea_pg_compact/plot.py solutions/madea-pg-compact-planar-reproduce
MPLCONFIGDIR=/tmp/dfaas-mpl XDG_CACHE_HOME=/tmp/dfaas-test-cache .venv/bin/python experiments/madea_pg_compact/temporal.py solutions/madea-pg-compact-planar-temporal 780
```

See `--help` for output/config/case overrides. `config.json` and both case lists use connected cubic planar graphs. Generated instances, detailed solutions and logs stay under ignored `solutions/`. The small measured results in `results_2026_10_01/` preserve the paired statistics, source provenance, plot and report.

The static benchmark compares the pre-compaction proposal with the compact proposal, **both using the shared native DP backend**. It freezes the initialization for each pair, neutralizes replayed solver counters and adds measured initialization wall time back to the total. It checks per-move prefix equality, feasibility and exact final allocations when neither run is time-censored. The temporal experiment uses three real production timesteps, without initialization replay, neutralized counters or per-move trace observers.

The first static batch was stopped to prioritize ten-function and load-stress cases. One duplicate orphan run was excluded; the combined table retains one successful orphan and one timeout, excluded from paired statistics. Timing is from one machine and one repetition per mode/case. The source snapshot precedes the later missing-input validation guard; 240 proposal oracle comparisons passed again after that guard.

The multiprocessing `-j` path uses the shared DP backend for independent node-local subproblems. The shared runners now keep one pool alive for the complete run. Each chunk carries the current full input snapshot, model, solver options and prices. PG refinements still apply sequential moves; no parallel PG commits or new global decision mechanism were introduced.

The measured `pool_microbenchmark.json` uses the actual `solve_single_agent` worker on the materialized temporal `n80-f10-s7` instance. Run `pool.py` from the repository root after generating that instance at its recorded path. It checks exact x/r/omega/z equivalence and separates startup from two warm batches. Its results isolate local solve/IPC; inputs are unchanged between batches, so changing-input serialization and the rest of the pipeline are not benchmarked.

`warm_pool.py` measures the integrated `solve_subproblem` with three changing-input
batches and prices, including snapshot transfer and result merging. `warm_pool_runner.py`
compares the full two-step MADEA-PG runner with persistent and recreated pools.
They use the temporal planar instances at the recorded paths; run from the repository
root with the same environment variables above. Small follow-up results are saved in
`results_warm_pool_2026_10_01/`; large solutions stay under ignored `solutions/`.
The cold runner comparison took about 164 seconds on the recorded machine.

`plasma_parallel.py` compares full PLASMA-Welfare runs with `-j 0`, `-j 2` and
`-j 4` on the four common planar instances used in `compare_families.py`.
It repeats each mode twice, rotates execution order and checks exact allocations,
welfare, messages and accepted transactions. Wall time includes pool startup and
shutdown; all runs use 20 rounds, with a five-minute experiment limit:

```sh
MPLCONFIGDIR=/tmp/dfaas-mpl XDG_CACHE_HOME=/tmp/dfaas-test-cache .venv/bin/python experiments/madea_pg_compact/plasma_parallel.py
```

The 24 completed runs are recorded in `results_plasma_parallel_2026_10_01/`.
Allocations, welfare, messages and accepted transactions match exactly across
all modes and repetitions. `summary.csv` reports median full-run times by case.
The last two measurements briefly overlapped final verification; this small
sample is indicative rather than a controlled hardware scaling study.

See [the Italian summary](../../docs/MADEA-PG-miglioramenti.md) for the final behavior,
measured gains and full-suite validation.

## Extended planar family comparison

`families_extended.py` compares MADEA, MADEA-one-shot, MADEA-PG and
PLASMA-Welfare on identical materialized inputs. Use `--deadline` with a future
UTC ISO timestamp to cap the experiment, `--workers 0 --plasma-workers 4` for
the measured configurations, and `--output` for the generated solutions.
The plan covers four seeds, 40/80/160 nodes and 5/10 functions, followed by
load stress, three-timestep cases and timing repeats if the deadline allows.
Different algorithms retain their native stopping rules.

```sh
.venv/bin/python experiments/madea_pg_compact/families_extended.py --help
MPLCONFIGDIR=/tmp/dfaas-mpl .venv/bin/python experiments/madea_pg_compact/families_report.py solutions/families-planar-hour-2026-10-01-adaptive experiments/madea_pg_compact/results_families_extended_2026_10_01 --pilot solutions/families-planar-hour-2026-10-01-main
```

The [Italian report](results_families_extended_2026_10_01/report.md) preserves
the tables, plot, feasibility checks and source hashes. Partial cases are
excluded from comparisons; timing repeats never count as extra welfare seeds.
The four-process pilot is kept separately because transferring full inputs
made the MADEA runners slower than their sequential DP on these instances.

`compare_one_shot_pg.py --seconds 300` reuses the same validated harness for
eight paired planar inputs (40/80 nodes, 5/10 functions, seeds 7/42), comparing
one-shot, the new `one-shot-pg`, and MADEA-PG in sequential mode. Its
[Italian report](results_one_shot_pg_2026_10_01/report.md) records full-run
times, feasibility, initial/final welfare, source hashes and suite failures.

`extend_one_shot_pg.py --seconds 600` runs only the new algorithm on the 35
completed inputs of the original extended comparison. It reads the original
materialized instances directly and merges the new measurements with the
previous four methods. The [five-method Italian report](results_families_five_methods_2026_10_01/report.md)
includes the new column without changing the earlier measured results:

```sh
MPLCONFIGDIR=/tmp/dfaas-mpl XDG_CACHE_HOME=/tmp/dfaas-test-cache .venv/bin/python experiments/madea_pg_compact/extend_one_shot_pg.py --seconds 600
MPLCONFIGDIR=/tmp/dfaas-mpl .venv/bin/python experiments/madea_pg_compact/families_report.py solutions/families-five-methods-2026-10-01 experiments/madea_pg_compact/results_families_five_methods_2026_10_01 --pilot solutions/families-planar-hour-2026-10-01-main
```

The same extension script also measures `hierarchical-one-shot-pg` on the
35 original materialized inputs, adding a sixth column to the five-method
data. Previous measurements stay unchanged; only the requested variant runs.
The hierarchy uses its existing iterative engine, depth 3 and direct-neighbor
flows. Runs use the original general limit `max(30, 0.5*N)` and PG limit
`0.25*N`, clipped to remaining time. The updated algorithm reserves the requested
PG time, capped at half the total limit. Two unchanged assignment/replica rounds
with price and weighted fairness changes within auction epsilon plus tolerance
stop the outer loop. This is a local-state heuristic, not a convergence certificate.
Zero PG time or sweeps disables both changes. Current rounds/proposals may finish
beyond those soft limits.

```sh
MPLCONFIGDIR=/tmp/dfaas-mpl XDG_CACHE_HOME=/tmp/dfaas-test-cache .venv/bin/python experiments/madea_pg_compact/extend_one_shot_pg.py --method hierarchical-one-shot-pg --source solutions/families-five-methods-2026-10-01 --output solutions/families-hierarchical-balanced-2026-10-01 --merged solutions/families-six-balanced-2026-10-01 --seconds 1800
MPLCONFIGDIR=/tmp/dfaas-mpl .venv/bin/python experiments/madea_pg_compact/families_report.py solutions/families-six-balanced-2026-10-01 experiments/madea_pg_compact/results_hierarchical_balanced_2026_10_01 --pilot solutions/families-planar-hour-2026-10-01-main
```

The [first six-method Italian report, before reservation and stagnation stopping](results_families_six_methods_2026_10_01/report.md)
contains all 35 cases and preserves the previous five columns. The 30-minute
window finished 34 cases; one remaining three-step case was completed in
60.23 seconds in a separate session with unchanged code and inputs. The
completion driver is preserved as `completion_snapshot.py.txt` in the report
folder (run from the repository root if exactly that case remains missing).
The interrupted pre-fix pilot is excluded. The report and `verification.json`
record this split, source hashes, incumbent improvement checks and the known
full-suite failure. No other method was rerun for the sixth column.

The [updated six-method Italian report](results_hierarchical_balanced_2026_10_01/report.md)
reruns only the hierarchy with stagnation stopping and PG reservation, preserving
the previous five columns. All 35 cases / 41 timesteps completed in 611.07 seconds
including validation. The pre-PG incumbent matches the previous hierarchy in every
timestep; PG now improves all 41. Standard-case welfare improves by 9.95% per
instance on average over the previous hierarchy, with median speedup 4.90×.
Against one-shot-PG, standard-case welfare improves by 7.42%, with median time
ratio 1.29×. The report includes before/after CSVs and source snapshots.

`ablate_hierarchical_madea_pg.py` measures the current hierarchical MADEA-PG
runner on eight fixed materialized planar inputs from the earlier comparison.
The [Italian ablation report](results_hierarchical_madea_ablation_2026_10_01/report.md)
preserves measurements and snapshots for time reservation, elapsed-time
accounting and two discarded stagnation variants. Only elapsed-time accounting
and PG reservation remain, gated to positive PG time and sweeps. Non-PG and
disabled-PG paths retain their earlier behavior; local solver options are unchanged.
Current work may finish beyond these soft time limits.

```sh
MPLCONFIGDIR=/tmp/dfaas-mpl XDG_CACHE_HOME=/tmp/dfaas-test-cache .venv/bin/python experiments/madea_pg_compact/ablate_hierarchical_madea_pg.py --output solutions/hierarchical-madea-ablation-repeat --seconds 900
```

This reruns the current code only. Historical trial snapshots in the report
folder document the removed variants; no independent holdout was measured.

`rerun_alpha_gt_beta.py --seconds 1800` creates corrected copies of all 35
previously materialized instances, using beta/alpha in `[0.1, 0.9]` with the
same generation seed. It verifies every physical parameter and the exact graph,
and copies the original request traces and load limits unchanged. Delta is
regenerated consistently because its configured multiplier derives it from beta.
Original inputs and their results remain available for reproducibility.
The driver measures all seven methods on twelve fixed planar cases, sequentially,
with a 30-minute deadline including instance preparation. It exports the corrected
config, checksums and an audit of the economic constraint. Old absolute welfare
values use different coefficients and must not be merged into this new table.

```sh
MPLCONFIGDIR=/tmp/dfaas-mpl XDG_CACHE_HOME=/tmp/dfaas-test-cache .venv/bin/python experiments/madea_pg_compact/rerun_alpha_gt_beta.py --seconds 1800
```

The [corrected seven-method Italian report](results_families_alpha_gt_beta_2026_10_01/report.md)
contains 84 successful runs / 112 timestep results across twelve cases, completed
in 22.07 minutes including preparation. All 35 corrected instances satisfy
alpha > beta, with unchanged graphs, resources and traces. On eight standard
cases, mean gains over MADEA are 0.58% for MADEA-PG, 0.84% for one-shot-PG and
1.02% for hierarchical one-shot-PG. The report separates stress and temporal
results, includes absolute welfare/times and records the known suite failure.
