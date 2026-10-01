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
