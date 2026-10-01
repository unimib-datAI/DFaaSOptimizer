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

The multiprocessing `-j` path already uses the shared DP backend for independent node-local subproblems. A persistent process pool with compact own-node payloads is the next parallelism experiment. PG refinements currently apply sequential moves: parallel precomputed proposals would need input-version validation and recomputation after conflicting changes to preserve that behavior. No parallel PG commits or new global decision mechanism were introduced in this experiment.

The measured `pool_microbenchmark.json` uses the actual `solve_single_agent` worker on the materialized temporal `n80-f10-s7` instance. Run `pool.py` from the repository root after generating that instance at its recorded path. It checks exact x/r/omega/z equivalence and separates startup from two warm batches. Its results isolate local solve/IPC; inputs are unchanged between batches, so changing-input serialization and the rest of the pipeline are not benchmarked.
