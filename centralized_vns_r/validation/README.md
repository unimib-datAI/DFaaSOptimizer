# VNS-R / VNS-RX validation, 2026-11-17

## Completed plateau pool campaigns: 120 and 300 seconds

Three 500-node inputs, two seeds, pools 64/128/256/512 and 512 candidates per
trajectory. Rolling window 200 iterations, relative tolerance 1e-6. All actual
runs use the existing hybrid Metal implementation on the MacBook. These are
stop-rule endpoint comparisons, not equal-time quality comparisons.

Both stop rules have 24 records at each safety cap. Extending to 300 seconds
reran the 17 individual-record cases previously capped by time, and reused the
seven already stopped by plateau. All 24 global-best cases had already stopped
before 120 seconds and were reused explicitly; their timings were not remeasured.
Thus the four campaigns contain 96 logical records and 65 distinct measured runs.

Mean endpoint welfare and elapsed seconds for individual-record plateau at the
300-second cap (two seeds per cell, larger welfare is better):

| Pool | Cubic welfare | Seconds | Melbourne welfare | Seconds | IIoT welfare | Seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 10191.41951 | 64.81 | 9975.62814 | 38.08 | 18275.28541 | 87.06 |
| 128 | 10191.54121 | 215.69 | 9979.42765 | 122.75 | 18275.28541 | 300.01 |
| 256 | 10192.35049 | 300.01 | 9982.15878 | 300.02 | 18275.77755 | 300.01 |
| 512 | 10192.35049 | 300.03 | 9983.39640 | 300.02 | 18273.57823 | 300.05 |

The larger pool does not uniformly improve quality: at this cap cubic ties at
256/512, Melbourne favors 512 on average, and IIoT favors 256. IIoT pool64 is
within 0.49214 welfare of pool256 while stopping much earlier. Nine of the 24
individual endpoints improved from the 120-second campaign; ten of 24 now stop
by plateau, and fourteen still reach the safety cap.

The best single cubic result is 10192.58869291, 0.033005% below the previous
matched campaign best of 10195.95390457 (not a proven optimum). Only cubic has a
confirmed effective-input match to that historical campaign; do not infer the
same historical comparison for Melbourne or IIoT.

Aggregate stop-rule trade-off across all 24 cases at each cap:

| Cap | Stop rule | Mean seconds | Mean paired welfare delta vs all records | Plateau stops |
| ---: | --- | ---: | ---: | ---: |
| 120 s | All individual records | 105.51 | reference | 7/24 |
| 120 s | Current top 10%, replay | 64.87 | -0.02053% | 16/24 |
| 120 s | Global best | 26.49 | -0.04218% | 24/24 |
| 300 s | All individual records | 219.05 | reference | 10/24 |
| 300 s | Current top 10%, replay | 105.07 | -0.02281% | 21/24 |
| 300 s | Global best, reused | 26.49 | -0.05443% | 24/24 |

Top10% is an offline replay, not a new timed solver mode. At each iteration it
selects ceil(0.1*pool) current archive leaders (stable index ties), comparing each
to its own record 200 iterations earlier. Times are original trace timestamps;
ranking overhead is not measured and no early-solution NPZ is reconstructed.
At 300 seconds it would save 52.03% aggregate time, preserve welfare in 11/24
cases, and lose at worst 0.13346%. Global-best saves 87.91% aggregate time relative
to individual-record stopping, at worst losing 0.16000% welfare. These results
support testing an elite policy; two seeds do not establish a final policy.

Independent verification reconstructed all 65 winning solutions and **16,768**
distinct population archives, checking physical feasibility and objective values.
It reconstructed every first plateau iteration, checked 58 distinct same-pool
search-prefix pairs and 53 first64 cross-pool prefix pairs exactly, verified
all source/input provenance and counted 31 reused records explicitly. All passed.
The six-panel welfare/time figure was visually inspected.

Full verified metrics: [macbook-pool-plateau-300s.json](macbook-pool-plateau-300s.json).
Scripts, source snapshots, archives and figure live in root
`output/vns-r-pool-plateau-300s-20261117/`; both elite replays in
`output/vns-r-pool-elite-replay-20261117/`. The separate
[`centralized_vns_spark` CUDA runner](../../centralized_vns_spark/README.md) was
implemented afterward in a new package; none of these timings measures that runner.

## Pool comparison at a fixed 15-second budget

24 sequential real-Metal runs: the same three 500-node inputs, seeds 4850/4851,
512 candidate slots per trajectory, pool sizes 64/128/256/512. Plateau disabled
for this comparison. The budget includes engine/table setup and search; JIT,
input loading/initial DP and diagnostics are excluded. Pool order varies by seed
and family. Actual elapsed times: 15.0007–15.0673 s, because limits are checked
between complete iterations. Ranking is unchanged using only trace values at or
before 15 s. Means of two seeds, larger welfare is better:

| Pool | Cubic welfare | Melbourne welfare | IIoT welfare |
| ---: | ---: | ---: | ---: |
| 64 | 10190.568 | 9975.097 | 18250.378 |
| 128 | 10189.204 | 9977.904 | 18235.122 |
| 256 | 10188.696 | 9979.263 | 17804.814 |
| 512 | 10186.869 | 9957.941 | 17284.000 |

Each performance cell gives million candidate slots per elapsed second / mean
milliseconds per search iteration. Counts include infeasible and empty slots;
they are not accepted moves or GPU utilization.

| Pool | Cubic: Mcand/s / ms | Melbourne: Mcand/s / ms | IIoT: Mcand/s / ms |
| ---: | ---: | ---: | ---: |
| 64 | 5.54 / 5.85 | 5.43 / 5.97 | 3.22 / 9.83 |
| 128 | 5.77 / 11.23 | 5.73 / 11.31 | 2.96 / 21.42 |
| 256 | 5.61 / 23.10 | 5.53 / 23.44 | 2.87 / 43.82 |
| 512 | 4.90 / 52.91 | 5.23 / 49.57 | 2.90 / 87.24 |

At this budget, 64 wins on cubic/IIoT; 256 wins on Melbourne. Larger pools can
improve the best over their first 64 trajectories at the same iteration, but
the reduced search depth can outweigh this at equal wall time. No universal
pool optimum is established by two seeds and one instance per family.

All final archives within each run are distinct, with sampled mean normalized
flow distances 0.503–0.736 (0 identical, 1 disjoint forwarding allocations).
These are endpoint measurements, not proof of disjoint search histories/basins.
They do not support adding diversified starts merely to eliminate duplicates.
All 24 winners and all 5760 archived allocations were independently reconstructed
and verified, including welfare and constraints. Source/input hashes match.

A separate 30-step cubic probe starting from saved populations attributes 20.7%
(pool64) and 24.9% (pool512) of step time to CPU acceptance/application alone.
Other CPU work includes scoring and replica boundaries. This is not an occupancy
measurement and does not establish GPU saturation; CUDA on Spark differs.

Metrics/source hashes: [macbook-pool.json](macbook-pool.json). Full scripts,
source snapshots, populations, traces and plot: root `output/vns-r-pool-20261117/`.
No production solver code was changed for these measurements. The follow-up
campaigns with plateau stopping are separate from this fixed-budget comparison.

## Plateau follow-up

Default rolling window: 200 iterations; tolerance: 1e-6 for every trajectory's
archive. Final combined VNS/VNS-R tests on real Metal: **114 passed, 18 skipped**,
7.20 s. The focused CUDA-simulator plateau parity test passed (5.08 s). Ruff and
diff whitespace checks pass. Review found a CLI validation/output-creation issue;
shared early validation and two regression tests fixed it. No remaining findings.
Original VNS source SHA256s still match the pre-VNS-R snapshot.

On cubic500 with 64 trajectories, 512 candidates and two seeds, enabling the
default plateau preserved all 3000 iterations and exact per-trajectory archive
values: ongoing progress correctly prevented early stopping. Final welfare was
10192.112295497813 and 10189.1556038792 with either setting.

Starting from the previous seed4850 fixed-x incumbent (welfare 10192.112295497813),
the plateau stopped at iteration **861 rather than 1000**, with identical final
welfare **10193.736625809699**. Observed wall times were 5.04 and 6.37 s; these are
single smoke measurements, not a speedup estimate. Both paths had identical
archive prefixes. Six saved solutions were independently checked for feasibility,
welfare, supplied replicas/rejections and local floors; the first iteration
satisfying the stop condition was independently reconstructed from the archives.

Metrics, verification and source hashes: [macbook-plateau.json](macbook-plateau.json).
Full traces, archives, scripts and source snapshots:
root `output/vns-r-plateau-20261117/`. This is not calibration across families,
candidate counts or DGX Spark. The broad repository suite was not rerun; its
previous baseline failures remain documented below. The earlier measurements in
this file predate plateau stopping; use `--patience 0` to reproduce that policy.

## Initial VNS-R implementation

Existing `centralized_vns/*.py` and `.metal` match their SHA256 snapshot from
before this task. Both old and new commands remain independently available.
No input instances, prior results, shared methods or central models were changed.

Tests: `.venv/bin/python -m pytest centralized_vns/tests centralized_vns_r/tests -q`
on real Apple Metal: **93 passed, 17 skipped**, 6.03 s. CUDA is unavailable in the
main Mac environment. New backend tests in temporary Numba 0.63.1 CUDASIM:
**11 passed**, 69.77 s. The new real-Metal backend subset independently passed
11 tests; 10 randomized mixed-memory/capacity problems × 30 iterations also
matched CPU candidates, states and welfare exactly.

Independent review found no substantive issue. Fifty additional small randomized
x-refinement cases satisfied reconstructed physics, local floors, nondecreasing
welfare and input immutability. Imports are tested for sparse edge reordering,
non-permitted flows, inconsistent supplied decisions, duplicate canonical CSV
columns, missing rows, native CSV/gzip and command-level output preservation.

The broader repository is not certified green: prior baseline failures and the
interrupted broad run remain documented in `centralized_vns/validation/README.md`.
An initial new-test filename collision was resolved by adding the tests package;
the combined module suite above then completed successfully. Ruff and diff
whitespace checks pass. GitNexus does not index the new module; a bare symbol
lookup resolved an unrelated postprocessing function. Scope was checked directly.

## Three 500-node inputs

Same MacBook M3 Pro (11 CPU cores, 14 GPU cores, 18 GB unified memory) and local
catalog instances used in the preceding VNS work. Two seeds: 4850/4851. Pool 64,
C=512; warm JIT, 15 s fixed-stage budget including engine/table setup, input
loading and DP excluded. Original/new fixed-stage order reverses by seed.
RX then gets a **separate** 15 s CPU budget (100,000 transaction/repair attempts).
Thus RX is not an equal-total-time competitor to either 15 s fixed-stage run.

Mean welfare over the two seeds, larger is better:

| Family | Original VNS | VNS-R fixed x | VNS-RX | RX gain over R |
| --- | ---: | ---: | ---: | ---: |
| Planar cubic | 10189.679902 | 10190.567683 | 10190.929311 | +0.361628 |
| Melbourne | 9981.599591 | 9975.096607 | 9980.418850 | +5.322243 |
| IIoT | 18271.043382 | 18250.378314 | 18346.257656 | +95.879342 |

The new fixed-stage neighborhoods do not uniformly beat the original at this
budget; the original remains available and has not been replaced. This is a
small exploratory comparison, not a statistically established ranking. More
seeds, held-out inputs and equal total-time comparisons are needed for a campaign.

RX improved all six fixed-stage starts. It changed local x in 2–21 nodes per run;
all 500 per-node floors were satisfied in every run (minimum margin 0, no deficit).
Cubic refinement stopped after approximately 3.8 s; Melbourne/IIoT used about
15 s. Fixed-stage artifacts remained byte-identical after refinement.

On cubic500, pool4/C64/30 iterations, CPU and real Metal matched every generated
candidate, incoming/rejected/flow cache and per-trajectory archive exactly.
All 18 saved original/fixed/variable solutions were independently reconstructed
and checked against input hashes, welfare, physical constraints and local floors.
The standalone CLI on the saved cubic seed4850 fixed solution reproduced the
integrated RX x/y/z/replicas/floor arrays exactly, without rerunning DP or VNS.

Compact raw metrics/source hashes: [macbook-500.json](macbook-500.json).
Scripts, original-source hash snapshot, full traces, immutable separate stage
artifacts and standalone result: root `output/vns-r-20261117/`.

Limits: the x stage is explicitly CPU and uses a bounded heuristic neighborhood.
The import adapter was tested on native-format fixtures; no archived historical
selfish-distributed run was available locally for a full end-to-end replay.
Real CUDA compilation, timings and tuning on Spark remain unverified. The input
floor protects only alpha-weighted local processing, not complete node utility.
