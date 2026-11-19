# Local validation, 2026-11-17

This is a new CUDA runner. No real CUDA GPU, DGX Spark compilation, throughput or
occupancy measurement was available on the Mac. No separate phase profiler was
created. Previous solver source hashes remain unchanged.

## Checks performed

| Check | Result |
| --- | --- |
| New package in CPU/Metal environment | 88 passed, 26 optional-backend skips |
| Fixed-x CUDA simulator tests | 23 passed, 85.03 s |
| X-stage CUDA simulator tests | 60 passed, 13.80 s |
| Calibration tests in Numba environment | 23 passed, 0.14 s; mocked hardware/timing |
| Ruff | passed |
| Original VNS and previous campaign source SHA256s | unchanged |

Fixed-x tests compare every generated move, flow/incoming/rejection cache,
objective/archive, neighborhood level and stale count to the CPU reference over
warm-started trajectories. They cover all three block widths, non-multiple
candidate counts, replica boundaries, all/global rolling stops, early stop masking
inside queued batches, empty graphs, memory/options and nonfinite objectives.

X-stage tests cover swaps, localization, receiver repairs, reward-ordered trimming,
local-gain floors, strict welfare improvement, input immutability, independent
scalar neighborhood enumeration, constrained batch equivalence, coalesced views,
dynamic hardware-sized batch bounds and integer conversion guards. The review's
large-plus-small floating-point floor regression now passes. Pairwise reductions
are checked against NumPy across lengths 0–8193 and runtime buffer sizes 16,
32 and 8192, for both node-wise and global sums.

Calibration is tested for memory exclusions including production plateau history,
mean-welfare selection across seeds, independent checking, throughput ties, absent
eligible configurations, excessive time-overrun exclusion and simulator refusal.
Those tests do not measure real calibration performance.

An end-to-end CUDASIM command starts from a supplied NPZ, saves separate fixed
and variable artifacts, then independently refines the fixed NPZ. Fixed welfare
is 1.0 and both integrated/standalone variable welfare are 2.6666666666666665.
All three outputs are independently reconstructed, local floors pass and the
fixed artifact remains byte-identical. This validates command plumbing without
rerunning DP; native-format import uses the existing tested adapter.

## Review corrections and remaining limits

Independent review identified NumPy/device summation-order differences, unfair
initial calibration batches, nonfinite objective handling and overstated shaking
candidate counts. Regression tests and fixes address these. The initial batch is
now one iteration; excessive calibration overruns are excluded and recorded.
Every counted candidate is generated and scored, including shaking candidates.
An additional pre-conversion ratio check prevents out-of-range CUDA integer casts.

This package is untracked/new and absent from the GitNexus index: impact queries
returned no target, and `detect_changes` returned zero indexed changes. Scope was
reviewed directly; that response is not evidence of no changes. No commit was made.

Run the same correctness tests on Spark before a campaign. Simulator validation
does not establish real compiler compatibility, hardware numerical parity or GPU
utilization. The full repository suite was not rerun; previously documented
baseline failures remain outside this change. X-stage enumeration differs from
the older CPU first-improvement RX, as documented in the main README.

Logs, current-source hashes and smoke artifacts:
root `output/vns-spark-20261117/`. Experiment results for the older hybrid Metal
runner remain in [`centralized_vns_r/validation`](../../centralized_vns_r/validation/README.md).
