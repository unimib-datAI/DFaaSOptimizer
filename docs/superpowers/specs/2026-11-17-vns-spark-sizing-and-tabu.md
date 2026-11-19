# VNS sizing on Spark and bounded visited-state memory

## Status and objective

The CPU/Metal candidate selector is now vectorized in float64. CUDA uses one
64-thread block per trajectory, with deterministic candidate reduction and a
cooperative archive copy. CUDA hardware compilation and timing are not validated
on the Mac. The calibration and tabu experiments below are proposed next steps,
not implemented production features. No remote experiments are launched here.

Optimize feasible welfare within the campaign's wall-time budget. Candidates per
second is a diagnostic, not the final objective: increasing candidates can reduce
the number of sequential improvements enough to worsen the final solution.
Any previous throughput knee must be remeasured after changing the selector.

## Spark calibration

DGX Spark has GB10, 128 GB unified memory, and 273 GB/s memory bandwidth, per
[NVIDIA hardware documentation](https://docs.nvidia.com/dgx/dgx-spark/hardware.html).
The advertised FP4 AI throughput does not size this integer/float64 algorithm.
Read SM count, compute capability, available memory and driver/runtime versions
from the actual device. Unified memory is shared with host allocations and OS;
the entire advertised capacity is not an allocation budget.

1. Run exact CPU/CUDA correctness cases on real hardware: compound moves, near
   ties, replica boundaries and full trajectory traces. CUDASIM is insufficient.
2. Profile candidate generation, selection/archive and host synchronization
   separately with CUDA events and total wall time. Warm compilation separately;
   report initialization/setup, calibration cost and cold total time as well.
   Use representative search states, including exchanges and shaking.
3. Explore pool sizes 32, 64, 128, 256, 512, 1024 and candidates 128, 256, 512,
   1024, 2048, 4096 by coarse doubling then refinement near the useful range.
   Prune allocations above an explicit cap with reserve for runtime and host data.
   Test multiple pool/candidate ratios at similar total candidate counts.
4. Current block sizes are fixed at 64. A later calibration change can compare
   64/128/256 for generation and reduction separately, changing shared-array sizes
   consistently and retaining power-of-two reduction widths. Do not change
   blockDim alone. Confirm each variant's exact decisions before timing it.
5. Keep a small shortlist on the measured throughput/memory/iteration-latency
   frontier. A smallest configuration within 95% of maximum throughput is an
   initial candidate, not automatically the final winner. More occupancy is not
   necessarily faster; inspect stalls/register pressure if scaling plateaus
   ([CUDA best practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)).
6. Compare the shortlist at equal wall time and several seeds on representative
   training instances, followed by held-out instances. Select by welfare and
   anytime progress; use memory/cost as tie breakers. Use the actual campaign
   budgets rather than extrapolating short-run quality.
7. Freeze the chosen profile before the campaign. Persist hardware/software/code
   hashes, numerical policy, instance class, budget, raw repetitions and decision.
   A hardware/code change invalidates the profile. Keep pool/candidates fixed
   within each run for reproducibility. Start with one GPU job at a time and
   measure any concurrent-job configuration explicitly.

The minimum implementation is a calibration script producing JSON and existing
CLI flags. Dynamic resizing within search, automatic remote orchestration and a
general tuning framework are unnecessary for the first experiment.

## Visited-state memory

For fixed x, an allocation y determines incoming and rejected requests. Remember
integer y, scoped to the same input and x, rather than floating-point welfare.
However, identical y does not imply identical future search: neighborhood level,
stale counter and counter-based random candidates can differ. A global permanent
ban therefore prunes useful exploration as well as duplicated work. This is tabu
search, not transparent memoization of the existing algorithm.

At 500 nodes the measured cases have 7,500, 14,680 and 21,460 int64 flow entries.
100,000 complete allocations need 6.00, 11.744 and 17.168 GB respectively before
indices/metadata. A 128-bit fingerprint takes 1.6 MB for 100,000 entries, but is
probabilistic unless a matching full state is verified.

Recommended first experiment: per-trajectory FIFO tabu memory, tenure 16/64/128
accepted state changes, with aspiration for a move that improves the trajectory's
best archive and a defined deterministic fallback when all candidates are tabu.
Test it against cache disabled at equal total wall time, including lookup cost.
Revisits during shaking are especially relevant; remaining in the same state
after rejecting every candidate must not be counted as a new revisit.

Use an incremental 128-bit integer fingerprint of (edge, flow). A candidate
changes at most four entries; combine repeated edges before computing the final
state fingerprint. Keep device lookups bounded and per-trajectory initially to
avoid a globally contended hash table. Hash matches should be verified against
stored integer allocations if exact duplicate detection is required. Never
silently treat a fingerprint as collision-free. The hash is a filter; on a
collision with unequal arrays, the candidate remains admissible.

For an exact ring with pool 256 and tenure 64, full y payload is approximately
0.98–2.81 GB on these instances (0.92–2.62 GiB), plus small fingerprints and
metadata. For pool 1024 the largest case is 11.25 GB. This is plausible on Spark
but must count against the shared-memory budget with all solver arrays. A
fingerprint-only variant is much smaller, with explicitly approximate semantics.

Before implementation, observe revisits without filtering them, recording:

- Changed-state revisits per trajectory, globally, and within recent tenures;
  distinguish rejected iterations and zero-net moves from returns to an old state.
- Welfare progress and the phases in which repeats occur.
- Instrumentation overhead separately from uninstrumented search timing.

Then test cache disabled versus bounded tenures on paired seeds and equal budgets.
Report welfare, accepted moves, tabu hits, hash collisions, lookup time and memory.
Retain a tenure only if its quality/diversity benefit repays its cost. Full-state
checks on cache hits can be bandwidth-heavy even when storage fits in RAM.

An objective-only cache is unlikely to pay: current scoring updates at most four
terms. Preventing costly cycles is the motivation, not avoiding that small sum.

## First observation, 2026-11-17

On the exact historical cubic n500 input, seed 4850, pool 64, candidates 512 and
1,000 iterations, 23,125 allocation changes included 336 per-trajectory revisits
(1.453%). All 336 were within the last 64 changed states. A global set found 352
revisits, including those same 336. The remaining 40,875 trajectory-iterations
kept the immediately previous allocation and are not counted as revisits.
These are BLAKE2b-128 observations, not collision-verified exact counts.

The separate full-array hashing/Python lookup observer consumed 2.924 s in addition
to 4.805 s in engine steps; it did not filter any candidate. This does not estimate
the cost of a future incremental GPU implementation. It does show that a naive
host-side cache is too expensive for the observed frequency on this one case.
Memory is affordable on Spark, but a cache should remain an optional measured
experiment, not an assumed improvement or default. Late-search behavior and other
instances may differ. See `centralized_vns/validation/macbook-parallel-selection.json`.
