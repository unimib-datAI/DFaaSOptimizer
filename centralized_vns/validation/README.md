# Phase 1 evidence

2026-11-17, Titan, CPU and CUDA simulator only. No .35/.175 access.

- Fresh minimal CPU environment: 24 tests passed, 3 GPU tests skipped.
- Dedicated Linux CUDA simulator: 3 GPU tests passed, including complete
  trajectory agreement and 81 compound-move scores against the CPU reference.
- CPU CLI tiny fixture: exact welfare 8/3, fixed x=[1,0], independently feasible.
- CPU CLI real Cilic Europe instance: n30/f5, pool4, candidates32, 100 iterations;
  initial all-residual-rejected welfare 549.3880423153472, final 594.1498226728329.
  This is a functionality smoke, not a comparative performance benchmark.
- Final fresh review: no remaining blocking issue after numerical/shaking fixes.

The JSONs record configuration, hashes and raw timing. They were captured before
committing (base commit, dirty=true); the Git commit containing these artifacts
identifies the source under test. CPU transitive packages are recorded separately;
platform-specific wheels mean this is evidence, not a cross-platform lockfile.

Real CUDA compilation, hardware numerical agreement and performance are not yet
validated. The fresh Linux CPU environment verifies that the documented minimal
dependency set is sufficient.

## MacBook CPU / Metal validation, 2026-11-17

Apple M3 Pro (11 CPU cores, 14 GPU cores, 18 GB RAM), Python 3.10.19,
NumPy 2.2.6, pytest 9.2.0, MLX 0.32.4. Real GPU access, no simulator.

- VNS tests: 34 passed, 3 CUDA skipped because Numba is absent. Includes exact
  per-iteration candidate, cache and archive equality, near ties, replica
  boundaries, exchanges, no edges, memory guard, CLI and benchmark artifacts.
- Independent read-only review found no blocking issue; additional probes checked
  59,880 insertion spaces and rejection of an oversized table before allocation.
- Local Cilic Huawei n30/f5 fixture, step 0, 64 candidates, 100 iterations,
  seeds 4850/4851/4852, three repeats each. 18 CPU/Metal pairs have exactly equal
  final y/z/replicas and every per-trajectory archive value.
- Tiny fixture: nine further pairs (pool4, candidates32, 50 iterations) matched
  exactly; median search CPU 0.077719 s, Metal 0.027869 s. All 27 saved pairs and
  their source/input hashes were rechecked after benchmarking.

| Pool | CPU median search | Hybrid Metal median search | Ratio |
| --- | ---: | ---: | ---: |
| 4 | 0.838792 s | 0.080025 s | 10.48x |
| 32 | 6.713320 s | 0.461237 s | 14.56x |

Search includes GPU synchronization, transfers and CPU acceptance. Tables/setup
and compilation warmup are recorded separately in `macbook-metal.json`; adding
median table/setup time gives 0.838810/0.081543 s (pool4) and
6.713344/0.462783 s (pool32). Shared CPU DP initialization took 0.801096 s.
These ratios compare the existing Python CPU reference with a hybrid backend,
not native optimized CPU code. They do not imply equal speedups on other inputs.
This is a different instance from the earlier Titan Azure smoke; no cross-host
speed ratio is claimed. JSON includes full input hashes and source hashes.

The broader repository suite was interrupted during unrelated long algorithm
regressions after 642 passed, 284 skipped and four failures. All four failures
reproduced in a clean archive of pre-change HEAD 2373e3f:

- `test_alpha_gt_beta_correction_preserves_physical_inputs`: delta/beta assertion.
- `test_local_numeric_unsupported_precision_never_initializes_cuda`, for demand,
  alpha and incoming_load with longdouble: reaches the forbidden CUDA hook.

The complete repository suite is therefore not certified green. No code outside
the standalone VNS module was changed to mask these failures.
Ruff passed on the changed Python files at this stage. The two then-existing F821
warnings in `gpu.py` were subsequently resolved by an explicit global CUDA binding
during the parallel-selection change below, preserving lazy imports and CUDASIM.

## 500-node parallelism sweep, 2026-11-17

Same MacBook and implementation as above. Three original local catalog instances
(instance seed 1, step 0): synthetic planar cubic / Huawei f5, EUA Melbourne /
Huawei f5, and Ether IIoT / Huawei f10. Their full paths and checksums are in
`macbook-metal-500.json`. Inputs and algorithm code remained unchanged.

Swept pools 64/128/256/512 and candidates 64/256, at 30 fixed iterations and
algorithm seeds 4850/4851/4852: 72 GPU runs across 24 configurations. Each
configuration had two discarded warmup steps and three fresh measured states.
Six direct CPU pairs (64×64 and 256×256, seed4850 on each instance) matched
exactly in y, z, replicas and all per-trajectory archive traces.

Search timings for the direct pairs at pool256/candidates256 (one CPU pair each,
not repeated timing medians; initialization/setup excluded):

| Instance | CPU | Hybrid Metal | Ratio |
| --- | ---: | ---: | ---: |
| Planar cubic, f5 | 63.940 s | 4.006 s | 15.96x |
| EUA Melbourne, f5 | 64.153 s | 4.050 s | 15.84x |
| Ether IIoT, f10 | 110.722 s | 4.315 s | 25.66x |

At 256 candidates, increasing the pool from 64 to 512 only improves median
candidate throughput by about 0–2%; the time grows roughly with the pool. Increasing
candidates from 64 to 256 at pool64 improves throughput by 37–61%. The largest
configuration generates 131,072 candidate slots per iteration. Counts include
empty/infeasible sampled moves, consistently with the reference counter.

Separate cProfile runs (10 steps at 512×256, excluded from timing samples) attribute
94.9%, 94.8% and 92.3% of the profiled step time to CPU.accept, respectively.
This identifies CPU selection/acceptance as the current scaling bottleneck;
it is not a hardware GPU utilization measurement, and profiling adds overhead.
The next targeted optimization is vectorized float64 selection with the same
tie/order policy, followed by rerunning exact-parity and scaling checks.

The 2 GiB array cap was used: estimated largest live GPU arrays are 389, 418
and 1241 MiB, respectively. These are estimates, not measured peak process memory;
the dense tables, runtime caches and host copies remain a limitation.

Raw timings, allocations, profiles, runnable experiment script, CSV and plot:
`output/vns-macbook-500-20261117/` in the parent experiment repository. Source/input
hashes and all six saved paired allocations/traces were verified after completion.

## Larger candidate batches and historical quality, 2026-11-17

Extended the same three 500-node inputs to 256/512/1024/2048 candidates and pools
64/256, with 30 iterations and seeds 4850/4851: 48 GPU runs. The largest grid has
524,288 candidate slots. The 2 GiB array cap still suffices. A direct CPU/Metal
pair at pool64/candidates2048 on the cubic input matches exactly, including the
full per-trajectory archive trace: 123.533 s CPU versus 6.963 s hybrid Metal.

Median search seconds across two seeds at pool256:

| Candidates | Cubic f5 | Melbourne f5 | IIoT f10 |
| --- | ---: | ---: | ---: |
| 256 | 3.816 | 3.909 | 4.216 |
| 512 | 7.040 | 7.197 | 7.677 |
| 1024 | 13.710 | 13.997 | 14.515 |
| 2048 | 27.508 | 27.821 | 28.170 |

At 2048 candidates throughput is about 0.56–0.57 million slots/s: only 11–20%
higher than 256 candidates despite eight times the work. Cubic welfare is unchanged
at 9740.111081 after 30 iterations. Melbourne improves from roughly 9687 to 9734;
IIoT improves from roughly 16585 to 16590 and plateaus by 1024 candidates.
Raw measurements are in the parent repo's `output/vns-macbook-candidates-20261117/`;
compact provenance and summaries are in `macbook-metal-candidates.json`.

For historical quality, rebuilt the cubic `dmean-v1` input using the existing
`recalculate_delta.rebuild`. Its metadata SHA256 exactly matches the frozen paper
snapshot: `4d84f531ea388679cdd955bc721118ce38c95cc9cc0cf347619a5db616e5634a`.
The revision changes delta and self-edge beta; VNS uses neighbor beta and excludes
self edges. The longer runs also verify identical x and permitted edges against
the original-input experiment. Source catalog files were never overwritten.

Ran 1000 iterations on the verified revised input, pool64/candidates256, seeds
4850/4851. Both solutions pass final feasibility. Welfare is 10187.025823 and
10184.045945 (mean 10185.535884). Search wall times including engine setup are
29.677/29.776 s; total wall times including input loading and DP are 38.473/38.206 s.

| Historical method/reference | Recorded welfare | VNS mean difference |
| --- | ---: | ---: |
| PG-R (10-seed mean) | 10124.772641 | +0.6001% |
| PG-R fixed-x (10-seed mean) | 10141.636795 | +0.4329% |
| Greedy-RR | 10134.279287 | +0.5058% |
| MADEA detailed | 10136.164076 | +0.4871% |
| MADEA selective | 10142.062647 | +0.4286% |
| One-shot detailed | 10182.281685 | +0.0320% |
| One-shot selective | 10180.885531 | +0.0457% |
| Centralized recorded incumbent | 10195.953905 | -0.1022% |

This is descriptive evidence on one verified input and two VNS seeds. The
centralized incumbent is not a proven optimum. Different machines, time clocks
and budgets prohibit a cross-campaign speedup claim. The frozen paper campaign
has no matching Melbourne/IIoT inputs, so no historical ranking is claimed for
those families. Full historical source checksum, all ten method/reference rows
and VNS run records are in `macbook-history-comparison.json`.

## Parallel selection and revised candidate sweep, 2026-11-17

CPU/Metal selection now uses NumPy float64 masking and first-index argmax.
CUDA uses one 64-thread block per trajectory, a deterministic reduction and a
parallel archive copy after ordered four-entry state updates. No floating-point
sums were reordered. Exact ties and first-valid shaking semantics are preserved.

On the same three 500-node inputs: pool 256, 30 iterations, seed 4850, two
repetitions in alternating original/updated order, warm JIT and setup separately.
The original CPU.accept function is loaded from the pre-change source snapshot;
all other Metal code and inputs are shared. All 28 measured pairs have exactly
equal flows, rejections, replicas and full per-trajectory archive traces. Four
initial cubic C=256/512 measurements overlapped test execution; clean isolated
repeats replace them in the 24 selected pairs below. Input hashes were rechecked.

Median search seconds at 2,048 candidates:

| Instance | Previous Metal | Updated Metal | Ratio |
| --- | ---: | ---: | ---: |
| Planar cubic | 27.914 | 1.110 | 25.14x |
| Melbourne | 28.340 | 1.145 | 24.74x |
| IIoT | 29.347 | 1.510 | 19.44x |

These compare two hybrid Metal implementations, not CPU versus GPU. The gain is
from removing the Python acceptance scan; this does not measure the CUDA change.
Post-change throughput continues increasing through C=2,048 on all three cases;
the earlier throughput knee at C=512/1,024 no longer applies.

The equal-wall-time sweep was repeated on the exact historical cubic input,
pool 64, seeds 4850/4851, 15 s including engine construction after JIT warmup.
Input loading/DP initialization is shared and excluded. Candidate order reverses
for the second seed. Mean final welfare (larger is better):

| Candidates | Previous selector | Updated selector | Updated iterations, two seeds |
| --- | ---: | ---: | ---: |
| 256 | 10184.976478 | 10188.007575 | 3928 / 3880 |
| 512 | 10185.905768 | 10189.998456 | 3391 / 3394 |
| 1,024 | 10135.372031 | 10189.522761 | 2706 / 2716 |
| 2,048 | 9947.626212 | 10189.529849 | 1958 / 1961 |

C=512 is the best observed quality/time compromise on this one input and budget;
the differences from 1,024/2,048 are small and two seeds do not establish a general
optimum. Its mean is 0.0584% below the historical centralized incumbent and
0.6442% above historical PG-R. This is exploratory tuning on the same input,
with different historical hardware/budgets; no causal method superiority or
cross-campaign speedup is claimed. The incumbent is not a certified optimum.

A separate 1,000-iteration observation at pool 64, C=512, seed 4850 found 336
per-trajectory revisits among 23,125 changed allocations (1.453%), all within
the last 64 changed states. The other 40,875 trajectory-iterations kept their
previous state. A global set counted 352 revisits including the same 336.
Full-array BLAKE2b-128/Python set/deque instrumentation added 2.924 s to 4.805 s
of engine steps; it was excluded from quality timing. Hash collisions were not
independently verified. No tabu filter was enabled. Memory may fit on Spark, but
this small revisit frequency does not yet justify a default cache.

Verification: real-Metal module suite 42 passed / 7 CUDA skipped; isolated Mac
Numba 0.63.1 CUDASIM suite 7 passed (including nonmultiple candidate counts, ties
across strides, negative shaking, duplicate edges and archive tails). Independent
review found no substantive bug; explicit all-invalid ordinary CUDA descent
starting at level 1/2 is a minor deferred test case. Real CUDA compilation and
performance remain unverified. The broader-suite failures documented above remain.
GitNexus does not index these VNS symbols; impact/diff analysis used direct source
and caller inspection. No remote job, commit or push was performed.

Compact evidence: `macbook-parallel-selection.json`. Scripts, original source,
raw timings/allocations/traces and observation: root
`output/vns-parallel-selection-20261117/`. The proposed calibration and tabu
protocol is in `docs/superpowers/specs/2026-11-17-vns-spark-sizing-and-tabu.md`.
