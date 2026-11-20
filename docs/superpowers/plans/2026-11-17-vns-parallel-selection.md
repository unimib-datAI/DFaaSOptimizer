# VNS parallel selection

Goal: remove the measured Python candidate scan and the CUDA serial per-trajectory
scan while preserving every decision at a fixed seed and iteration budget.

Scope: `core.py` selection and shared CPU/Metal acceptance; `gpu.py` block reduction
and archive copy; focused tests; local paired benchmarks and validation notes.
No changes to candidate generation, precision, initialization or input data.

- [x] Add independent selection fixtures for ties, near ties, nonfinite gains,
  empty moves, shaking, and no valid candidate; run before implementing helper.
- [x] Replace Python scan with NumPy float64 filtering and first-index argmax.
- [x] Characterize CUDA acceptance against the original implementation on
  candidates spanning lanes/strides, compound moves, rejected moves and archives.
- [x] Use one 64-thread block per trajectory, deterministic reduction and parallel
  archive copy; keep four-entry updates in thread zero, with uniform barriers.
- [x] Run CPU/Metal tests on real Apple GPU and CUDA tests in temporary CUDASIM.
- [x] Compare original/updated Metal at fixed iterations on three 500-node inputs,
  require exact arrays and traces, and measure setup and search separately.
- [x] Record Spark calibration design and bounded tabu-cache experiment separately.

Review focus: lowest index on ties across strides; first valid shaking candidate
even if negative; archive copying after updates; duplicate move-edge entries;
candidate counts smaller than or not divisible by the thread block size.

The user authorized continuing implementation without further confirmation.
CUDA simulator evidence does not establish hardware compilation or speedup.

Execution record:

- Eight selector fixtures failed before the helper existed and passed afterward.
- Seven CUDA characterization cases passed before and after the reduction change.
  This is a behavior-preserving optimization; no hardware speed claim is tested
  in CUDASIM. Final CUDASIM run: 7 passed in 31.01 s (Numba 0.63.1, temporary Mac
  environment); real Metal module suite: 42 passed, 7 CUDA skipped in 6.94 s.
- GitNexus impact could not find the VNS symbols and detect_changes returned no
  changes despite the git diff. Direct caller/source analysis bounds the change
  to CPU/Metal acceptance and CUDA kernels/launches. No commits or remote jobs.
- Independent reviewer found no substantive issue. Minor deferred coverage:
  an explicit all-invalid CUDA ordinary-descent row starting at level 1 or 2;
  existing tests cover all-invalid shaking and ordinary trajectories.
- Ruling: characterize the CUDA decision contract rather than add a flaky runtime
  threshold test; real Spark timing remains required and can show no speedup.
- Ruling: initial cubic C=256/512 timings overlapped test execution; retain those
  raw files but replace them in the selected summary with clean isolated repeats.
- The broader repository's four pre-existing failures remain documented in
  validation/README.md; this change does not claim a green full repository suite.
- All 28 measured pairs (24 selected, four superseded) have exact flows,
  rejections, replicas and full per-trajectory archive traces. Raw artifacts and
  reproducible scripts: root `output/vns-parallel-selection-20261117/`.
- The repeated equal-time quality sweep has eight complete runs. A separate
  observational cache probe found 336 revisits / 23,125 state changes (1.453%);
  no candidate filter or tabu mechanism was enabled.
