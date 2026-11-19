# Resident VNS-R execution on DGX Spark

User requests all parallelizable search work on the GPU, without separate phase
profiling. Preserve existing algorithms/results and float64 decisions. No remote
execution is authorized or possible in this session.

Implement a separate `centralized_vns_spark` package so running Metal experiments
and their source hashes remain unchanged. Reuse Problem, Options, VNS-R moves,
independent checker and result formats. CPU performs input/initialization, launches,
deadline polling, final independent validation and I/O. All candidate work,
selection, state/archive updates and rolling plateau computation execute on CUDA.

Fuse a trajectory's candidate generation/scoring/reduction/acceptance in one
block, allowing 64/128/256 threads. Keep deterministic candidate-index ties and
float64 arithmetic without fastmath/FMA contraction. Batch launches with GPU stop
masking, so plateau stops at its first qualifying iteration even within a batch.
Poll wall deadlines between bounded batches; report this granularity honestly.
Global and all-record plateau supported; default remains all-record. Trace records
are checkpoints with real host timestamps, not invented per-iteration timings.

Standalone CUDA x refinement evaluates independent local transactions in parallel,
commits only a selected feasible improvement, retains initial per-node local-gain
floors, and checks the final solution independently. It may use a deterministic
batch best-improvement order instead of the existing CPU randomized first-improvement
order; identify that variant explicitly and test against a scalar oracle. Both
fixed and variable solutions must be saved separately. No silent CPU fallback.

Calibration tests several pool/candidate/block configurations, prunes memory before
allocation, separates warm-up, ranks welfare at equal search time (throughput tie
break), and saves hardware/configuration/validation evidence. Hardware performance,
occupancy, compiler compatibility and bitwise parity must be verified on Spark;
CUDASIM proves neither real compilation nor speed.

Alternatives: only increase candidates (does not remove synchronization); port to
another framework (unnecessary dependency and precision changes). Chosen approach:
reuse scalar CUDA primitives with a separate fused runner and device monitor.
