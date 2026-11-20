# Centralized fixed-x population VNS implementation plan

Goal: a self-contained centralized VNS with fixed LSP_detailed x, CPU reference,
CUDA population and candidate parallelism, portable CPU testing on macOS.
Approved design: fixed x; feasible sparse flows; derived replicas; one accepted
move per trajectory per iteration; global best preserved; seeded perturbations.
All new code, tests and docs live here. Existing LSP/DP modules are imported.

- [x] Model: sparse permitted edges, integer load/x, welfare, independent feasibility.
  Test conservation, memory thresholds, directed adjacency, zero load and roles.
- [x] Moves: insert/remove/reroute/two-flow exchange, evaluate compound deltas
  atomically. Test deltas against full recomputation and exact tiny enumeration.
- [x] CPU population: deterministic candidate generation, VND, increasing shaking,
  best archive, budgets, trace; test reproducibility and nondecreasing incumbent.
- [x] CUDA: resident population; parallel candidate generation/evaluation, one
  accepted move per trajectory. Lazy optional import; test CPU/CUDASIM agreement.
- [x] CLI: materialized instance loader, LSP CPU/CUDA initialization, portable output,
  input hashes, config and commit provenance; smoke on a real local instance.
- [x] Independent review, final tests, documentation and Mac handoff, commit/push.

Use executing-plans inline and a fresh final reviewer. No .35/.175 use in phase 1.
Real CUDA driver execution, speedup, pool sizing remain phase 2 validation.
Test using the existing Python 3.10 solver environment, a separate small CUDA
simulator environment if necessary; never sync optional RL dependencies.
GitNexus unavailable: direct scope/diff inspection replaces impact tooling.

Ruling: use a dedicated branch in the current submodule checkout, no duplicate
worktree/environment. No existing code changes; running remote campaigns retain
already provisioned source. Local campaign launches should await handoff.

Validation ledger (2026-11-17): CPU clean environment 24 passed, 3 optional GPU
tests skipped; dedicated Linux CUDASIM environment 3 passed. Tiny exact oracle
and final full feasibility checks passed. Real Cilic n30/f5, 4 trajectories,
32 candidates, 100 iterations: welfare 549.3880423153472 -> 594.1498226728329.
Independent review caught and verified fixes for int64 memory overflow, LSP
replica boundary tolerance, insufficient shaking, extreme capacity conversion
and corrupted flow accumulation. No remaining blocker in final review.

Ruling: phase 1 uses independent trajectories with best archives and increasing
whole-flow destruction; elite migration is deferred until quality benchmarking.
Persistent CUDA buffers are bounded; JIT/driver overhead is not included in the
buffer estimate. Actual device compilation/speed/memory remain phase 2 tasks.

Validation JSONs were captured before committing, so their commit points to the
pre-feature base and dirty=true. The containing Git commit identifies the tested
source; subsequent runs record their actual current revision automatically.
