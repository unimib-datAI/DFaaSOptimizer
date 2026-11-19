# VNS-R plateau implementation plan

**Goal:** Stop stagnant VNS-R populations without discarding improving trajectories.
**Design:** Use the per-trajectory best archives already returned by every backend.
At each iteration compare against the archive `patience` iterations earlier:
`(best_now - best_then) / max(1, abs(best_then)) <= plateau_tolerance` for every
trajectory. Defaults: patience=200, tolerance=1e-6. Zero patience disables it;
zero tolerance requires unchanged archives. Positive patience must be at least
`3 * shake_after + 1` steps, the length of a fully stalled search/shaking cycle.
This gives shaking opportunities, not a guarantee that every trajectory shook.
The rolling window includes the initial archive. Time and iteration caps take
precedence at a simultaneous stop. Report reason and completed iterations.
These defaults are an implementation choice, not calibrated experimental results.

**Scope:** `centralized_vns_r/core.py`, `__main__.py`, README and new stop tests.
Preserve original VNS and independent x refinement. No dependencies or GPU kernels.
Existing callers retain their signature; add keyword-only patience/tolerance.
CLI saves effective settings through the existing result writer.

- [x] Add failing tests: plateau on real saturated search, disabled/default stop,
  limits, invalid options, cumulative progress, subordinate trajectories, zero and
  negative objectives, and CLI persistence with separate x-stage artifacts.
- [x] Implement the rolling deque and CLI flags; document stop semantics/cost.
- [x] Run focused tests, VNS suites on Metal, CUDA simulator stop parity, lint.
- [x] Smoke-test a 500-node instance with plateau enabled/disabled at equal
  iteration caps. Independently check feasibility and archive-prefix parity.
- [x] Request fresh review, address findings, record verification. Do not commit.

Review risks: premature stop before shaking; best-only population reduction;
window off-by-one; scale near zero/negative welfare; hard caps misreported.

Verification: 114 passed / 18 skipped combined module tests on real Metal;
focused CUDASIM parity 1 passed; ruff clean. Six 500-node solutions independently
checked. Cold-start pairs run to 3000 iterations unchanged; warm start stops at
861 instead of 1000 with the same final welfare. See validation/macbook-plateau.json
under centralized_vns_r. Review fix: share plateau validation between solve/CLI
and reject invalid settings before initialization/output creation (2 RED then
GREEN regressions). Fresh review confirmed the fix. No commits or remote jobs.
