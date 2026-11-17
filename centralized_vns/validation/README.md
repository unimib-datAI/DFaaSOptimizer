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
validated. Tests on macOS have not been run; the fresh Linux CPU environment
verifies that the documented minimal dependency set is sufficient.
