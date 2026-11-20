# Resident Spark VNS Implementation Plan

> **For agentic workers:** use Superpowers TDD and independent-domain parallel implementation; review the complete integration before completion.

**Goal:** Execute all search arithmetic on CUDA with fewer host synchronizations,
including standalone guarded x refinement and hardware calibration.

**Architecture:** Separate package reuses existing data/checker/move functions.
Fixed search and x-stage are independent implementations; calibration uses the
fixed search public API. Existing sources stay unchanged during current campaigns.

**Tech Stack:** Python, NumPy, existing Numba CUDA; CUDASIM correctness locally.

**Spec:** ../specs/2026-11-17-vns-spark-resident.md

## Constraints and review focus

No commits, remote jobs, changes to existing solver sources, or GPU phase profiling.
Do not run compute-heavy tests concurrently with timed Metal experiments.
Float64; deterministic ties; memory guards before allocation; final CPU verification.
Review empty graphs, replica boundaries, plateau during a queued batch, floor
preservation under flow trimming, memory exhaustion and calibration timing fairness.

### Task 1: fused fixed-x CUDA engine (root)
- [x] Add failing tests: CPU prefix parity, warm starts, all/global plateau,
  batch-boundary masking, memory/option guards, 64/128/256 block widths.
- [x] Implement `gpu.py`: `solve(p, options=None, initial_y=None, *, patience=200,
  plateau_tolerance=1e-6, plateau_scope='all', batch_size=16, threads=128)` returning
  existing result fields plus `execution` metadata; `estimate_device_bytes(p,o,patience,batch_size)`.
- [x] Verify on simulator after benchmarks finish; preserve reference sources.

### Task 2: standalone CUDA x stage (independent agent)
- [x] Add failing tests and scalar oracle comparisons, including interacting flows,
  floors, near ties, zero proposals, malformed inputs and memory guards.
- [x] Implement `xstage.py`: `refine_x(p,y,*,iterations=1000,time_limit=15.,seed=4850,
  max_device_bytes=512*1024**2)` returning compatible RX result fields.
- [x] All proposal evaluation and acceptance on device; independent final CPU check.

### Task 3: calibration and CLI (root after engine interface)
- [x] Test invalid settings before output creation, calibration choice/memory filtering.
- [x] Add `__main__.py` and `calibrate.py`, standalone `refine.py`; separate saved stages.
- [x] Document Spark installation, real-hardware validation and timing limitations.

### Task 4: verification and experimental report
- [x] Complete existing 300s campaign, comprehensive validation and top10% replay.
- [x] Run focused simulator/new CPU tests; review new package independently.
- [x] Fix important findings, report measured evidence and untested hardware limits.

## Completion record

All four tasks completed locally. Full new-package CPU environment: 88 passed,
26 optional skips; CUDA simulator fixed search:23 passed; x stage:60 passed;
calibration:23 passed. Integrated and standalone CUDA simulator CLI outputs agree,
fixed artifacts unchanged. Ruff and whitespace checks pass. No real CUDA hardware
claim. Evidence/source hashes: `centralized_vns_spark/validation/local.json`.

Independent review corrections: consistent NumPy pairwise floor/objective sums,
one-iteration initial batch and overrun exclusion, nonfinite welfare guards, all
counted shaking candidates evaluated. Additional replica conversion range guard
has a RED/GREEN regression. Dynamic x scratch targets eight blocks per actual SM,
with configured-memory/work caps and coalesced candidate axis.

Previous campaigns: both300 reports24 cases,65 unique measured runs across120/300,
16768 archived solutions independently verified,31 explicit reused records.
Top10 replay and six-panel figure completed; validation README updated. Original
VNS and all prior campaign solver source hashes unchanged. GitNexus does not index
the new package; direct scope/caller review substituted for missing graph entries.
No commits, remote jobs or separate phase profiling performed.
