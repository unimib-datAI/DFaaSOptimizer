# Task 6 Report: End-to-end Gurobi-gated smoke + reproducibility tests for FaaS-MABR

## Config Used

Identical to the brief's config with no adaptations needed:
- `neighborhood: {"type": "planar", "degree": 3}` with `Nn=10` (even, >= 6) — instance built successfully
- `solver_options` keys: `br_s`, `br_r`, `br_o` with `latency_weight: 0.0`, `fairness_weight: 0.0`
- `seed: 21`, `max_iterations: 2`, `patience: 1`, `max_steps: 8`, single time step `min_run_time=max_run_time=1`

No adaptations from the brief were required for the config.

## Artifact Filename Confirmation

Confirmed by reading `decentralized_bestresponse._run()` (lines 427-443):
- `obj.csv` — written via `pd.DataFrame(obj_dict["LSPr_final"], columns=[method_name]).to_csv(...)` at line 435
- `runtime.csv` — written via `pd.DataFrame({"tot": runtime_list}).to_csv(...)` at line 441
- `termination_condition.csv` — written at line 438
- `LSPc_solution.csv` — written via `save_solution(..., "LSPc", solution_folder)` at line 432, which produces `{model_name}_solution.csv`

These match `test_powerd_e2e.py`'s assertions exactly.

## Evidence: Tests RAN Under Gurobi and PASSED

```
$ uv run pytest tests/test_bestresponse_e2e.py -v
tests/test_bestresponse_e2e.py::test_br_s_artifacts PASSED               [ 25%]
tests/test_bestresponse_e2e.py::test_br_r_artifacts PASSED               [ 50%]
tests/test_bestresponse_e2e.py::test_br_o_artifacts PASSED               [ 75%]
tests/test_bestresponse_e2e.py::test_br_s_reproducible PASSED            [100%]
4 passed in 3.49s
```

All 4 tests PASSED (none SKIPPED — Gurobi was available and used).

### obj.csv Verification

Each method produced `obj.csv` with >= 1 row containing finite numeric values in the method-named column (`FaaS-MABR-S`, `FaaS-MABR-R`, `FaaS-MABR-O`). The `_assert_artifacts` helper explicitly checks `len(obj) >= 1` and `np.isfinite(pd.to_numeric(...)).all()`.

### Reproducibility Verification

`test_br_s_reproducible` runs `run_br_s` twice with the same config (same seed=21) but different output folders (`tmp_path / "a"` and `tmp_path / "b"`), reads both `obj.csv` files, asserts identical shapes with >= 1 row, and verifies `np.allclose(oa, ob)`.

## Full Suite Count

```
$ uv run pytest -q
257 passed, 1 warning in 14.55s
```

## Files Changed

- `tests/test_bestresponse_e2e.py` (NEW) — 4 test functions

## Self-Review Findings

1. All 4 e2e tests RAN under Gurobi and PASSED (not skipped) — confirmed
2. Each test asserts the method-name obj column (`FaaS-MABR-S`/`-R`/`-O`) with finite values — confirmed
3. obj.csv has >= 1 row (not vacuous on empty DataFrame) — confirmed via `assert len(obj) >= 1`
4. Reproducibility test genuinely compares two independent runs' obj values via `np.allclose` — confirmed
5. Artifact filename assertions (`obj.csv`, `runtime.csv`, `termination_condition.csv`, `LSPc_solution.csv`) match what the runner actually writes — confirmed by source inspection
6. Uses 2-space indentation throughout — confirmed
7. No production code changes — confirmed (test-only file)

## Concerns

None. The brief's config and assertions were accurate. No adaptations were needed.
