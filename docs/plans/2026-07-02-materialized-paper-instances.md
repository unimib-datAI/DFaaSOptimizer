# Materialized Paper Instances Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Materialize complete paper instances once, verify them with SHA-256 checksums, and make every remote runner consume the exact stored inputs.

**Architecture:** Add one small instance module that derives a canonical generation specification, writes/loads the five-file instance format, and materializes a batch with deduplication. Extend the existing CLI and job mapping, then add an additive materialized branch to the shared `init_problem` function so all algorithms benefit without runner-specific changes.

**Tech Stack:** Python 3.10 standard library (`hashlib`, `json`, `pathlib`), NumPy, NetworkX, existing generators, pytest, ray-dispatcher input specifications.

---

### Task 1: Instance format, identity, and checksums

**Files:**
- Create: `remote_experiments/instances.py`
- Create: `tests/test_remote_experiments_instances.py`

1. Write failing tests for canonical identity shared across algorithms, different identity when generation inputs differ, complete materialization, graph/load round-trip, and checksum failure.
2. Run `uv run pytest -q tests/test_remote_experiments_instances.py` and confirm failure because the module is absent.
3. Implement the minimal generation-spec, SHA-256, materialize, validate, and load functions using existing generators and JSON helpers.
4. Run the focused tests and confirm they pass.

### Task 2: Batch materialization CLI

**Files:**
- Modify: `remote_experiments/cli.py`
- Modify: `tests/test_remote_experiments_cli.py`

1. Write failing tests for `materialize <batch> [-o root]` parsing and suite output creation.
2. Run the focused tests and confirm the missing-command failure.
3. Add `cmd_materialize` and the parser entry; default output is `remote_experiments/instances`.
4. Run the CLI and instance tests.

### Task 3: Transfer materialized inputs to remote jobs

**Files:**
- Modify: `remote_experiments/jobs.py`
- Modify: `remote_experiments/cli.py`
- Modify: `tests/test_remote_experiments_jobs.py`
- Modify: `tests/test_remote_experiments_cli.py`

1. Write failing tests requiring the job configuration to select materialized mode and requiring all five files as job inputs.
2. Run the focused tests and confirm the old config-only behavior fails.
3. Extend `experiment_to_job` with an instance root, validate the instance, copy the configuration before changing `limits`, and pass the root from `cmd_run` through a new `--instances` option.
4. Run the focused tests.

### Task 4: Shared runner loading path

**Files:**
- Modify: `run_centralized_model.py`
- Modify: `tests/test_run_helpers.py`

1. Write a failing test that materializes a small instance, calls `init_problem` in materialized mode, and compares base data, temporal traces, agents, and graph edge attributes.
2. Run the test and confirm failure because materialized mode is unsupported.
3. Add the materialized branch at the start of `init_problem`; validate/load and copy provenance files to the solution directory. Leave the existing generation branch unchanged.
4. Run the focused test plus existing runner-helper tests.

### Task 5: Documentation and data hygiene

**Files:**
- Modify: `remote_experiments/README.md`
- Modify: `docs/hierarchical_model_experiment_plan.md`
- Modify: `.gitignore`

1. Document define → materialize → run, the distinction between generation and algorithm seeds, temporal loads as instance data, checksums, layout, and ZIP archival.
2. Ignore `remote_experiments/instances/*/data/` while retaining the generated suite README and manifest as optional reviewable artifacts.
3. Do not generate or commit the full paper datasets during implementation.

### Task 6: Verification

1. Run `uv run pytest -q tests/test_remote_experiments_instances.py tests/test_remote_experiments_jobs.py tests/test_remote_experiments_cli.py tests/test_run_helpers.py`.
2. Run the complete test suite with `MPLCONFIGDIR=/tmp/matplotlib uv run pytest -q`.
3. Run Ruff on all changed Python files.
4. Run `git diff --check`.
5. Run GitNexus change detection (or the CLI graph diff equivalent available in this environment) and verify only the expected generation, job dispatch, CLI, and shared initialization flows are affected.
6. Leave all changes uncommitted.
