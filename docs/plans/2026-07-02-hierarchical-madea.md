# Hierarchical MADeA Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a separately runnable hierarchical method that extends the production FaaS-MADeA level-one auction without modifying existing algorithm functions.

**Architecture:** A new runner reuses public MADeA helpers for level one and invokes the existing hierarchy engine for residual demand. Existing `hierarchical` remains unchanged; dispatch and paper suites use the explicit `hierarchical-madea` identifier.

**Tech Stack:** Python 3.10, NumPy, pandas, existing Pyomo/Gurobi runners, pytest, GitNexus.

---

### Task 1: New runner contract

**Files:**
- Create: `hierarchical_auction/madea_runner.py`
- Create: `tests/test_hierarchical_madea_runner.py`

1. Write failing tests for option normalization, MADeA helper provenance, offloaded-demand calculation, hierarchical progress, and CLI parsing.
2. Run the focused tests and confirm the module is absent.
3. Implement the helpers and CLI shell with imports from `run_faasmadea`.
4. Run the focused tests.

### Task 2: MADeA level one plus higher levels

**Files:**
- Modify: `hierarchical_auction/madea_runner.py`
- Modify: `tests/test_hierarchical_madea_runner.py`
- Modify: `tests/test_e2e_gurobi_planar.py`

1. Write failing wiring tests that constrain calls to MADeA `define_bids`, `evaluate_bids`, fixed-y validation, LSPr, hierarchy engine, and stopping criteria.
2. Implement the minimal combined iteration and persistence loop.
3. Run focused wiring and Gurobi planar tests.

### Task 3: Local and remote registration

**Files:**
- Modify: `run.py`
- Modify: `remote_experiments/jobs.py`
- Modify: `tests/test_hierarchical_madea_runner.py`
- Modify: `tests/test_remote_experiments_jobs.py`

1. Write failing parser, dispatch, command, and result-label tests.
2. Register `hierarchical-madea` without altering legacy `hierarchical` behavior.
3. Run focused tests.

### Task 4: Paper suite migration

**Files:**
- Modify: `remote_experiments/definitions/paper.py`
- Modify: `remote_experiments/definitions/smoke.py`
- Modify: `tests/test_paper_experiment_suites.py`
- Modify: `tests/test_remote_experiments_smoke_suite.py`

1. Write failing tests that paper defaults select `hierarchical-madea` and smoke exposes both variants.
2. Update method sets and algorithm-to-option mappings.
3. Run paper, smoke, materialization, and job tests.

### Task 5: Verification

1. Run focused hierarchical and remote-experiment tests.
2. Run Ruff on all changed Python files.
3. Run the complete test suite.
4. Run `git diff --check`.
5. Run GitNexus change detection and review every affected flow.
