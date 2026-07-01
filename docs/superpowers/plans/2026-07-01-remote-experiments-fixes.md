# Remote Experiments Reliability Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make remote experiment provisioning portable, generated commands executable, and terminal metrics/messages accurate.

**Architecture:** Keep the current batch/manifest/runner/TUI boundaries. Change only dependency resolution, job argv construction, final CLI reporting, and session-relative throughput; each behavior gets one focused regression test before implementation.

**Tech Stack:** Python 3.10, pytest, uv, ray-dispatcher, Rich.

---

## File map

- Modify `pyproject.toml` and `uv.lock`: resolve `ray-dispatcher` from a pinned Git revision.
- Modify `remote_experiments/jobs.py`: construct Python commands, including module execution for hierarchical.
- Modify `remote_experiments/cli.py`: report terminal failures accurately.
- Modify `remote_experiments/stats.py` and `remote_experiments/tui.py`: compute throughput relative to session start.
- Modify focused `tests/test_remote_experiments_*.py` files; add one dependency-source test.
- Modify `remote_experiments/README.md`: remove the obsolete sibling-checkout prerequisite.

### Task 1: Portable ray-dispatcher dependency

**Files:**
- Create: `tests/test_remote_experiments_dependencies.py`
- Modify: `pyproject.toml:38-39`
- Modify: `uv.lock`
- Modify: `remote_experiments/README.md:6-15`

- [ ] **Step 1: Write the failing source-portability test**

```python
import tomllib
from pathlib import Path


def test_ray_dispatcher_source_is_remote_resolvable():
  config = tomllib.loads(Path("pyproject.toml").read_text())
  source = config["tool"]["uv"]["sources"]["ray-dispatcher"]
  assert source == {
    "git": "https://github.com/miciav/ray-dispatcher.git",
    "rev": "30e91c81959eab30908675102e639a8953945049",
  }
```

- [ ] **Step 2: Run the test and verify RED**

Run: `.venv/bin/pytest -q tests/test_remote_experiments_dependencies.py`

Expected: FAIL because the current source is `{"path": "../ray-dispatcher", "editable": true}`.

- [ ] **Step 3: Run GitNexus impact checks before editing configuration consumers**

Run: `npx gitnexus impact -r DFaaSOptimizer --direction upstream cmd_run`

Expected: review direct CLI/test dependants; warn before proceeding only if risk is HIGH or CRITICAL.

- [ ] **Step 4: Replace the source and refresh the lock**

Set:

```toml
[tool.uv.sources]
ray-dispatcher = { git = "https://github.com/miciav/ray-dispatcher.git", rev = "30e91c81959eab30908675102e639a8953945049" }
```

Run: `uv lock`

Update the README prerequisites to require SSH access, Gurobi, and network access for provisioning, without requiring a sibling checkout.

- [ ] **Step 5: Verify GREEN**

Run: `.venv/bin/pytest -q tests/test_remote_experiments_dependencies.py`

Expected: `1 passed`.

### Task 2: Executable job commands

**Files:**
- Modify: `tests/test_remote_experiments_jobs.py`
- Modify: `remote_experiments/jobs.py:12-38`

- [ ] **Step 1: Change command expectations to the required behavior**

```python
def test_experiment_to_job_builds_expected_command_for_plain_algorithm(tmp_path):
  job = experiment_to_job(_experiment("centralized"), tmp_path)
  assert job.command == (
    "python", "run_centralized_model.py", "-c", "config.json", "--disable_plotting",
  )


def test_experiment_to_job_runs_hierarchical_as_module(tmp_path):
  job = experiment_to_job(_experiment("hierarchical"), tmp_path)
  assert job.command == (
    "python", "-m", "hierarchical_auction.runner", "-c", "config.json",
    "--disable_plotting",
  )
```

Also update the variant expectation to start with `("python", "decentralized_bestresponse.py", ...)`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `.venv/bin/pytest -q tests/test_remote_experiments_jobs.py`

Expected: three command assertions fail because commands currently start with `uv run` and hierarchical is a file path.

- [ ] **Step 3: Run GitNexus impact analysis**

Run: `npx gitnexus impact -r DFaaSOptimizer --direction upstream experiment_to_job`

Expected: direct callers are `cmd_run` and job tests; report risk before editing.

- [ ] **Step 4: Implement the minimal command mapping**

Store each algorithm's executable tail rather than only a script:

```python
COMMAND_BY_ALGORITHM = {
  "centralized": ("run_centralized_model.py",),
  "hierarchical": ("-m", "hierarchical_auction.runner"),
  # existing scripts and variant arguments follow the same tuple format
}
```

Build the job command as:

```python
command = COMMAND_BY_ALGORITHM[experiment.algorithm]
Job(command=("python", *command, "-c", "config.json", "--disable_plotting"), ...)
```

Keep the existing public constant name if changing it would add unnecessary churn.

- [ ] **Step 5: Verify GREEN and the real entry point**

Run: `.venv/bin/pytest -q tests/test_remote_experiments_jobs.py`

Run: `MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python -m hierarchical_auction.runner --help`

Expected: all job tests pass and module help exits 0.

### Task 3: Accurate terminal message

**Files:**
- Modify: `tests/test_remote_experiments_cli.py`
- Modify: `remote_experiments/cli.py:65-69`

- [ ] **Step 1: Add a failed-dispatcher CLI test**

Create a fake derived from `_FakeDispatcher` whose status sequence ends in `JobStatus.FAILED`, run `cmd_run`, capture stdout with `capsys`, and assert:

```python
assert "batch finished with 1 unsuccessful experiment" in capsys.readouterr().out
assert "batch complete" not in capsys.readouterr().out
```

Read captured output once and store it in a variable before both assertions.

- [ ] **Step 2: Run the test and verify RED**

Run: `.venv/bin/pytest -q tests/test_remote_experiments_cli.py::test_cmd_run_reports_terminal_failures`

Expected: FAIL because the CLI prints `batch complete`.

- [ ] **Step 3: Run GitNexus impact analysis**

Run: `npx gitnexus impact -r DFaaSOptimizer --direction upstream cmd_run`

Expected: CLI tests are direct dependants; report risk before editing.

- [ ] **Step 4: Implement selected-experiment reporting**

After `run_batch`, retain the interrupted message when `completed` is false. Otherwise count selected experiments whose manifest status is not `SUCCEEDED`; print `batch complete` only when the count is zero, else print `batch finished with N unsuccessful experiment(s)`.

- [ ] **Step 5: Verify GREEN**

Run: `.venv/bin/pytest -q tests/test_remote_experiments_cli.py`

Expected: all CLI tests pass.

### Task 4: Session-relative throughput

**Files:**
- Modify: `tests/test_remote_experiments_stats.py`
- Modify: `remote_experiments/stats.py:46-67`
- Modify: `remote_experiments/tui.py:65-76`

- [ ] **Step 1: Add the resumed-session regression test**

```python
def test_summarize_throughput_excludes_successes_from_before_session(tmp_path):
  manifest = Manifest(tmp_path / "m.json")
  manifest.record("old", status="succeeded", duration_s=5.0)
  manifest.record("new", status="succeeded", duration_s=5.0)
  stats = summarize(
    ["old", "new"], manifest, _inventory(), {}, elapsed_s=30.0,
    succeeded_at_start=1,
  )
  assert stats.throughput_per_min == 2.0
```

- [ ] **Step 2: Run the test and verify RED**

Run: `.venv/bin/pytest -q tests/test_remote_experiments_stats.py::test_summarize_throughput_excludes_successes_from_before_session`

Expected: FAIL because `summarize` does not accept `succeeded_at_start`.

- [ ] **Step 3: Run GitNexus impact analysis for both edited symbols**

Run: `npx gitnexus impact -r DFaaSOptimizer --direction upstream summarize`

Run: `npx gitnexus impact -r DFaaSOptimizer --direction upstream live_view`

Expected: TUI and tests are direct dependants; report risk before editing.

- [ ] **Step 4: Implement baseline subtraction**

Add keyword parameter `succeeded_at_start: int = 0` to `summarize` and compute:

```python
session_succeeded = max(succeeded - succeeded_at_start, 0)
throughput_per_min = session_succeeded / elapsed_s * 60 if elapsed_s > 0 else 0.0
```

In `live_view`, calculate the initial number of succeeded experiment IDs once before opening `Live`, then pass it to every `summarize` call.

- [ ] **Step 5: Verify GREEN**

Run: `.venv/bin/pytest -q tests/test_remote_experiments_stats.py tests/test_remote_experiments_tui.py`

Expected: all stats and TUI tests pass.

### Task 5: Full verification and scope audit

**Files:**
- Verify all files changed above.

- [ ] **Step 1: Run the complete focused suite**

Run: `.venv/bin/pytest -q tests/test_remote_experiments_*.py`

Expected: all tests pass.

- [ ] **Step 2: Verify dependency lock and command entry point**

Run: `uv lock --check`

Run: `MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python -m hierarchical_auction.runner --help`

Expected: both commands exit 0.

- [ ] **Step 3: Check diff quality**

Run: `git diff --check`

Expected: no output and exit 0.

- [ ] **Step 4: Run GitNexus change detection**

Run: `npx gitnexus cypher -r DFaaSOptimizer "MATCH (n) WHERE n.filePath STARTS WITH 'remote_experiments/' RETURN n.filePath, n.name LIMIT 5"`

Run the available GitNexus change-detection tool for the working tree; if the CLI lacks `detect-changes`, use the configured MCP tool. Confirm only dependency provisioning, job construction, CLI reporting, stats/TUI, docs, and tests are affected.

- [ ] **Step 5: Report completion without committing implementation unless requested**

Summarize changed behavior and exact verification results. Preserve the user's untracked `test_instances/2026-06-30_13-54-13.841040/` directory.
