# PLASMA Review-Findings Fix Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve the 8 code-review findings on the PLASMA implementation (branch `feat/uv-migration-and-extended-tests`).

**Architecture:** No new modules. Four behavioral fixes (config tracking, W-scaling, spare flooring, zero-load objective guard), one input validation, two deletions. Finding 4 (missing Tasks 11–14) is NOT re-planned here: it is executed from the existing `docs/plans/2026-07-04-plasma-implementation.md`, which already specifies those tasks in full.

**Tech Stack:** unchanged — Python 3.10, numpy, pandas; `uv run pytest`.

## Global Constraints

- Same as the implementation plan: 2-space indent, no new deps, never edit existing non-plasma files (`run.py` already carries its additive registration — do not touch it further).
- There are pre-existing UNCOMMITTED changes (`run.py` registration, `plasma/runner.py` termination-condition format, `tests/test_plasma_e2e.py` wiring tests) that belong to the implementation plan's Task 10. Task 1 below commits them together with the force-added config — do not lose them.
- Run `gitnexus_detect_changes()` before each commit (project CLAUDE.md rule).
- Commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Track the gitignored config + commit pending Task-10 work (finding 1)

`.gitignore:2` has `config_files/*`; every other config there was force-added. Without `-f` the new config never ships and `test_plasma_comparison_config_exists_and_has_section` fails on a clean clone.

- [ ] **Step 1: Verify current state**

Run: `git check-ignore config_files/plasma_comparison.json && git status --short`
Expected: the file is ignored; `run.py`, `plasma/runner.py`, `tests/test_plasma_e2e.py` are modified.

- [ ] **Step 2: Run the wiring tests**

Run: `uv run pytest tests/test_plasma_e2e.py tests/test_gcaa_wiring.py -q`
Expected: all PASS.

- [ ] **Step 3: Commit (force-adding the config)**

```bash
git add -f config_files/plasma_comparison.json
git add run.py plasma/runner.py tests/test_plasma_e2e.py
git commit -m "register plasma method in run.py dispatch"
```

---

### Task 2: Validate PlasmaOptions (finding 6)

**Files:**
- Modify: `plasma/core/types.py` (add `__post_init__` to `PlasmaOptions`)
- Test: `tests/test_plasma_protocol.py` (types tests live here)

**Interfaces:** unchanged; invalid options now raise `ValueError` at construction instead of a `KeyError` deep inside `PlasmaEngine.run_rounds`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_plasma_protocol.py`)

```python
def test_options_reject_nonpositive_rounds_per_step():
  with pytest.raises(ValueError, match="rounds_per_step"):
    PlasmaOptions(rounds_per_step=0)


def test_options_reject_nonpositive_w():
  with pytest.raises(ValueError, match="W"):
    PlasmaOptions(W=0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_plasma_protocol.py -v -k options_reject`
Expected: 2 FAIL (no exception raised)

- [ ] **Step 3: Implement** — add to `PlasmaOptions` in `plasma/core/types.py`, after the field list:

```python
  def __post_init__(self) -> None:
    if self.rounds_per_step < 1:
      raise ValueError(f"rounds_per_step must be >= 1, got {self.rounds_per_step}")
    if self.W <= 0.0:
      raise ValueError(f"W must be > 0, got {self.W}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_plasma_protocol.py -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add plasma/core/types.py tests/test_plasma_protocol.py
git commit -m "validate plasma options at construction"
```

---

### Task 3: Make the runner correct for W != 1 (finding 2)

**Files:**
- Modify: `plasma/runner.py:103-109`
- Test: `tests/test_plasma_e2e.py`

A round lasts `W` seconds, so a window must receive `loadt * W` arrivals; today it receives `loadt`, while capacity gates scale with `W` — with `W = 2` utilization can reach `max_utilization * 2` and the `assert feasible` at `plasma/runner.py:123` fires. Feasibility/objective must then be checked against per-window counts, i.e. `incoming_load = arrivals` (already the case — the fix is only the arrival scaling).

- [ ] **Step 1: Write the failing test** (append to `tests/test_plasma_e2e.py`)

```python
def test_runner_supports_w_not_one(tmp_path):
  config = _config(tmp_path)
  config["solver_options"]["plasma"]["W"] = 2.0
  folder = run_plasma(config, parallelism=0)  # must not trip check_feasibility
  obj = pd.read_csv(os.path.join(folder, "obj.csv"))["Plasma"]
  assert np.isfinite(obj).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_plasma_e2e.py::test_runner_supports_w_not_one -v`
Expected: FAIL — either `AssertionError: max utilization ...` from the runner's feasibility assert, or (if traffic happens to stay low) investigate: the fix below must make the test pass for the right reason. If it passes before the fix, raise the config's load (`"max": {"min": 60, "max": 80}`) until the assert fires.

- [ ] **Step 3: Implement** — in `plasma/runner.py`, scale arrivals by `W`:

```python
    arrivals = np.array([
      [int(round(loadt[(n + 1, f + 1)] * opts.W)) for f in range(Nf)]
      for n in range(Nn)
    ])
```

(No other change: `incoming_load` is already set to `arrivals`, so conservation, utilization, and the objective are all per-window and consistent. `plasma_messages.csv` already divides by `rounds_per_step * W`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_plasma_e2e.py -q`
Expected: all PASS (W=1 tests unchanged: scaling by 1.0 is identity)

- [ ] **Step 5: Commit**

```bash
git add plasma/runner.py tests/test_plasma_e2e.py
git commit -m "scale plasma arrivals by round period W"
```

---

### Task 4: Advertise floored spare in heartbeats (finding 3)

**Files:**
- Modify: `plasma/core/node.py:138-140` (`end_window`)
- Test: `tests/test_plasma_routing.py`

`_capacity_units` floors admission, but `end_window` advertises the continuous remainder: with `r*u_max*W = 2.085` and 2 admissions, `spare = 0.085 > 0` keeps neighbor gates fully open toward a node that will NACK everything.

- [ ] **Step 1: Write the failing test** (append to `tests/test_plasma_routing.py`)

```python
def test_spare_advertises_floored_capacity():
  node = _node(r=(1,), u_max=(2.085,))  # capacity_units = 2
  node.begin_window()
  assert node.admit_forward(0)
  assert node.admit_forward(0)
  assert not node.admit_forward(0)  # floored capacity exhausted
  node.end_window()
  hb = node.make_heartbeat()
  assert hb.spare[0] == 0.0  # not 0.085: nothing more is admittable
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_plasma_routing.py::test_spare_advertises_floored_capacity -v`
Expected: FAIL with `assert 0.08... == 0.0`

- [ ] **Step 3: Implement** — in `end_window`, replace the `_spare_last` assignment:

```python
    self._spare_last = np.maximum(
      0.0,
      np.array([self._capacity_units(f) for f in range(self.Nf)])
      - self._admitted,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_plasma_routing.py tests/test_plasma_clock.py -q`
Expected: all PASS (the engine scenario tests must stay green — same seeds)

- [ ] **Step 5: Commit**

```bash
git add plasma/core/node.py tests/test_plasma_routing.py
git commit -m "advertise floored spare capacity in heartbeats"
```

---

### Task 5: Guard the objective against zero-load pairs (finding 5)

**Files:**
- Modify: `plasma/runner.py` (new module-level helper + one call-site change)
- Test: `tests/test_plasma_e2e.py`

`compute_centralized_objective` divides every term by `incoming_load[(n,f)]`; a trace value < 0.5 rounds to 0 and yields inf/nan in `obj.csv`. A zero-load pair has `x = y = z = 0`, so its correct contribution is 0 — dividing by 1 instead of 0 produces exactly that. Only the objective call gets the guarded load; `check_feasibility` keeps the true zeros (0 == 0 conservation must still hold).

- [ ] **Step 1: Write the failing test** (append to `tests/test_plasma_e2e.py`)

```python
from plasma.runner import objective_load


def test_objective_load_floors_zero_pairs_only():
  load = {(1, 1): 0, (1, 2): 7}
  assert objective_load(load) == {(1, 1): 1, (1, 2): 7}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_plasma_e2e.py::test_objective_load_floors_zero_pairs_only -v`
Expected: FAIL with `ImportError: cannot import name 'objective_load'`

- [ ] **Step 3: Implement** — in `plasma/runner.py`:

```python
def objective_load(incoming_load: dict) -> dict:
  # a zero-load (n, f) contributes x = y = z = 0 to the objective; floor the
  # divisor to 1 so its contribution is exactly 0 instead of 0/0 = nan
  return {k: max(int(v), 1) for k, v in incoming_load.items()}
```

and change the objective call (runner loop) to:

```python
    obj_data = update_data(
      data, {"incoming_load": objective_load(data[None]["incoming_load"])}
    )
    obj_list.append(
      compute_centralized_objective(obj_data, res.x, res.y, res.z)
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_plasma_e2e.py -q`
Expected: all PASS (with strictly positive loads `objective_load` is the identity, so existing obj.csv values are unchanged — the determinism test confirms it)

- [ ] **Step 5: Commit**

```bash
git add plasma/runner.py tests/test_plasma_e2e.py
git commit -m "guard plasma objective against zero-load pairs"
```

---

### Task 6: Deletions — stale comment and dead accessor (findings 7, 8)

**Files:**
- Modify: `plasma/core/node.py:65` (delete the comment line `# Layer B state initialized in sb_setup (Task 7)`)
- Modify: `plasma/core/protocol.py:53-58` (delete the unused `HeartbeatCache.alpha` method; the `alpha` WIRE field stays — it is the spec's message format and feeds the paper's privacy comparison; whether to drop it from the wire is a LaTeX-note-time decision, out of scope here)

- [ ] **Step 1: Delete both**

No new tests — deletions of dead code; the existing suite is the check.

- [ ] **Step 2: Verify nothing referenced them**

Run: `grep -rn "sb_setup\|cache.alpha\|\.alpha(" plasma/ tests/test_plasma_*.py`
Expected: no matches (heartbeat `.alpha` FIELD accesses like `hb.alpha` are fine and expected).

- [ ] **Step 3: Run the full plasma suite**

Run: `uv run pytest tests/test_plasma_routing.py tests/test_plasma_sbm.py tests/test_plasma_protocol.py tests/test_plasma_clock.py tests/test_plasma_e2e.py -q`
Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add plasma/core/node.py plasma/core/protocol.py
git commit -m "drop dead heartbeat alpha accessor and stale comment"
```

---

### Task 7: Full regression + scope closure (finding 4)

- [ ] **Step 1: Full suite**

Run: `uv run pytest tests/ -q`
Expected: 541+ passed, 0 failed.

- [ ] **Step 2: Finding 4 — remaining implementation scope**

Execute Tasks 11–14 of `docs/plans/2026-07-04-plasma-implementation.md` exactly as written there (baselines, `eval/regret.py`, `tests/test_plasma_baselines.py`, LP-convergence + MILP-gap acceptance tests, `faas-plasma-note/`). They are fully specified in that plan — no re-planning needed. If the user prefers to defer them, tick this box with a note in the implementation plan marking M3+ as deferred.
