# FaaS-MALD Findings Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make FaaS-MALD Cloud-consistent, prevent premature stopping after replica creation, expose auditable fixed-capacity certificates, and remove MALD-only dead paths and dense-graph overhead.

**Architecture:** Keep the public runner and CLI unchanged. Inside MALD, optimize Cloud-relative advantage `beta - latency - fairness + gamma`, retain the primal assignment and price vector that produced the certificate, use outer memory bids as the sole replica mechanism, and record one certificate row per outer coordination iteration. Existing coordinators and shared helpers remain untouched.

**Tech Stack:** Python 3.10, numpy, pandas, scipy, Pyomo/Gurobi for optional E2E tests, pytest, LaTeX/latexmk.

---

## Global Constraints

- Modify only `decentralized_dual.py`, `tests/test_dual_helpers.py`, `tests/test_dual_e2e.py`, `faas-mald-note/faas-mald.tex`, and `faas-mald-note/README.md`.
- Do not modify MADiG, MABR, MAPoD, MADEA, shared solver helpers, configs, or method registries.
- Preserve unrelated dirty-worktree changes and stage only files named by each task.
- Before editing each existing MALD symbol, run `gitnexus impact -r DFaaSOptimizer -d upstream --depth 3 --include-tests <symbol>` and report the blast radius. Current expected risk is MEDIUM for `buyer_price_response` and LOW for `dual_coordination_round`.
- Before every commit, run the available GitNexus change-scope check. If the local CLI does not expose `detect_changes`, record that limitation and verify staged scope with `git diff --cached --name-only` and `git diff --cached --check`.
- Follow 2-space indentation and existing repository import conventions.
- Use TDD: write the failing test, observe the intended failure, implement the minimum fix, rerun the focused and neighboring suites.
- Use `uv run pytest ...` from the repository root.
- One logical commit per task with a short imperative subject.

## File Responsibilities

- `decentralized_dual.py`: Cloud-relative pair advantages, dual response/round, capacity-state refresh, runner, certificate CSV.
- `tests/test_dual_helpers.py`: pure mathematical, validation, sparse traversal, retained-primal, and price-identity tests.
- `tests/test_dual_e2e.py`: runner artifacts, replica lifecycle helper, reproducibility, defaults.
- `faas-mald-note/faas-mald.tex`: equations, algorithm, certificate scope, complexity.
- `faas-mald-note/README.md`: concise fixed-capacity certificate description and artifact list.

---

### Task 1: Cloud-relative advantage, option validation, and bound-producing prices

**Files:**
- Modify: `decentralized_dual.py:pair_scores`
- Modify: `decentralized_dual.py:buyer_price_response`
- Modify: `decentralized_dual.py:dual_coordination_round`
- Test: `tests/test_dual_helpers.py`

- [ ] **Step 1: Run impact analysis**

Run:

```bash
gitnexus impact -r DFaaSOptimizer -d upstream --depth 3 --include-tests pair_scores
gitnexus impact -r DFaaSOptimizer -d upstream --depth 3 --include-tests buyer_price_response
gitnexus impact -r DFaaSOptimizer -d upstream --depth 3 --include-tests dual_coordination_round
```

Expected: only the MALD round, MALD runner, and MALD tests are affected; warn before editing if risk is HIGH or CRITICAL.

- [ ] **Step 2: Add failing Cloud-relative regression tests**

Append to `tests/test_dual_helpers.py`:

```python
def test_negative_node_score_better_than_cloud_is_selected():
  data = make_data(Nn=2, Nf=1, gamma=0.10)
  data[None]["beta"][(1, 2, 1)] = -0.05
  neighborhood = full_neighborhood(2)
  advantage, eligible = pair_scores(
    data, neighborhood, np.zeros((2, 2)), np.zeros((2, 1)), DUAL_OPTIONS
  )
  bids, demand, buyer_term = buyer_price_response(
    np.array([[1.0], [0.0]]), np.ones((2, 1)), np.zeros((2, 1)),
    advantage, eligible,
  )
  assert eligible[0, 1, 0]
  assert advantage[0, 1, 0] == pytest.approx(0.05)
  assert bids.to_dict("records") == [
    {"i": 0, "j": 1, "f": 0, "d": 1.0, "utility": 0.05}
  ]
  assert demand[1, 0] == 1.0
  assert buyer_term == pytest.approx(0.05)


def test_cloud_relative_filter_rejects_node_worse_than_cloud():
  data = make_data(Nn=2, Nf=1, gamma=0.10)
  data[None]["beta"][(1, 2, 1)] = -0.11
  advantage, eligible = pair_scores(
    data, full_neighborhood(2), np.zeros((2, 2)), np.zeros((2, 1)),
    DUAL_OPTIONS,
  )
  assert not eligible[0, 1, 0]
  assert advantage[0, 1, 0] == -np.inf
```

Add `import pytest` at the top of the test file.

- [ ] **Step 3: Run the new tests to verify RED**

Run:

```bash
uv run pytest \
  tests/test_dual_helpers.py::test_negative_node_score_better_than_cloud_is_selected \
  tests/test_dual_helpers.py::test_cloud_relative_filter_rejects_node_worse_than_cloud \
  -v
```

Expected: the first test fails because the current score is `-0.05` and no bid is emitted.

- [ ] **Step 4: Implement Cloud-relative pair advantage with sparse traversal**

Replace `pair_scores` with:

```python
def pair_scores(
  data: dict,
  neighborhood: np.array,
  latency: np.array,
  fairness: np.array,
  dual_options: dict,
) -> Tuple[np.array, np.array]:
  """Return Cloud-relative pair advantages and their eligibility mask."""
  values = data[None]
  nn = values["Nn"][None]
  nf = values["Nf"][None]
  advantages = np.full((nn, nn, nf), -np.inf)
  eligible = np.zeros((nn, nn, nf), dtype=bool)
  for i in range(nn):
    for j in np.flatnonzero(neighborhood[i]):
      j = int(j)
      for f in range(nf):
        advantage = (
          values["beta"][(i + 1, j + 1, f + 1)]
          - dual_options["latency_weight"] * latency[i, j]
          - dual_options["fairness_weight"] * fairness[i, f]
          + values["gamma"][(i + 1, f + 1)]
        )
        if advantage > 0:
          advantages[i, j, f] = advantage
          eligible[i, j, f] = True
  return advantages, eligible
```

The returned quantity is now an advantage over Cloud, not the raw horizontal score. Keep the function name to avoid an unnecessary public rename.

- [ ] **Step 5: Make buyer traversal sparse and deterministic**

Inside `buyer_price_response`, replace dense adjusted-score construction with:

```python
    sellers = np.flatnonzero(elig[i, :, f])
    if not len(sellers):
      continue
    adjusted = s[i, sellers, f] - lam[sellers, f]
    positive = adjusted > 0
    sellers = sellers[positive]
    adjusted = adjusted[positive]
    if not len(sellers):
      continue
    order = np.lexsort((sellers, -adjusted))
    sellers = sellers[order]
    adjusted = adjusted[order]
    best = int(sellers[0])
    demand[best, f] += omega[i, f]
    dual_term += omega[i, f] * adjusted[0]

    remaining = float(omega[i, f])
    for j in sellers:
      j = int(j)
      quantity = min(remaining, float(capacity[j, f]))
      if quantity > 0:
        rows.append(
          {"i": int(i), "j": j, "f": int(f), "d": quantity,
           "utility": float(s[i, j, f])}
        )
        remaining -= quantity
      if remaining <= 0:
        break
```

Update the docstring to say that `s` contains Cloud-relative advantages.

- [ ] **Step 6: Add failing option-validation and best-price tests**

Append:

```python
@pytest.mark.parametrize(
  ("overrides", "message"),
  [
    ({"alpha0": 0.0}, "alpha0"),
    ({"alpha0": np.nan}, "alpha0"),
    ({"theta": 0.0, "step_rule": "polyak"}, "theta"),
    ({"theta": np.inf, "step_rule": "polyak"}, "theta"),
    ({"gap_tolerance": -1.0}, "gap_tolerance"),
    ({"gap_tolerance": np.nan}, "gap_tolerance"),
    ({"latency_weight": np.inf}, "latency_weight"),
    ({"fairness_weight": np.nan}, "fairness_weight"),
  ],
)
def test_invalid_dual_numeric_options_are_rejected(overrides, message):
  data, neighborhood, omega, capacity = dual_round_setup()
  options = {**DUAL_ROUND_OPTIONS, **overrides}
  with pytest.raises(ValueError, match=message):
    run_round(data, neighborhood, omega, capacity, options)


def test_best_lambda_reproduces_reported_upper_bound():
  data, neighborhood, omega, capacity = dual_round_setup(seed=19)
  _, _, _, gap_info, _ = run_round(data, neighborhood, omega, capacity)
  scores, eligible = pair_scores(
    data, neighborhood, np.zeros_like(neighborhood), np.zeros_like(omega),
    DUAL_ROUND_OPTIONS,
  )
  _, _, buyer_term = buyer_price_response(
    omega, capacity, gap_info["best_lam"], scores, eligible
  )
  reproduced = float((gap_info["best_lam"] * capacity).sum() + buyer_term)
  assert reproduced == pytest.approx(gap_info["UB"])
```

Task 2 later updates this unpacking when it simplifies the round return contract.

- [ ] **Step 7: Run validation/price tests to verify RED**

Run:

```bash
uv run pytest tests/test_dual_helpers.py \
  -k "invalid_dual_numeric_options or best_lambda" -v
```

Expected: missing validation cases fail and `best_lam` raises `KeyError`.

- [ ] **Step 8: Implement validation and retain the UB-producing lambda**

At the start of `dual_coordination_round`, validate:

```python
  max_inner = dual_options["max_inner_iterations"]
  if isinstance(max_inner, bool) or not isinstance(max_inner, (int, np.integer)):
    raise ValueError("max_inner_iterations must be an integer")
  if max_inner < 1:
    raise ValueError("max_inner_iterations must be at least 1")
  step_rule = dual_options["step_rule"]
  if step_rule not in {"sqrt", "polyak"}:
    raise ValueError("step_rule must be 'sqrt' or 'polyak'")
  if not np.isfinite(dual_options["gap_tolerance"]) or (
      dual_options["gap_tolerance"] < 0
    ):
    raise ValueError("gap_tolerance must be finite and non-negative")
  for name in ("latency_weight", "fairness_weight"):
    if not np.isfinite(dual_options[name]):
      raise ValueError(f"{name} must be finite")
  step_name = "alpha0" if step_rule == "sqrt" else "theta"
  if not np.isfinite(dual_options[step_name]) or dual_options[step_name] <= 0:
    raise ValueError(f"{step_name} must be finite and positive")
```

Initialize `best_lam = lam.copy()`. Replace direct UB minimization with:

```python
    current_ub = float((lam * capacity).sum() + buyer_term)
    if current_ub < best_ub:
      best_ub = current_ub
      best_lam = lam.copy()
```

Expose both vectors explicitly:

```python
  gap_info = {
    "LB": best_lb,
    "UB": best_ub,
    "gap": gap,
    "inner_iterations": k,
    "best_lam": best_lam,
    "final_lam": lam,
    "lb_history": lb_history,
  }
```

- [ ] **Step 9: Update existing helper expectations and run Task 1 suite**

Update the existing exact expectations:

```python
# test_pair_scores_masks_non_neighbors_and_dominated_pairs
assert eligible[1, 0, 0] and scores[1, 0, 0] == pytest.approx(1.05)

# test_buyer_response_zero_prices_picks_best_score_seller
assert dual_term == pytest.approx(5.0 * 2.05)

# test_buyer_response_price_shifts_demand
assert dual_term == pytest.approx(5.0 * 1.05)
```

The tie-order assertion remains `[1, 2]`. The LP oracle needs no structural
change because it already consumes the array returned by `pair_scores`; after
this task that array contains Cloud-relative advantages.

Run:

```bash
uv run pytest tests/test_dual_helpers.py -v
```

Expected: all helper tests pass.

- [ ] **Step 10: Commit Task 1**

Run the required scope checks, then:

```bash
git add decentralized_dual.py tests/test_dual_helpers.py
git commit -m "align FaaS-MALD rewards with Cloud rejection"
```

---

### Task 2: Retain the best primal assignment and remove dead round paths

**Files:**
- Modify: `decentralized_dual.py:dual_coordination_round`
- Modify: `tests/test_dual_helpers.py`

- [ ] **Step 1: Run impact analysis**

```bash
gitnexus impact -r DFaaSOptimizer -d upstream --depth 3 --include-tests dual_coordination_round
```

- [ ] **Step 2: Add a failing retained-assignment test**

Use `monkeypatch` to count evaluator calls:

```python
def test_round_returns_retained_best_y_without_post_loop_reevaluation(monkeypatch):
  data, neighborhood, omega, capacity = dual_round_setup(seed=23)
  calls = 0
  original = decentralized_dual.evaluate_assignments

  def counted(*args, **kwargs):
    nonlocal calls
    calls += 1
    return original(*args, **kwargs)

  monkeypatch.setattr(decentralized_dual, "evaluate_assignments", counted)
  options = {**DUAL_ROUND_OPTIONS, "max_inner_iterations": 7,
             "gap_tolerance": 0.0}
  y_inc, _, _, gap_info, _ = run_round(
    data, neighborhood, omega, capacity, options
  )
  assert calls == gap_info["inner_iterations"]
  assert y_inc.sum() >= 0
```

Add `import decentralized_dual` to the test file.

- [ ] **Step 3: Verify RED**

```bash
uv run pytest \
  tests/test_dual_helpers.py::test_round_returns_retained_best_y_without_post_loop_reevaluation \
  -v
```

Expected: evaluator calls exceed `gap_info["inner_iterations"]` because the current round re-evaluates the best bids after the loop.

- [ ] **Step 4: Simplify the round state and return contract**

Change the signature to remove `force_memory_bids`, `ell`, and `r`:

```python
def dual_coordination_round(
  omega: np.array,
  residual_capacity: np.array,
  data: dict,
  neighborhood: np.array,
  rho: np.array,
  dual_options: dict,
  latency: np.array,
  fairness: np.array,
) -> Tuple[np.array, pd.DataFrame, dict, int]:
```

Initialize:

```python
  best_y = np.zeros((nn, nn, nf))
```

When LB improves:

```python
      if candidate_lb > best_lb:
        best_lb = candidate_lb
        best_y = candidate_y.copy()
```

Delete `best_bids`, the post-loop evaluator calls, tentative replica handling, and `additional_replicas`. Build memory bids from `best_y`:

```python
  memory_rows = []
  placed = best_y.sum(axis=1)
  for i, f in zip(*np.nonzero(omega > 0)):
    i, f = int(i), int(f)
    if placed[i, f] + 1e-12 >= omega[i, f]:
      continue
    memory_requirement = data[None]["memory_requirement"][f + 1]
    for j in np.flatnonzero(neighborhood[i]):
      j = int(j)
      if rho[j] >= memory_requirement:
        memory_rows.append({"i": i, "j": j, "f": f})
  memory_bids = pd.DataFrame(memory_rows, columns=["i", "j", "f"])
  return best_y, memory_bids, gap_info, n_active
```

- [ ] **Step 5: Update helper calls and tests**

Update `run_round`:

```python
  return dual_coordination_round(
    omega, capacity, data, neighborhood,
    rho=np.zeros(Nn), dual_options=options or DUAL_ROUND_OPTIONS,
    latency=np.zeros((Nn, Nn)), fairness=np.zeros((Nn, Nf)),
  )
```

Update all unpacking from five values to four:

```python
y_inc, memory_bids, gap_info, n_active = run_round(...)
```

Update the retained-assignment test introduced in Step 2 to:

```python
  y_inc, _, gap_info, _ = run_round(
    data, neighborhood, omega, capacity, options
  )
```

Update the Task 1 best-price test to:

```python
  _, _, gap_info, _ = run_round(data, neighborhood, omega, capacity)
```

Replace the old forced-memory test with:

```python
def test_memory_bids_are_emitted_only_for_unplaced_demand():
  Nn, Nf = 2, 1
  data = make_data(Nn, Nf)
  neighborhood = full_neighborhood(Nn)
  rho = np.array([0.0, 2.0])
  common = dict(
    data=data, neighborhood=neighborhood, rho=rho,
    dual_options=DUAL_ROUND_OPTIONS,
    latency=np.zeros((Nn, Nn)), fairness=np.zeros((Nn, Nf)),
  )
  full = dual_coordination_round(
    omega=np.array([[1.0], [0.0]]),
    residual_capacity=np.array([[0.0], [1.0]]), **common,
  )[1]
  short = dual_coordination_round(
    omega=np.array([[2.0], [0.0]]),
    residual_capacity=np.array([[0.0], [1.0]]), **common,
  )[1]
  assert full.empty
  assert short.to_dict("records") == [{"i": 0, "j": 1, "f": 0}]
```

- [ ] **Step 6: Run helper and sibling tests**

```bash
uv run pytest tests/test_dual_helpers.py tests/test_diffusion_helpers.py -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 2**

```bash
git add decentralized_dual.py tests/test_dual_helpers.py
git commit -m "simplify FaaS-MALD dual recovery"
```

---

### Task 3: Repair replica lifecycle and write per-round certificates

**Files:**
- Modify: `decentralized_dual.py:run`
- Modify: `tests/test_dual_e2e.py`

- [ ] **Step 1: Run impact analysis**

```bash
gitnexus context -r DFaaSOptimizer -f decentralized_dual.py run
gitnexus impact -r DFaaSOptimizer -d upstream --depth 3 --include-tests \
  Function:decentralized_dual.py:run
```

- [ ] **Step 2: Add a pure capacity-state helper and failing lifecycle test**

Add the test first:

```python
from decentralized_dual import _capacity_state


def test_capacity_state_reflects_newly_started_replicas():
  data = {None: {
    "Nn": {None: 1}, "Nf": {None: 1},
    "max_utilization": {1: 0.8}, "demand": {(1, 1): 1.0},
  }}
  x = np.zeros((1, 1))
  y = np.zeros((1, 1, 1))
  before = _capacity_state(x, y, np.zeros((1, 1)), data)
  after = _capacity_state(x, y, np.ones((1, 1)), data)
  assert before[3][0, 0] == 0.0
  assert after[3][0, 0] == pytest.approx(0.8)
```

- [ ] **Step 3: Verify RED**

```bash
uv run pytest \
  tests/test_dual_e2e.py::test_capacity_state_reflects_newly_started_replicas \
  -v
```

Expected: import fails because `_capacity_state` does not exist.

- [ ] **Step 4: Implement and use `_capacity_state`**

Add before `run`:

```python
def _capacity_state(
  sp_x: np.array,
  y: np.array,
  sp_r: np.array,
  sp_data: dict,
) -> Tuple[np.array, np.array, np.array, np.array]:
  capacity, residual_capacity, ell = compute_residual_capacity(
    sp_x, y, sp_r, sp_data
  )
  blackboard = np.maximum(0.0, capacity - sp_x)
  return capacity, residual_capacity, ell, blackboard
```

Use it at the top of each outer coordination iteration. After `sp_r += additional_replicas`, call it again when any increment exceeds tolerance:

```python
      if len(memory_bids) > 0:
        s = datetime.now()
        additional_replicas, sp_rho = start_additional_replicas(
          memory_bids, sp_r, sp_data, sp_rho
        )
        sp_r += additional_replicas
        total_runtime += (datetime.now() - s).total_seconds()
        if (additional_replicas > tolerance).any():
          capacity, residual_capacity, ell, blackboard = _capacity_state(
            sp_x, y, sp_r, sp_data
          )
```

- [ ] **Step 5: Adapt the round call and remove dead runner state**

Remove `deque` import, `n_accepted_queue`, and the forced-memory expression. Call:

```python
      y_inc, memory_bids, gap_info, n_active = dual_coordination_round(
        omega, residual_capacity, sp_data, neighborhood, coordination_rho,
        dual_options, latency, fairness,
      )
      additional_replicas = np.zeros((Nn, Nf))
```

Keep the no-progress guard after memory-bid processing so a positive replica increment keeps the loop alive.

- [ ] **Step 6: Add certificate-row collection**

Initialize before the timestep loop:

```python
  certificate_rows = []
```

After shared stopping criteria and the no-progress guard determine the reason, append:

```python
      certificate_rows.append({
        "timestep": t,
        "outer_iteration": it,
        "LB": gap_info["LB"],
        "UB": gap_info["UB"],
        "gap": gap_info["gap"],
        "inner_iterations": gap_info["inner_iterations"],
        "stop_reason": why_stop_searching if stop_searching else "",
      })
```

At output time:

```python
  pd.DataFrame(
    certificate_rows,
    columns=[
      "timestep", "outer_iteration", "LB", "UB", "gap",
      "inner_iterations", "stop_reason",
    ],
  ).to_csv(
    os.path.join(solution_folder, "coordination_certificate.csv"), index=False
  )
```

Change termination wording to `fixed-C gap`, `fixed-C LB`, and `fixed-C UB`.

- [ ] **Step 7: Extend E2E artifact assertions**

In `test_dual_runner_produces_expected_artifacts_with_gap`, add:

```python
  certificate = pd.read_csv(Path(folder, "coordination_certificate.csv"))
  assert list(certificate.columns) == [
    "timestep", "outer_iteration", "LB", "UB", "gap",
    "inner_iterations", "stop_reason",
  ]
  assert len(certificate) >= 1
  assert (certificate["UB"] + 1e-8 >= certificate["LB"]).all()
  assert (certificate["gap"] >= -1e-8).all()
  assert tc.iloc[:, -1].astype(str).str.contains("fixed-C gap:").all()
```

Extend reproducibility to compare stable certificate columns excluding runtime and free-form stop text:

```python
  cert_columns = ["timestep", "outer_iteration", "LB", "UB", "gap",
                  "inner_iterations"]
  cert_a = pd.read_csv(Path(folder_a, "coordination_certificate.csv"))
  cert_b = pd.read_csv(Path(folder_b, "coordination_certificate.csv"))
  pd.testing.assert_frame_equal(cert_a[cert_columns], cert_b[cert_columns])
```

- [ ] **Step 8: Run runner and sibling suites**

```bash
uv run pytest \
  tests/test_dual_helpers.py tests/test_dual_e2e.py \
  tests/test_diffusion_helpers.py tests/test_powerd_helpers.py \
  -v
uv run python decentralized_dual.py --help
```

Expected: all tests pass; Gurobi E2E skips only when Gurobi is unavailable; CLI exits zero.

- [ ] **Step 9: Commit Task 3**

```bash
git add decentralized_dual.py tests/test_dual_e2e.py
git commit -m "repair FaaS-MALD replica lifecycle and certificates"
```

---

### Task 4: Align the technical note and README

**Files:**
- Modify: `faas-mald-note/faas-mald.tex`
- Modify: `faas-mald-note/README.md`

- [ ] **Step 1: Update the mathematical formulation**

Define:

```latex
a_{ij}^f = s_{ij}^f + \gamma_i^f
  = \beta_{ij}^f - w_{\mathrm{lat}}L_{ij}
    - w_{\mathrm{fair}}\phi_i^f + \gamma_i^f .
```

State that Cloud is the zero incremental baseline and eligibility is
`a_{ij}^f > 0`, equivalent to `s_{ij}^f > -\gamma_i^f`.

Replace the LP objective and dual buyer term with `a`:

```latex
\max_{y\geq0}\ \sum_{i,j,f} a_{ij}^f y_{ij}^f
```

```latex
g(\lambda)=\sum_{j,f}\lambda_j^fC_j^f+
\sum_{i,f}\omega_i^f\max\!\left\{0,
\max_j(a_{ij}^f-\lambda_j^f)\right\}.
```

- [ ] **Step 2: Update algorithm and certificate artifacts**

The pseudocode must show:

- validation of numeric options;
- sparse neighbor/eligible traversal;
- `best_y`, `best_lam`, and `final_lam`;
- no tentative replicas;
- no forced-memory branch;
- return `(best_y, memory_bids, gap_info, n_active)`;
- outer memory bids as the only replica mechanism;
- capacity refresh before stopping;
- `coordination_certificate.csv` as one row per outer iteration.

Retain the warning that the certificate covers only the fixed-capacity inner LP.

- [ ] **Step 3: Update complexity and README wording**

State score evaluation as `O(|E| |F|)` and buyer ranking as
`O(sum_{i,f} |A_i^f| log |A_i^f|)`. Do not claim that the practical gap-based
`polyak` option has classical Polyak convergence guarantees.

Change the README introduction to:

```markdown
A focused, paper-ready LaTeX section describing FaaS-MALD and its
fixed-residual-capacity transportation-LP certificate.
```

Add `coordination_certificate.csv` to the listed runtime artifacts.

- [ ] **Step 4: Compile and visually verify**

Run:

```bash
cd faas-mald-note
latexmk -C main.tex
latexmk -pdf main.tex
pdftoppm -png main.pdf /tmp/faas-mald-remediation
```

Check `main.log` for undefined references, undefined citations, duplicate destinations, and overfull boxes. Inspect every rendered PNG for clipping, overlap, malformed equations, and unreadable algorithm lines. Run `latexmk -c main.tex` after inspection.

Expected: PDF compiles; no undefined references/citations or overfull boxes; visual inspection passes.

- [ ] **Step 5: Cross-check note against code**

Read `pair_scores`, `buyer_price_response`, `dual_coordination_round`, and the MALD runner loop line-by-line. Confirm that reward, thresholds, state, return values, replica lifecycle, certificate fields, and complexity match the note. Fix the note, never the validated code, for documentation-only discrepancies.

- [ ] **Step 6: Commit Task 4**

```bash
git add faas-mald-note/faas-mald.tex faas-mald-note/README.md
git commit -m "align FaaS-MALD note with remediated algorithm"
```

---

### Task 5: Final scope and regression verification

**Files:**
- Verify only; no planned source modifications.

- [ ] **Step 1: Verify changed scope with GitNexus and Git**

Run the available GitNexus change detector. Then run:

```bash
git diff --check
git diff --name-only b4d1848..HEAD
```

Expected changed implementation paths:

```text
decentralized_dual.py
tests/test_dual_helpers.py
tests/test_dual_e2e.py
faas-mald-note/faas-mald.tex
faas-mald-note/README.md
```

No unrelated user changes may be staged or committed.

- [ ] **Step 2: Run MALD and sibling verification**

```bash
uv run pytest \
  tests/test_dual_helpers.py tests/test_dual_e2e.py \
  tests/test_diffusion_helpers.py tests/test_diffusion_e2e.py \
  tests/test_powerd_helpers.py tests/test_powerd_e2e.py \
  -v
```

Expected: all available tests pass; solver-dependent tests skip only when their solver is unavailable.

- [ ] **Step 3: Run the complete repository suite**

```bash
uv run pytest -v
```

Expected: zero failures. Record exact passed/skipped counts and warnings.

- [ ] **Step 4: Final mathematical audit**

Confirm with the SciPy oracle tests that:

```text
LB <= LP optimum <= UB
```

Confirm the oracle objective, buyer response, primal LB, dual UB, and LaTeX note all use the same Cloud-relative advantage `s + gamma`.

- [ ] **Step 5: Final code review**

Request one independent final review of commits created by Tasks 1-4. The reviewer must check:

- Cloud decision parity with sibling algorithms;
- stale-blackboard regression;
- certificate/price identity;
- absence of tentative/forced-memory dead paths;
- sparse traversal;
- runner/CLI compatibility;
- note/code agreement.

Resolve every Critical or Important finding and rerun affected tests before completion.

---

## Self-Review

- Spec coverage: every requirement in `docs/superpowers/specs/2026-07-02-faas-mald-findings-remediation-design.md` maps to Tasks 1-4; Task 5 verifies the combined result.
- Scope: no existing coordinator or shared helper is modified; common-runner extraction remains out of scope.
- Type consistency: the final round return is consistently `(np.array, pd.DataFrame, dict, int)` in implementation, runner, tests, and note.
- Certificate consistency: advantage, LB, UB, oracle, `best_lam`, CSV, and note use the same fixed-capacity LP.
- Placeholder scan: no deferred steps or unspecified error handling remain.
