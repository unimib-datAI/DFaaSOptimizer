# Centralized Feasibility Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every heuristic generate and report only solutions feasible for `LoadManagementModel`, with fail-fast diagnostics as a final safety check.

**Architecture:** Add a complete array-level validator beside the existing lightweight feasibility helper, preserving that high-risk helper's signature. Invoke the validator once in `combine_solutions`, the shared assembly point used by all command-line heuristic runners. Prevent the hierarchical engine from requesting non-neighbor or ping-pong allocations so validation is an invariant check rather than normal control flow.

**Tech Stack:** Python 3.10, NumPy, pytest, Pyomo/Gurobi for the final comparison.

---

### Task 1: Canonical centralized-feasibility validator

**Files:**
- Modify: `utils/centralized.py`
- Create: `tests/test_centralized_feasibility.py`

- [ ] **Step 1: Re-run impact analysis immediately before editing**

Run GitNexus impact for the existing `check_feasibility` symbol and record that its current signature remains unchanged. The known result is CRITICAL: 44 dependents and 9 affected execution flows. This task adds a new sibling function rather than changing those callers.

- [ ] **Step 2: Write failing validator tests**

Create a compact two-node, one-function fixture and tests for a valid solution plus every constraint family:

```python
import numpy as np
import pytest

from utils.centralized import validate_centralized_solution


def _data():
  return {None: {
    "Nn": {None: 2},
    "Nf": {None: 1},
    "incoming_load": {(1, 1): 2.0, (2, 1): 2.0},
    "neighborhood": {(1, 1): 0, (1, 2): 1, (2, 1): 1, (2, 2): 0},
    "demand": {(1, 1): 0.4, (2, 1): 0.4},
    "max_utilization": {1: 0.8},
    "memory_requirement": {1: 1},
    "memory_capacity": {1: 4, 2: 4},
  }}


def _solution():
  return (
    np.array([[2.0], [2.0]]),
    np.zeros((2, 2, 1)),
    np.zeros((2, 1)),
    np.ones((2, 1)),
  )


def test_validate_centralized_solution_accepts_valid_arrays():
  x, y, z, r = _solution()
  validate_centralized_solution(x, y, z, r, _data())


def test_validate_centralized_solution_rejects_non_neighbor():
  x, y, z, r = _solution()
  y[0, 0, 0] = 1.0
  x[0, 0] = 1.0
  with pytest.raises(ValueError, match="offload_only_to_neighbors"):
    validate_centralized_solution(x, y, z, r, _data())


def test_validate_centralized_solution_rejects_ping_pong():
  x, y, z, r = _solution()
  x[:, 0] = 1.0
  y[0, 1, 0] = y[1, 0, 0] = 1.0
  with pytest.raises(ValueError, match="no_ping_pong"):
    validate_centralized_solution(x, y, z, r, _data())


@pytest.mark.parametrize(
  ("field", "value", "message"),
  [
    ("x", 1.0, "no_traffic_loss"),
    ("r", 0.0, "utilization_equilibrium"),
    ("r", 3.0, "utilization_equilibrium2"),
  ],
)
def test_validate_centralized_solution_rejects_balance_and_utilization(
  field, value, message,
):
  x, y, z, r = _solution()
  {"x": x, "r": r}[field][0, 0] = value
  with pytest.raises(ValueError, match=message):
    validate_centralized_solution(x, y, z, r, _data())


def test_validate_centralized_solution_rejects_memory_excess():
  x, y, z, r = _solution()
  data = _data()
  data[None]["memory_capacity"][1] = 0
  with pytest.raises(ValueError, match="residual_capacity"):
    validate_centralized_solution(x, y, z, r, data)
```

- [ ] **Step 3: Run tests and verify RED**

Run:

```bash
.venv/bin/pytest tests/test_centralized_feasibility.py -q
```

Expected: collection fails because `validate_centralized_solution` does not exist.

- [ ] **Step 4: Implement the minimal validator**

Add `validate_centralized_solution(x, y, z, r, data, tolerance=1e-6) -> None` to `utils/centralized.py`. It must:

```python
def validate_centralized_solution(x, y, z, r, data, tolerance=1e-6):
  Nn = data[None]["Nn"][None]
  Nf = data[None]["Nf"][None]
  if x.shape != (Nn, Nf) or y.shape != (Nn, Nn, Nf):
    raise ValueError("solution_shape: incompatible x/y dimensions")
  if z.shape != (Nn, Nf) or r.shape != (Nn, Nf):
    raise ValueError("solution_shape: incompatible z/r dimensions")
  for name, values in (("x", x), ("y", y), ("z", z), ("r", r)):
    bad = np.argwhere(values < -tolerance)
    if len(bad):
      raise ValueError(f"nonnegative_{name}: {tuple(bad[0])}")
  bad = np.argwhere(np.abs(r - np.rint(r)) > tolerance)
  if len(bad):
    raise ValueError(f"integer_replicas: {tuple(bad[0])}")

  neighborhood = np.zeros((Nn, Nn))
  for (n, m), adjacent in data[None]["neighborhood"].items():
    neighborhood[n - 1, m - 1] = adjacent
  bad = np.argwhere((y > tolerance) & (neighborhood[:, :, None] == 0))
  if len(bad):
    raise ValueError(f"offload_only_to_neighbors: {tuple(bad[0])}")

  outgoing = y.sum(axis=1)
  incoming = y.sum(axis=0)
  bad = np.argwhere((outgoing > tolerance) & (incoming > tolerance))
  if len(bad):
    raise ValueError(f"no_ping_pong: {tuple(bad[0])}")

  for n in range(Nn):
    used_memory = 0.0
    for f in range(Nf):
      load = data[None]["incoming_load"][(n + 1, f + 1)]
      if abs(x[n, f] + outgoing[n, f] + z[n, f] - load) > tolerance:
        raise ValueError(f"no_traffic_loss: {(n, f)}")
      processed = x[n, f] + incoming[n, f]
      demand = data[None]["demand"][(n + 1, f + 1)]
      max_utilization = data[None]["max_utilization"][f + 1]
      utilization = demand * processed
      if utilization - r[n, f] * max_utilization > tolerance:
        raise ValueError(f"utilization_equilibrium: {(n, f)}")
      if (r[n, f] - 1) * max_utilization - utilization > tolerance:
        raise ValueError(f"utilization_equilibrium2: {(n, f)}")
      used_memory += r[n, f] * data[None]["memory_requirement"][f + 1]
    if used_memory - data[None]["memory_capacity"][n + 1] > tolerance:
      raise ValueError(f"residual_capacity: {n}")
```

Do not modify `check_feasibility`; existing callers remain compatible.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```bash
.venv/bin/pytest tests/test_centralized_feasibility.py tests/test_utilities.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit the validator**

```bash
git add utils/centralized.py tests/test_centralized_feasibility.py
git commit -m "add centralized feasibility validator"
```

Before committing, run `gitnexus_detect_changes({scope: "staged"})` and verify only the new validator and its tests are reported.

### Task 2: Enforce the contract at the shared solution boundary

**Files:**
- Modify: `run_faasmacro.py:273-317`
- Modify: `tests/test_faas_helpers_extended.py`

- [ ] **Step 1: Re-run impact analysis immediately before editing**

Run `gitnexus_impact` upstream for `combine_solutions`. The known result is CRITICAL: 8 direct callers, 49 impacted symbols, and 8 execution flows. This broad reach is intentional because these callers are the command-line heuristic algorithms that require the common safety boundary. Reconfirm all d=1 callers before proceeding.

- [ ] **Step 2: Write a failing boundary test**

Extend the existing combine-solution tests with a valid two-node data fixture and a non-neighbor flow:

```python
def _two_node_data(neighborhood):
  data = {None: {
    "Nn": {None: 2},
    "Nf": {None: 1},
    "incoming_load": {(1, 1): 1.0, (2, 1): 1.0},
    "neighborhood": {
      (1, 1): 0, (1, 2): 1, (2, 1): 1, (2, 2): 0,
    },
    "demand": {(1, 1): 0.4, (2, 1): 0.4},
    "max_utilization": {1: 0.8},
    "memory_requirement": {1: 1},
    "memory_capacity": {1: 2, 2: 2},
    "alpha": {(1, 1): 1.0, (2, 1): 1.0},
    "beta": {
      (1, 1, 1): 0.9, (1, 2, 1): 0.9,
      (2, 1, 1): 0.9, (2, 2, 1): 0.9,
    },
    "gamma": {(1, 1): 0.8, (2, 1): 0.8},
  }}
  data[None]["neighborhood"].update(neighborhood)
  return data


def test_combine_solutions_rejects_centralized_infeasibility():
  data = _two_node_data(neighborhood={(1, 2): 0, (2, 1): 0})
  y = np.zeros((2, 2, 1))
  y[0, 1, 0] = 1.0
  with pytest.raises(ValueError, match="offload_only_to_neighbors"):
    combine_solutions(
      2, 1, data, {(1, 1): 1.0, (2, 1): 1.0},
      np.zeros((2, 1)), np.ones((2, 1)), np.zeros(2),
      None, y, None, None, None, None,
    )
```

Keep the fixture local to `tests/test_faas_helpers_extended.py` and include all parameters required by `compute_utilization` and the validator.

- [ ] **Step 3: Run the boundary test and verify RED**

Run:

```bash
.venv/bin/pytest tests/test_faas_helpers_extended.py::test_combine_solutions_rejects_centralized_infeasibility -q
```

Expected: FAIL because `combine_solutions` currently returns the invalid solution.

- [ ] **Step 4: Validate before returning the shared solution**

Import `validate_centralized_solution` in `run_faasmacro.py`, assign the current return literal to `solution`, validate its `sp` arrays, then return it:

```python
solution = {
  "sp": {
    "x": sp_x, "y": rmp_y, "z": sp_z, "r": spr_r,
    "xi": sp_xi, "rho": sp_rho, "U": spr_U,
  },
  "rmp": {
    "x": rmp_x, "y": rmp_y, "z": rmp_z, "r": rmp_r,
    "xi": rmp_xi, "rho": rmp_rho, "U": spr_U,
  },
}
validate_centralized_solution(
  solution["sp"]["x"], solution["sp"]["y"],
  solution["sp"]["z"], solution["sp"]["r"], sp_data,
)
return solution
```

Do not add checks to each runner; all eight direct algorithm callers already pass through this boundary.

- [ ] **Step 5: Run shared-helper and runner unit tests**

Run:

```bash
.venv/bin/pytest tests/test_faas_helpers_extended.py tests/test_algorithm_helpers.py tests/test_bestresponse_run.py tests/test_diffusion_wiring.py tests/test_powerd_wiring.py tests/test_hierarchical_runner.py -q
```

Expected: all tests pass, or mocked fixtures must be completed only with data required by the real centralized contract. Do not weaken validation for mocks.

- [ ] **Step 6: Commit the shared safety boundary**

```bash
git add run_faasmacro.py tests/test_faas_helpers_extended.py
git commit -m "enforce centralized feasibility for heuristic solutions"
```

Before committing, run `gitnexus_detect_changes({scope: "staged"})` and verify the eight known algorithm flows are the only production flows affected.

### Task 3: Prevent invalid hierarchical allocations by construction

**Files:**
- Modify: `hierarchical_auction/engine.py:80-315`
- Modify: `tests/test_hierarchical_engine.py`

- [ ] **Step 1: Re-run impact analysis immediately before editing**

Run `gitnexus_impact` upstream for `_generate_level_requests`. The known risk is LOW: one direct caller (`run_higher_levels`) and two affected execution flows. Report this blast radius before editing.

- [ ] **Step 2: Change the non-neighbor expectation into a failing prevention test**

Replace `test_level2_allocates_from_adjacent_structure_seller_node` with:

```python
def _engine(neighborhood):
  return HierarchicalAuctionEngine(
    neighborhood=neighborhood,
    num_functions=1,
    service_quantum=np.ones(1),
    max_depth=2,
    auction_options={
      "epsilon": 0.01, "eta": [0.5, 0.3],
      "latency_weight": 0.0, "fairness_weight": 0.0,
    },
  )


def test_higher_level_never_allocates_to_non_neighbor_node():
  neighborhood = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0],
  ], dtype=float)
  engine = _engine(neighborhood)
  result = engine.run_higher_levels(
    y=np.zeros((3, 3, 1)),
    omega=np.array([[3.0], [0.0], [0.0]]),
    residual_capacity=np.array([[0.0], [0.0], [5.0]]),
    node_prices=np.zeros((3, 1)),
    latency=np.zeros((3, 3)),
    fairness=np.zeros((3, 1)),
  )
  assert result.y[0, 2, 0] == 0.0
  assert result.omega[0, 0] == 3.0
```

Also change the service-quantum test topology to a complete three-node graph so it still tests token conversion using an admissible direct edge.

- [ ] **Step 3: Add a failing ping-pong prevention test**

```python
def test_higher_level_never_makes_receiver_send_same_function():
  neighborhood = np.ones((3, 3), dtype=float) - np.eye(3)
  y = np.zeros((3, 3, 1))
  y[1, 0, 0] = 1.0
  result = _engine(neighborhood).run_higher_levels(
    y=y,
    omega=np.array([[2.0], [0.0], [0.0]]),
    residual_capacity=np.array([[0.0], [0.0], [2.0]]),
    node_prices=np.zeros((3, 1)),
    latency=np.zeros((3, 3)),
    fairness=np.zeros((3, 1)),
  )
  assert result.y[0, 2, 0] == 0.0
```

- [ ] **Step 4: Run both tests and verify RED**

Run:

```bash
.venv/bin/pytest tests/test_hierarchical_engine.py::test_higher_level_never_allocates_to_non_neighbor_node tests/test_hierarchical_engine.py::test_higher_level_never_makes_receiver_send_same_function -q
```

Expected: both tests fail because remote and ping-pong requests are currently generated.

- [ ] **Step 5: Add direct-edge and no-ping-pong request guards**

Pass `current_y` into `_generate_level_requests`. At the start of that method derive provisional roles:

```python
sending = current_y.sum(axis=1) > 1e-10
receiving = current_y.sum(axis=0) > 1e-10
```

Before creating a candidate, require:

```python
if self._neighborhood[buyer_node, seller_node] <= 0:
  continue
if receiving[buyer_node, f] or sending[seller_node, f]:
  continue
```

After registering a request, reserve the roles conservatively for the remainder of that auction level:

```python
sending[buyer_node, f] = True
receiving[seller_node, f] = True
```

This may exclude a later request if an earlier request loses conflict resolution, but it cannot create an invalid solution and requires no rollback protocol.

- [ ] **Step 6: Run hierarchical tests and verify GREEN**

Run:

```bash
.venv/bin/pytest tests/test_hierarchical_engine.py tests/test_hierarchical_flow_mapper.py tests/test_hierarchical_runner.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit hierarchical prevention**

```bash
git add hierarchical_auction/engine.py tests/test_hierarchical_engine.py
git commit -m "keep hierarchical allocations centrally feasible"
```

Before committing, run `gitnexus_detect_changes({scope: "staged"})` and verify only hierarchical request generation and its two known flows are affected.

### Task 4: Prove the centralized upper bound and run regression checks

**Files:**
- Modify: `tests/test_e2e_gurobi_planar.py:89-143`

- [ ] **Step 1: Add the objective-bound assertion**

After running the centralized and hierarchical methods, load both `obj.csv` files and assert the bound with solver tolerance:

```python
centralized_obj = pd.read_csv(Path(centralized_folder, "obj.csv"))["LoadManagementModel"]
hierarchical_obj = pd.read_csv(Path(hierarchical_folder, "obj.csv"))["HierarchicalAuction"]
assert (hierarchical_obj <= centralized_obj + 1e-6).all()
```

- [ ] **Step 2: Run the focused Gurobi end-to-end test**

Run:

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/pytest tests/test_e2e_gurobi_planar.py -q
```

Expected: PASS, including the new upper-bound assertion.

- [ ] **Step 3: Run the complete test suite**

Run:

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/pytest -q
```

Expected: all tests pass with no unexpected warnings or errors.

- [ ] **Step 4: Verify change scope and commit the regression assertion**

Run:

```bash
git add tests/test_e2e_gurobi_planar.py
```

Then run `gitnexus_detect_changes({scope: "staged"})`, confirm the affected scope matches the centralized/distributed/hierarchical planar flow, and commit:

```bash
git commit -m "test centralized objective upper bound"
```

- [ ] **Step 5: Refresh GitNexus after all commits**

Because `.gitnexus/meta.json` reports zero embeddings, run:

```bash
npx gitnexus analyze
```

Then read `gitnexus://repo/DFaaSOptimizer/context` and confirm the index is current.
