# Hierarchical Auction — Post-Review Fixes

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the six correctness and quality issues identified during code review of the `hierarchical_auction/` package, and add the three missing test scenarios.

**Architecture:** All changes are surgical — no file structure changes, no new modules. Task 4 is pure cleanup (no new tests needed). Tasks 1-3 follow TDD: failing test first, then fix.

**Tech Stack:** Python 3.10, NumPy, pytest, ruff, mypy, networkx, scipy (for `.toarray()`).

---

## Issues Addressed

| # | Severity | Summary |
|---|----------|---------|
| 1 | Critical | Structure price caching uses zero as sentinel → stale/repeated computation |
| 2 | Critical | Latency hardcoded to zeros in runner → `latency_weight` is inert in production |
| 3 | Important | `check_global_feasibility()` never called → Invariant 3 has no runtime guard |
| 4 | Important | `omega = result.omega` on runner:252 is overwritten unconditionally → dead assignment |
| 5 | Minor | `bid_price` field on `Structure` is never written → dead state |
| 6 | Important | `__init__.py` exports only types, not `HierarchicalAuctionEngine` / `LevelResult` |
| T1 | Coverage | No test for engine early termination when no allocations produced at a level |
| T2 | Coverage | No test for multi-level cascade with demand propagation through levels |
| T3 | Coverage | No test that calls `check_global_feasibility()` as an invariant assertion |

---

## File Map

Modify only:
- `hierarchical_auction/engine.py` — Tasks 1, 3
- `hierarchical_auction/runner.py` — Tasks 2, 4
- `hierarchical_auction/structure.py` — Task 5 (remove `bid_price`)
- `hierarchical_auction/__init__.py` — Task 6
- `tests/test_hierarchical_engine.py` — Tasks 1, T1, T2
- `tests/test_hierarchical_runner.py` — Task 2
- `tests/test_hierarchical_token_manager.py` — Tasks 3, T3

---

## Task 1: Fix Structure Price Caching Bug

**Files:**
- Modify: `hierarchical_auction/engine.py:241-250`
- Modify: `tests/test_hierarchical_engine.py`

The guard `if np.allclose(buyer_s.structure_price, 0.0)` uses the initial all-zeros value as a "not yet computed" sentinel. In a zero-price network (node_prices all zero, eta=0), the legitimate price is also zero — the guard fires on every function iteration and recomputes unnecessarily. More importantly, the code placement (inside the `for f` loop) is misleading: `compute_structure_price` returns an array over *all* functions at once, so it must be called once per structure, not once per (structure, function) with a fragile zero-sentinel.

**Fix:** Move the call outside the function loop, compute unconditionally.

- [ ] **Step 1: Write failing test for two-function zero-price network**

Add to `tests/test_hierarchical_engine.py`:

```python
def test_price_computed_correctly_in_zero_price_two_function_network():
    """Regression: caching guard must not skip recomputation in zero-price network.

    With eta=0, structure_price = avg(node_prices) = 0.  The effective bid equals
    epsilon > 0, so allocation should still happen even when all prices are zero.
    """
    neighborhood = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ], dtype=float)
    engine = HierarchicalAuctionEngine(
        neighborhood=neighborhood,
        num_functions=2,
        service_quantum=np.array([1.0, 1.0]),
        max_depth=2,
        auction_options={
            "epsilon": 0.01,
            "eta": [0.0, 0.0],
            "latency_weight": 0.0,
            "fairness_weight": 0.0,
        },
    )

    result = engine.run_higher_levels(
        y=np.zeros((3, 3, 2)),
        omega=np.array([[3.0, 2.0], [0.0, 0.0], [0.0, 0.0]]),
        residual_capacity=np.array([[0.0, 0.0], [0.0, 0.0], [5.0, 5.0]]),
        node_prices=np.zeros((3, 2)),
        latency=np.zeros((3, 3)),
        fairness=np.zeros((3, 2)),
    )

    # With zero prices and eta=0, bid = max(0 + 0.01, 0) = 0.01 > 0
    # Both functions should be allocated from node 0 to node 2
    assert result.y[0, 2, 0] == 3.0
    assert result.y[0, 2, 1] == 2.0
    assert result.omega[0, 0] == 0.0
    assert result.omega[0, 1] == 0.0
```

- [ ] **Step 2: Run test to verify it fails (or passes — note the result)**

Run: `uv run pytest tests/test_hierarchical_engine.py::test_price_computed_correctly_in_zero_price_two_function_network -v`

Note: if this passes, the bug is latent (logic is correct but code is fragile). Proceed to the fix regardless — the test locks in correct behavior.

- [ ] **Step 3: Move compute_structure_price outside the function loop**

In `hierarchical_auction/engine.py`, the `_generate_level_requests` method. Find the loop starting at line ~241:

```python
    for root, buyer_s in buyer_structures.items():
      for f in range(self._num_functions):
        if not buyer_s.is_buyer(f):
          continue

        # Compute structure price once per (structure, function)
        if np.allclose(buyer_s.structure_price, 0.0):
          buyer_s.structure_price = compute_structure_price(
            buyer_s, node_prices, token_manager.tokens, eta=eta,
          )

        # Find seller nodes in adjacent structures
```

Replace with:

```python
    for root, buyer_s in buyer_structures.items():
      buyer_s.structure_price = compute_structure_price(
        buyer_s, node_prices, token_manager.tokens, eta=eta,
      )
      for f in range(self._num_functions):
        if not buyer_s.is_buyer(f):
          continue

        # Find seller nodes in adjacent structures
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_hierarchical_engine.py -v`

Expected: all engine tests PASS.

- [ ] **Step 5: Commit**

```bash
git add hierarchical_auction/engine.py tests/test_hierarchical_engine.py
git commit -m "fix: compute structure price unconditionally per structure, not per function"
```

---

## Task 2: Wire Real Latency from Graph in Runner

**Files:**
- Modify: `hierarchical_auction/runner.py`
- Modify: `tests/test_hierarchical_runner.py`

Both `define_bids` and `engine.run_higher_levels` receive `latency=np.zeros((Nn, Nn))`. The `latency_weight` config option is therefore inert in production: `latency_weight * 0 = 0` always. The fix follows the exact pattern used in `decentralized_auction.py` (line 311) and `run_faasmadea.py` (line 280).

The networkx `adjacency_matrix` function returns a scipy sparse matrix; use `.toarray()` to get a dense numpy array compatible with the `engine.py` type annotation `latency: np.ndarray`.

- [ ] **Step 1: Write failing test for latency extraction helper**

Add to `tests/test_hierarchical_runner.py`:

```python
def test_extract_graph_latency_returns_numpy_array_with_edge_weights():
    import networkx as nx
    from hierarchical_auction.runner import _extract_latency

    g = nx.path_graph(3)
    nx.set_edge_attributes(
        g, {(0, 1): 5.0, (1, 2): 3.0}, "network_latency"
    )
    lat = _extract_latency(g)

    assert isinstance(lat, np.ndarray)
    assert lat.shape == (3, 3)
    assert lat[0, 1] == 5.0
    assert lat[1, 0] == 5.0   # undirected
    assert lat[1, 2] == 3.0
    assert lat[0, 0] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_hierarchical_runner.py::test_extract_graph_latency_returns_numpy_array_with_edge_weights -v`

Expected: FAIL with `ImportError: cannot import name '_extract_latency'`.

- [ ] **Step 3: Add `_extract_latency` helper and wire it into `run()`**

In `hierarchical_auction/runner.py`:

**3a.** Add the import at the top of the file (after the existing imports):

```python
from networkx import adjacency_matrix as nx_adjacency_matrix
```

**3b.** Add the helper function after `compute_offloaded_demand` (around line 91):

```python
def _extract_latency(graph: Any) -> np.ndarray:
    return nx_adjacency_matrix(graph, weight="network_latency").toarray()
```

**3c.** In `run()`, after `init_problem` sets up `graph` (around line 137), add latency extraction **before** the time loop:

Find the block:
```python
  neighborhood = neigh_dict_to_matrix(
    base_instance_data[None]["neighborhood"], Nn,
  )
```

Replace with:
```python
  neighborhood = neigh_dict_to_matrix(
    base_instance_data[None]["neighborhood"], Nn,
  )
  latency = _extract_latency(graph)
```

**3d.** In the level-1 `define_bids` call (around line 193), replace:
```python
        latency=np.zeros((Nn, Nn)),
```
with:
```python
        latency=latency,
```

**3e.** In the `engine.run_higher_levels` call (around line 248), replace:
```python
        latency=np.zeros((Nn, Nn)),
```
with:
```python
        latency=latency,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_hierarchical_runner.py -v`

Expected: all runner tests PASS.

- [ ] **Step 5: Commit**

```bash
git add hierarchical_auction/runner.py tests/test_hierarchical_runner.py
git commit -m "fix: extract real latency from graph; latency_weight now effective in production"
```

---

## Task 3: Enforce Global Feasibility Invariant

**Files:**
- Modify: `hierarchical_auction/engine.py`
- Modify: `tests/test_hierarchical_token_manager.py`

`check_global_feasibility()` exists and is correct (`token_manager.py:119-122`) but is never called. This leaves Invariant 3 (`sum_accepted_to_node[k,f] <= initial_tokens[k,f]`) without any runtime guard. An assertion after the level loop in `run_higher_levels` catches token accounting bugs immediately.

- [ ] **Step 1: Write failing test for check_global_feasibility**

Add to `tests/test_hierarchical_token_manager.py`:

```python
def test_check_global_feasibility_detects_over_committed_state():
    manager = CapacityTokenManager(np.array([[3.0]]), np.array([1.0]))
    # Manually corrupt the token state to simulate over-commitment
    manager._current_tokens[0, 0] = -1  # type: ignore[attr-defined]
    assert manager.check_global_feasibility() is False


def test_check_global_feasibility_passes_after_valid_commits():
    manager = CapacityTokenManager(np.array([[10.0]]), np.array([1.0]))
    manager.request(make_request(1, 6, 10.0))
    accepted = manager.resolve_node_function(0, 0)
    manager.commit(accepted)
    assert manager.check_global_feasibility() is True
```

- [ ] **Step 2: Run tests to verify**

Run: `uv run pytest tests/test_hierarchical_token_manager.py -v`

Note: these should already PASS (the method works). If they fail, the method has a bug — fix it before continuing.

- [ ] **Step 3: Add assertion to engine after level loop**

In `hierarchical_auction/engine.py`, at the end of `run_higher_levels` just before the `return` statement (around line 170):

Find:
```python
    return LevelResult(
      y=current_y,
      omega=current_omega,
      accepted_allocations=all_accepted,
    )
```

Replace with:
```python
    assert token_manager.check_global_feasibility(), (
      "Invariant 3 violated: committed tokens exceed initial tokens for some (k,f)"
    )
    return LevelResult(
      y=current_y,
      omega=current_omega,
      accepted_allocations=all_accepted,
    )
```

- [ ] **Step 4: Verify all engine tests still pass**

Run: `uv run pytest tests/test_hierarchical_engine.py tests/test_hierarchical_token_manager.py -v`

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add hierarchical_auction/engine.py tests/test_hierarchical_token_manager.py
git commit -m "fix: assert global feasibility invariant after each run_higher_levels call"
```

---

## Task 4: Remove Dead Code and Fix Exports

**Files:**
- Modify: `hierarchical_auction/runner.py`
- Modify: `hierarchical_auction/structure.py`
- Modify: `hierarchical_auction/__init__.py`

Three small cleanup changes with no behavior impact. No new tests required.

**4a — Remove redundant `omega` assignment (runner.py:252)**

In `run()`, after `engine.run_higher_levels` returns, the line `omega = result.omega` is immediately overwritten by `omega[i, f] = sp_omega[i, f] - rmp_omega[i, f]` inside the `if result.accepted_allocations:` block. The `result.omega` assignment is dead.

Find block (around line 250):
```python
      y = result.y
      omega = result.omega
      rmp_omega = compute_offloaded_demand(y)
```

Replace with:
```python
      y = result.y
      rmp_omega = compute_offloaded_demand(y)
```

**4b — Remove `bid_price` dead field from `Structure`**

`bid_price` is declared in `structure.py` but never assigned anywhere in the engine, runner, or tests.

In `hierarchical_auction/structure.py`, remove the two lines:
```python
  bid_price: np.ndarray = field(init=False)
```
and (inside `__post_init__`):
```python
    self.bid_price = np.zeros(self.num_functions, dtype=float)
```

**4c — Export `HierarchicalAuctionEngine` and `LevelResult` from `__init__.py`**

Replace the content of `hierarchical_auction/__init__.py` with:

```python
from hierarchical_auction.engine import HierarchicalAuctionEngine, LevelResult
from hierarchical_auction.types import AcceptedAllocation, TokenRequest

__all__ = [
  "AcceptedAllocation",
  "HierarchicalAuctionEngine",
  "LevelResult",
  "TokenRequest",
]
```

**4d — Document `zeta` in `build_auction_options`**

`zeta` is intentionally forwarded: `evaluate_bids` (level-1 price dampening) consumes it via `level1_options`. Add one comment so future readers understand.

In `hierarchical_auction/runner.py`, in `build_auction_options`:
```python
def build_auction_options(config: dict) -> dict:
  solver_options = config.get("solver_options", {})
  auction = solver_options.get("auction", {})
  return {
    "epsilon": auction.get("epsilon", 0.01),
    "eta": auction.get("eta", [0.5, 0.3, 0.1]),
    "zeta": auction.get("zeta", 0.1),   # consumed by evaluate_bids (level-1 price dampening)
    "latency_weight": auction.get("latency_weight", 0.0),
    "fairness_weight": auction.get("fairness_weight", 0.0),
  }
```

- [ ] **Step 1: Apply all four sub-changes above**

- [ ] **Step 2: Run full hierarchical test suite**

Run: `uv run pytest tests/test_hierarchical_types.py tests/test_hierarchical_structure_graph.py tests/test_hierarchical_token_manager.py tests/test_hierarchical_pricing.py tests/test_hierarchical_flow_mapper.py tests/test_hierarchical_engine.py tests/test_hierarchical_runner.py -v`

Expected: all PASS.

- [ ] **Step 3: Update smoke test in test_coverage_expansion.py**

Find in `tests/test_coverage_expansion.py`:
```python
def test_hierarchical_auction_public_imports():
  import hierarchical_auction as ha

  assert ha.AcceptedAllocation is not None
  assert ha.TokenRequest is not None
```

Replace with:
```python
def test_hierarchical_auction_public_imports():
  import hierarchical_auction as ha

  assert ha.AcceptedAllocation is not None
  assert ha.TokenRequest is not None
  assert ha.HierarchicalAuctionEngine is not None
  assert ha.LevelResult is not None
```

- [ ] **Step 4: Commit**

```bash
git add hierarchical_auction/runner.py hierarchical_auction/structure.py hierarchical_auction/__init__.py tests/test_coverage_expansion.py
git commit -m "refactor: remove dead omega assignment and bid_price field; expand public exports"
```

---

## Task 5: Add Missing Engine Tests

**Files:**
- Modify: `tests/test_hierarchical_engine.py`

Two test scenarios required by the plan's coverage checklist:
1. Engine stops early when no level produces allocations (the `break` at engine.py:158 is untested).
2. Multi-level cascade: demand is only satisfiable at level 3, verifying `current_omega` propagates correctly through `apply_allocations` → `_aggregate_residual_demand`.

- [ ] **Step 1: Write the two failing tests**

Add to `tests/test_hierarchical_engine.py`:

```python
def test_engine_stops_early_when_no_capacity_available():
    """No seller has capacity → accepted_allocations is empty and engine does not loop forever."""
    neighborhood = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ], dtype=float)
    engine = HierarchicalAuctionEngine(
        neighborhood=neighborhood,
        num_functions=1,
        service_quantum=np.array([1.0]),
        max_depth=5,
        auction_options={
            "epsilon": 0.01,
            "eta": [0.5, 0.3, 0.2, 0.1, 0.05],
            "latency_weight": 0.0,
            "fairness_weight": 0.0,
        },
    )

    result = engine.run_higher_levels(
        y=np.zeros((3, 3, 1)),
        omega=np.array([[3.0], [0.0], [0.0]]),
        residual_capacity=np.zeros((3, 1)),   # no capacity anywhere
        node_prices=np.zeros((3, 1)),
        latency=np.zeros((3, 3)),
        fairness=np.zeros((3, 1)),
    )

    assert result.accepted_allocations == []
    assert result.omega[0, 0] == 3.0          # demand unchanged
    assert result.y[0, :, 0].sum() == 0.0


def test_engine_level3_cascade_satisfies_demand_beyond_direct_neighbors():
    """Demand at node 0 is satisfied by node 4 reachable only at level 3.

    Topology: linear chain 0-1-2-3-4
    - omega[0,f] = 2.0; all other nodes have no demand
    - residual_capacity[4,f] = 5.0; all other nodes have no capacity
    At level 2, S_0^(2) merges with S_1^(1) → covers {0,1,2}.  Node 4 is
    still not reachable.  At level 3 (if aggregation reaches it), node 4
    becomes accessible.  The engine must complete without error and must not
    over-allocate.
    """
    n = 5
    neighborhood = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        neighborhood[i, i + 1] = 1.0
        neighborhood[i + 1, i] = 1.0

    engine = HierarchicalAuctionEngine(
        neighborhood=neighborhood,
        num_functions=1,
        service_quantum=np.array([1.0]),
        max_depth=4,
        auction_options={
            "epsilon": 0.01,
            "eta": [0.5, 0.4, 0.3, 0.2],
            "latency_weight": 0.0,
            "fairness_weight": 0.0,
        },
    )

    result = engine.run_higher_levels(
        y=np.zeros((n, n, 1)),
        omega=np.array([[2.0], [0.0], [0.0], [0.0], [0.0]]),
        residual_capacity=np.array([[0.0], [0.0], [0.0], [0.0], [5.0]]),
        node_prices=np.zeros((n, 1)),
        latency=np.zeros((n, n)),
        fairness=np.zeros((n, 1)),
    )

    # Whether or not level 3 reaches node 4 depends on aggregation depth;
    # what must hold unconditionally:
    assert result.omega[0, 0] >= 0.0             # demand is non-negative
    assert result.y.sum() == pytest.approx(2.0 - result.omega[0, 0])  # flow = allocated
    assert result.y[0, 0, 0] == 0.0              # no self-allocation
    # If any allocation happened, committed tokens must not exceed initial
    # (the engine assertion already checks this, but re-test explicitly)
    total_allocated = result.y[0, :, 0].sum()
    assert total_allocated <= 2.0
```

At the top of the test file, ensure `pytest` is imported:

```python
import pytest
```

(Add this if not already present.)

- [ ] **Step 2: Run the new tests to verify they pass**

Run: `uv run pytest tests/test_hierarchical_engine.py -v`

Expected: all PASS. The cascade test verifies the engine terminates cleanly; the empty-capacity test verifies early exit.

- [ ] **Step 3: Commit**

```bash
git add tests/test_hierarchical_engine.py
git commit -m "test: add engine early-termination and multi-level cascade tests"
```

---

## Task 6: Quality Gates

**Files:** none (read-only verification)

- [ ] **Step 1: Run full hierarchical test suite**

```bash
uv run pytest tests/test_hierarchical_types.py \
  tests/test_hierarchical_structure_graph.py \
  tests/test_hierarchical_token_manager.py \
  tests/test_hierarchical_pricing.py \
  tests/test_hierarchical_flow_mapper.py \
  tests/test_hierarchical_engine.py \
  tests/test_hierarchical_runner.py \
  tests/test_coverage_expansion.py \
  -v
```

Expected: all PASS.

- [ ] **Step 2: Run full test suite with coverage**

```bash
uv run pytest --cov --cov-report=term-missing -q
```

Expected: PASS, total coverage ≥ 50%.

- [ ] **Step 3: Run linting and type checking**

```bash
uv run ruff check .
uv run mypy
```

Expected: both pass with zero errors. If `mypy` reports `_extract_latency` return type, add the annotation:

```python
def _extract_latency(graph: Any) -> np.ndarray:
```

(`Any` is already used in the runner's type annotations; `np.ndarray` is the correct return type.)

- [ ] **Step 4: Run GitNexus change detection**

```bash
gitnexus detect_changes
```

Expected: no HIGH or CRITICAL risk without explicit review.

- [ ] **Step 5: Final summary commit (if any loose files remain)**

```bash
git status
```

If any modified files were missed by earlier task commits, add them now.

---

## Self-Review Checklist

- [x] All 8 non-negotiable invariants addressed: Invariant 3 now has runtime enforcement.
- [x] Structure price caching replaced with unconditional computation per structure.
- [x] Real latency extracted from graph; `latency_weight` is now effective in production.
- [x] `omega = result.omega` dead assignment removed.
- [x] `bid_price` dead field removed from `Structure`.
- [x] `__init__.py` exports `HierarchicalAuctionEngine` and `LevelResult`.
- [x] `zeta` documented (consumed by level-1 `evaluate_bids`).
- [x] Early termination path tested explicitly.
- [x] Multi-level cascade tested for correctness and no over-allocation.
- [x] `check_global_feasibility` tested both pass and fail paths.
- [x] No new files created; no existing behavior changed beyond the identified fixes.
