# Hierarchical Auction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the hierarchical multi-structure auction from `Decentralized_FaaS_coordination.pdf` Sections 4.5-4.7 so accepted higher-level reservations become concrete node-to-node FaaS offloading flows and remain compatible with the existing experiment pipeline.

**Architecture:** Build a new `hierarchical_auction/` package around explicit allocation records. Structures coordinate demand, but capacity remains node-owned; every accepted token must map to a concrete `(buyer_node, seller_node, function, quantity)` allocation and update the normal `y[i,j,f]` matrix. Keep the first implementation runnable through a standalone runner, then integrate with `run.py` only after output files match the existing `faas-madea` conventions.

**Tech Stack:** Python 3.10, NumPy, Pandas, existing Pyomo runners and utilities, pytest, ruff, mypy.

---

## Non-Negotiable Invariants

These invariants come directly from the PDF and must be tested before the runner is considered correct:

1. Structures never own capacity. Node `k` owns tokens `T[k,f] = floor(C[k,f] / delta[f])`.
2. Multiple structures may request the same `(k,f)` capacity. Conflict resolution happens after all requests are collected.
3. Accepted reservations must satisfy `sum_accepted_to_node[k,f] <= initial_tokens[k,f]` across all levels.
4. Every accepted reservation must map to one or more concrete flows `y[buyer_node, seller_node, f]`.
5. Residual demand after a level is computed from concrete allocations, not from abstract structure counters.
6. Higher levels query adjacent structures and candidate seller nodes inside those adjacent structures.
7. Structure price is only an indicative congestion signal; it must not imply ownership of aggregate tokens.
8. The runner must emit the same artifact shape expected by existing postprocessing: solution CSVs, `obj.csv`, `termination_condition.csv`, and checkpoints when configured.

---

## File Structure

Create:

- `hierarchical_auction/__init__.py`: public exports.
- `hierarchical_auction/types.py`: dataclasses for token requests and accepted allocations.
- `hierarchical_auction/structure.py`: `Structure` dataclass and demand helpers.
- `hierarchical_auction/structure_graph.py`: level-1 structures, adjacency, recursive aggregation.
- `hierarchical_auction/token_manager.py`: request collection, conflict resolution, cumulative token commits.
- `hierarchical_auction/pricing.py`: structure prices, structure bids, effective seller-side bid values.
- `hierarchical_auction/flow_mapper.py`: convert accepted tokens into concrete `y` updates.
- `hierarchical_auction/engine.py`: level-by-level orchestration around existing one-hop auction state.
- `hierarchical_auction/runner.py`: standalone `run(config, ...)` entry point.

Create tests:

- `tests/test_hierarchical_types.py`
- `tests/test_hierarchical_structure_graph.py`
- `tests/test_hierarchical_token_manager.py`
- `tests/test_hierarchical_pricing.py`
- `tests/test_hierarchical_flow_mapper.py`
- `tests/test_hierarchical_engine.py`
- `tests/test_hierarchical_runner.py`

Modify only after the standalone runner is verified:

- `run.py`: add `hierarchical` to method choices and experiment bookkeeping.
- `tests/test_coverage_expansion.py`: add import smoke coverage for the new package.

---

## Task 1: Allocation Types

**Files:**
- Create: `hierarchical_auction/types.py`
- Create: `hierarchical_auction/__init__.py`
- Create: `tests/test_hierarchical_types.py`

- [ ] **Step 1: Write failing tests for request and allocation records**

```python
import pytest

from hierarchical_auction.types import AcceptedAllocation, TokenRequest


def test_token_request_rejects_negative_quantity():
  with pytest.raises(ValueError, match="tokens must be positive"):
    TokenRequest(
      level=2,
      buyer_structure=0,
      buyer_node=0,
      seller_node=1,
      function=0,
      tokens=0,
      bid_value=1.0,
      quantity=0.0,
    )


def test_accepted_allocation_records_concrete_flow():
  allocation = AcceptedAllocation(
    level=2,
    buyer_structure=0,
    buyer_node=0,
    seller_node=3,
    function=1,
    tokens=4,
    quantity=4.0,
    bid_value=2.5,
  )
  assert allocation.buyer_node == 0
  assert allocation.seller_node == 3
  assert allocation.function == 1
  assert allocation.quantity == 4.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_hierarchical_types.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'hierarchical_auction'`.

- [ ] **Step 3: Implement minimal dataclasses**

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenRequest:
  level: int
  buyer_structure: int
  buyer_node: int
  seller_node: int
  function: int
  tokens: int
  bid_value: float
  quantity: float

  def __post_init__(self) -> None:
    if self.tokens <= 0:
      raise ValueError("tokens must be positive")
    if self.quantity <= 0:
      raise ValueError("quantity must be positive")


@dataclass(frozen=True)
class AcceptedAllocation:
  level: int
  buyer_structure: int
  buyer_node: int
  seller_node: int
  function: int
  tokens: int
  quantity: float
  bid_value: float
```

`hierarchical_auction/__init__.py`:

```python
from hierarchical_auction.types import AcceptedAllocation, TokenRequest

__all__ = [
  "AcceptedAllocation",
  "TokenRequest",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_hierarchical_types.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hierarchical_auction/__init__.py hierarchical_auction/types.py tests/test_hierarchical_types.py
git commit -m "feat: add hierarchical auction allocation types"
```

---

## Task 2: Structures and Structure Graph

**Files:**
- Create: `hierarchical_auction/structure.py`
- Create: `hierarchical_auction/structure_graph.py`
- Create: `tests/test_hierarchical_structure_graph.py`

- [ ] **Step 1: Write failing graph tests**

```python
import numpy as np

from hierarchical_auction.structure_graph import StructureGraph


def chain_graph(size: int) -> np.ndarray:
  graph = np.zeros((size, size), dtype=float)
  for i in range(size - 1):
    graph[i, i + 1] = 1.0
    graph[i + 1, i] = 1.0
  return graph


def test_level1_structure_is_one_hop_neighborhood():
  sg = StructureGraph(chain_graph(4))
  structures = sg.build_level1(num_functions=2)
  assert structures[0].member_nodes == {0, 1}
  assert structures[1].member_nodes == {0, 1, 2}
  assert structures[2].member_nodes == {1, 2, 3}
  assert structures[3].member_nodes == {2, 3}


def test_adjacency_uses_physical_edge_between_members():
  sg = StructureGraph(chain_graph(4))
  structures = sg.build_level1(num_functions=1)
  assert sg.are_adjacent(structures[0], structures[2]) is True
  assert sg.are_adjacent(structures[0], structures[3]) is False


def test_aggregation_rebuilds_adjacency_for_new_level():
  sg = StructureGraph(chain_graph(5))
  level1 = sg.build_level1(num_functions=1)
  level2 = sg.aggregate_to_next_level(level1, num_functions=1)
  assert all(s.level == 2 for s in level2.values())
  assert level1[0].member_nodes.issubset(level2[0].member_nodes)
  assert all(root not in s.adjacent_structures for root, s in level2.items())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_hierarchical_structure_graph.py -q`

Expected: FAIL with import errors.

- [ ] **Step 3: Implement `Structure` and `StructureGraph`**

Implementation rules:

- `Structure.residual_demand`, `structure_price`, `indicative_tokens`, and `bid_price` are NumPy arrays of shape `(num_functions,)`.
- `build_level1()` must call `build_adjacency()`.
- `aggregate_to_next_level()` must merge the root structure with its adjacent previous-level structures, then call `build_adjacency()` on the new level. Do not inherit stale adjacency sets.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_hierarchical_structure_graph.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hierarchical_auction/structure.py hierarchical_auction/structure_graph.py tests/test_hierarchical_structure_graph.py
git commit -m "feat: add hierarchical structure graph"
```

---

## Task 3: Token Manager With Deferred Conflict Resolution

**Files:**
- Create: `hierarchical_auction/token_manager.py`
- Create: `tests/test_hierarchical_token_manager.py`

- [ ] **Step 1: Write failing token tests**

```python
import numpy as np

from hierarchical_auction.token_manager import CapacityTokenManager
from hierarchical_auction.types import TokenRequest


def make_request(root: int, tokens: int, bid: float) -> TokenRequest:
  return TokenRequest(
    level=2,
    buyer_structure=root,
    buyer_node=root,
    seller_node=0,
    function=0,
    tokens=tokens,
    bid_value=bid,
    quantity=float(tokens),
  )


def test_collects_oversubscribed_requests_before_resolution():
  manager = CapacityTokenManager(np.array([[10.0]]), np.array([1.0]))
  manager.request(make_request(1, 6, 10.0))
  manager.request(make_request(2, 8, 5.0))
  assert len(manager.pending_requests(0, 0)) == 2
  assert manager.available_tokens(0, 0) == 10


def test_resolve_conflicts_does_not_exceed_node_tokens():
  manager = CapacityTokenManager(np.array([[10.0]]), np.array([1.0]))
  manager.request(make_request(1, 6, 10.0))
  manager.request(make_request(2, 8, 5.0))
  accepted = manager.resolve_node_function(0, 0)
  assert sum(a.tokens for a in accepted) == 10
  assert accepted[0].buyer_structure == 1
  assert accepted[0].tokens == 6
  assert accepted[1].tokens == 4


def test_commit_is_cumulative_across_rounds():
  manager = CapacityTokenManager(np.array([[10.0]]), np.array([1.0]))
  manager.request(make_request(1, 4, 1.0))
  accepted = manager.resolve_node_function(0, 0)
  manager.commit(accepted)
  assert manager.available_tokens(0, 0) == 6

  manager.request(make_request(2, 5, 1.0))
  accepted = manager.resolve_node_function(0, 0)
  manager.commit(accepted)
  assert manager.available_tokens(0, 0) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_hierarchical_token_manager.py -q`

Expected: FAIL with import errors.

- [ ] **Step 3: Implement token manager**

Implementation rules:

- `request()` appends to pending requests and does not reduce availability.
- `resolve_node_function(k, f)` sorts by descending effective `bid_value` and accepts full or partial token quantities until capacity is exhausted.
- `commit(allocations)` subtracts from current tokens, not from initial tokens.
- `check_global_feasibility()` validates committed allocations against initial tokens.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_hierarchical_token_manager.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hierarchical_auction/token_manager.py tests/test_hierarchical_token_manager.py
git commit -m "feat: add cumulative hierarchical token manager"
```

---

## Task 4: Pricing and Effective Bid Values

**Files:**
- Create: `hierarchical_auction/pricing.py`
- Create: `tests/test_hierarchical_pricing.py`

- [ ] **Step 1: Write failing pricing tests**

```python
import numpy as np
import pytest

from hierarchical_auction.pricing import (
  compute_effective_bid,
  compute_structure_price,
  generate_structure_bid,
)
from hierarchical_auction.structure import Structure


def test_structure_price_matches_pdf_equation_27():
  structure = Structure(level=2, root_node=0, member_nodes={0, 1}, num_functions=1)
  structure.residual_demand[:] = [6.0]
  node_prices = np.array([[0.2], [0.4]])
  node_tokens = np.array([[4], [2]])
  price = compute_structure_price(structure, node_prices, node_tokens, eta=0.5)
  assert price[0] == pytest.approx(0.3 + 0.5 * (6.0 / (6.0 + 6.0 + 1e-6)))


def test_bid_to_node_is_at_least_node_price():
  structure_price = 0.1
  node_price = 0.5
  assert generate_structure_bid(structure_price, node_price, epsilon=0.01) == 0.5


def test_effective_bid_penalizes_latency_and_fairness():
  effective = compute_effective_bid(
    bid=2.0,
    node_price=0.5,
    latency=3.0,
    fairness=4.0,
    latency_weight=0.1,
    fairness_weight=0.2,
  )
  assert effective == pytest.approx(2.0 - 0.5 - 0.3 - 0.8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_hierarchical_pricing.py -q`

Expected: FAIL with import errors.

- [ ] **Step 3: Implement pricing helpers**

Implementation rules:

- `compute_structure_price()` implements PDF Eq.27.
- `generate_structure_bid(structure_price, node_price, epsilon)` implements Eq.28 plus Eq.22 with `max(structure_price + epsilon, node_price)`.
- `compute_effective_bid()` implements PDF Eq.19.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_hierarchical_pricing.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hierarchical_auction/pricing.py tests/test_hierarchical_pricing.py
git commit -m "feat: add hierarchical structure pricing"
```

---

## Task 5: Flow Mapping

**Files:**
- Create: `hierarchical_auction/flow_mapper.py`
- Create: `tests/test_hierarchical_flow_mapper.py`

- [ ] **Step 1: Write failing flow mapping tests**

```python
import numpy as np

from hierarchical_auction.flow_mapper import apply_allocations
from hierarchical_auction.types import AcceptedAllocation


def test_apply_allocations_updates_y_and_residual_demand():
  y = np.zeros((3, 3, 1))
  omega = np.array([[5.0], [0.0], [0.0]])
  allocations = [
    AcceptedAllocation(
      level=2,
      buyer_structure=0,
      buyer_node=0,
      seller_node=2,
      function=0,
      tokens=3,
      quantity=3.0,
      bid_value=1.0,
    )
  ]
  new_y, new_omega = apply_allocations(y, omega, allocations)
  assert new_y[0, 2, 0] == 3.0
  assert new_omega[0, 0] == 2.0


def test_apply_allocations_clips_to_remaining_demand():
  y = np.zeros((2, 2, 1))
  omega = np.array([[2.0], [0.0]])
  allocations = [
    AcceptedAllocation(2, 0, 0, 1, 0, 5, 5.0, 1.0)
  ]
  new_y, new_omega = apply_allocations(y, omega, allocations)
  assert new_y[0, 1, 0] == 2.0
  assert new_omega[0, 0] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_hierarchical_flow_mapper.py -q`

Expected: FAIL with import errors.

- [ ] **Step 3: Implement `apply_allocations()`**

Implementation rules:

- Copy input arrays.
- For each allocation, add `min(quantity, residual demand)` to `y[buyer_node, seller_node, function]`.
- Subtract the same quantity from `omega[buyer_node, function]`.
- Never create negative residual demand.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_hierarchical_flow_mapper.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hierarchical_auction/flow_mapper.py tests/test_hierarchical_flow_mapper.py
git commit -m "feat: map hierarchical tokens to flows"
```

---

## Task 6: Hierarchical Engine

**Files:**
- Create: `hierarchical_auction/engine.py`
- Create: `tests/test_hierarchical_engine.py`

- [ ] **Step 1: Write failing engine tests**

```python
import numpy as np

from hierarchical_auction.engine import HierarchicalAuctionEngine


def test_level2_allocates_from_adjacent_structure_seller_node():
  neighborhood = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0],
  ], dtype=float)
  engine = HierarchicalAuctionEngine(
    neighborhood=neighborhood,
    num_functions=1,
    service_quantum=np.array([1.0]),
    max_depth=2,
    auction_options={
      "epsilon": 0.01,
      "eta": [0.5, 0.3],
      "latency_weight": 0.0,
      "fairness_weight": 0.0,
    },
  )
  y = np.zeros((3, 3, 1))
  omega = np.array([[3.0], [0.0], [0.0]])
  residual_capacity = np.array([[0.0], [0.0], [5.0]])
  node_prices = np.zeros((3, 1))
  latency = np.zeros((3, 3))
  fairness = np.zeros((3, 1))

  result = engine.run_higher_levels(
    y=y,
    omega=omega,
    residual_capacity=residual_capacity,
    node_prices=node_prices,
    latency=latency,
    fairness=fairness,
  )

  assert result.y[0, 2, 0] == 3.0
  assert result.omega[0, 0] == 0.0
  assert result.accepted_allocations
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_hierarchical_engine.py -q`

Expected: FAIL with import errors.

- [ ] **Step 3: Implement engine around completed helpers**

Implementation rules:

- Build level 1 structures from the physical graph.
- At each higher level, aggregate structures, compute residual demand, compute indicative tokens, and generate token requests only for seller nodes in adjacent structures.
- For each buyer structure, distribute demand to concrete buyer nodes that still have residual `omega`.
- For each candidate seller node, compute bid with Eq.28/Eq.22 and effective bid with Eq.19.
- Collect all requests, resolve per `(seller_node, function)`, commit accepted allocations, then call `apply_allocations()`.
- Stop when no structure has residual demand, no accepted allocations are produced at a level, or `max_depth` is reached.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_hierarchical_engine.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hierarchical_auction/engine.py tests/test_hierarchical_engine.py
git commit -m "feat: add hierarchical auction engine"
```

---

## Task 7: Standalone Runner

**Files:**
- Create: `hierarchical_auction/runner.py`
- Create: `tests/test_hierarchical_runner.py`

- [ ] **Step 1: Write failing runner tests**

```python
from hierarchical_auction.runner import build_auction_options


def test_build_auction_options_accepts_scalar_or_list_eta():
  config = {
    "solver_options": {
      "auction": {
        "epsilon": 0.01,
        "eta": [0.5, 0.3],
        "zeta": 0.1,
        "latency_weight": 0.0,
        "fairness_weight": 0.0,
      }
    }
  }
  opts = build_auction_options(config)
  assert opts["eta"] == [0.5, 0.3]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_hierarchical_runner.py -q`

Expected: FAIL with import errors.

- [ ] **Step 3: Implement standalone runner**

Implementation rules:

- Reuse `decentralized_auction.run()` structure for initialization, logging, time loop, checkpointing, and saving.
- Reuse existing one-hop auction functions for level 1.
- Invoke `HierarchicalAuctionEngine.run_higher_levels()` after level-1 allocation and before convergence checks.
- Do not pass `None` into `combine_solutions()` where arrays are required; create zero arrays of the correct shape when a component is absent.
- Save outputs with the `LSPc` prefix so existing postprocessing can parse them.

- [ ] **Step 4: Run runner helper tests**

Run: `uv run pytest tests/test_hierarchical_runner.py -q`

Expected: PASS.

- [ ] **Step 5: Run a minimal smoke command**

Use a config where list-valued parameters match dimensions. For `Nn=3`, `memory_capacity.values` must contain three values.

Run: `uv run python -m hierarchical_auction.runner -c config_files/manual_config.json -j 0 --disable_plotting`

Expected: command starts, creates a solution folder, and does not fail during initialization.

- [ ] **Step 6: Commit**

```bash
git add hierarchical_auction/runner.py tests/test_hierarchical_runner.py
git commit -m "feat: add hierarchical auction runner"
```

---

## Task 8: `run.py` Integration

**Files:**
- Modify: `run.py`
- Create or modify: `tests/test_hierarchical_runner.py`

- [ ] **Step 1: Write failing integration test for method registration**

```python
from run import parse_arguments


def test_run_py_accepts_hierarchical_method(monkeypatch):
  monkeypatch.setattr(
    "sys.argv",
    ["run.py", "-c", "config_files/config.json", "--methods", "hierarchical"],
  )
  args = parse_arguments()
  assert args.methods == ["hierarchical"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_hierarchical_runner.py::test_run_py_accepts_hierarchical_method -q`

Expected: FAIL because `hierarchical` is not in argparse choices.

- [ ] **Step 3: Modify `run.py`**

Implementation rules:

- Import `run` from `hierarchical_auction.runner` as `run_hierarchical`.
- Add `"hierarchical"` to `--methods` choices.
- Add `"hierarchical": []` to `solution_folders`.
- Add run flag detection and execution analogous to `faas-madea`.
- Update result labels in postprocessing dispatch only after verifying output files match existing expectations.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_hierarchical_runner.py::test_run_py_accepts_hierarchical_method -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add run.py tests/test_hierarchical_runner.py
git commit -m "feat: register hierarchical auction method"
```

---

## Task 9: Verification and Quality Gates

**Files:**
- Modify: `tests/test_coverage_expansion.py`

- [ ] **Step 1: Add import smoke test**

```python
def test_hierarchical_auction_public_imports():
  import hierarchical_auction as ha

  assert ha.AcceptedAllocation is not None
  assert ha.TokenRequest is not None
```

- [ ] **Step 2: Run focused tests**

Run:

```bash
uv run pytest tests/test_hierarchical_types.py tests/test_hierarchical_structure_graph.py tests/test_hierarchical_token_manager.py tests/test_hierarchical_pricing.py tests/test_hierarchical_flow_mapper.py tests/test_hierarchical_engine.py tests/test_hierarchical_runner.py -q
```

Expected: PASS.

- [ ] **Step 3: Run full test suite with coverage**

Run:

```bash
uv run pytest --cov --cov-report=term-missing -q
```

Expected: PASS and total coverage remains at least 50%.

- [ ] **Step 4: Run quality gates**

Run:

```bash
uv run ruff check .
uv run mypy
```

Expected: both pass.

- [ ] **Step 5: Run GitNexus change detection before commit**

Run:

```bash
gitnexus detect_changes
```

Expected: no HIGH or CRITICAL risk without explicit review.

- [ ] **Step 6: Final commit**

```bash
git add hierarchical_auction tests run.py
git commit -m "feat: add hierarchical multi-structure auction"
```

---

## Self-Review Checklist

- [x] The plan no longer treats accepted tokens as sufficient by themselves.
- [x] Every accepted reservation must become concrete `y[buyer,seller,function]`.
- [x] Token conflict resolution allows oversubscribed pending requests before clearing.
- [x] Token commits are cumulative.
- [x] Higher levels query adjacent structures and candidate seller nodes inside those structures.
- [x] Effective bid logic includes node price, latency, and fairness penalties.
- [x] Runner integration is separated from the first standalone runner.
- [x] Existing files are modified only when required for actual `run.py` support.
- [x] Smoke config warning covers dimension-sensitive fields such as `memory_capacity.values`.
- [x] Verification includes pytest coverage, ruff, mypy, and GitNexus.
