# Euclidean Planar Experiment Generators Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the deterministic circular-ladder topology with reproducible Euclidean planar graphs that have constant spatial density and a target mean degree, add distance-derived edge latency, and align the paper experiment suites with the revised protocol.

**Architecture:** `generate_neighborhood` remains the single topology entry point. Euclidean planar graphs are sampled from random coordinates, use Delaunay edges as the planar candidate set, retain a Euclidean minimum spanning tree for connectivity, and add candidate edges up to the requested edge budget. `add_network_latency` remains the single edge-attribute entry point and supports constant, random, Euclidean, and Euclidean-permuted latency without changing non-spatial generators.

**Tech Stack:** Python 3.10, NumPy, SciPy `spatial.Delaunay`, NetworkX, pytest.

---

### Task 1: Specify Euclidean planar topology behavior

**Files:**
- Modify: `tests/test_generate_data_extended.py`
- Modify: `generators/generate_data.py:184-230`

- [ ] **Step 1: Replace the circular-ladder test with failing geometric tests**

```python
def test_generate_euclidean_planar_neighborhood_with_target_mean_degree():
  limits = {
    "neighborhood": {
      "type": "euclidean_planar", "mean_degree": 3, "density": 0.5,
    }
  }
  matrix, graph = generate_neighborhood(20, limits, _rng(7))
  positions = nx.get_node_attributes(graph, "pos")

  assert matrix.shape == (20, 20)
  assert nx.is_connected(graph)
  assert nx.check_planarity(graph)[0]
  assert graph.number_of_edges() == round(20 * 3 / 2)
  assert len(positions) == 20
  assert all(0 <= x <= np.sqrt(20 / 0.5) for x, _ in positions.values())
  assert all(0 <= y <= np.sqrt(20 / 0.5) for _, y in positions.values())
  assert all(data["edge_length"] > 0 for _, _, data in graph.edges(data=True))


def test_euclidean_planar_neighborhood_changes_with_seed():
  limits = {
    "neighborhood": {
      "type": "euclidean_planar", "mean_degree": 3, "density": 1.0,
    }
  }
  _, first = generate_neighborhood(20, limits, _rng(1))
  _, second = generate_neighborhood(20, limits, _rng(2))
  assert nx.get_node_attributes(first, "pos") != nx.get_node_attributes(second, "pos")


def test_euclidean_planar_neighborhood_rejects_infeasible_edge_budget():
  limits = {
    "neighborhood": {
      "type": "euclidean_planar", "mean_degree": 1, "density": 1.0,
    }
  }
  with pytest.raises(ValueError, match="connected Euclidean planar"):
    generate_neighborhood(20, limits, _rng(1))
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `uv run pytest -q tests/test_generate_data_extended.py -k euclidean_planar`

Expected: failures because the current implementation returns a circular ladder and does not store coordinates or edge lengths.

- [ ] **Step 3: Implement the minimal Delaunay-backed generator**

Import `Delaunay` from `scipy.spatial`. Add a private `_generate_euclidean_planar` helper that validates positive density, samples coordinates in a square of side `sqrt(Nn / density)`, converts Delaunay simplices to candidate edges, assigns Euclidean `edge_length`, computes the minimum spanning tree, shuffles the remaining candidate edges with the supplied NumPy generator, and adds edges until `round(Nn * mean_degree / 2)` edges exist. Accept `type = "euclidean_planar"` as the canonical configuration and map the legacy planar keys `degree` or `k` to `mean_degree`.

- [ ] **Step 4: Run the focused tests**

Run: `uv run pytest -q tests/test_generate_data_extended.py -k 'euclidean_planar or neighborhood_with_k_regular or probability_neighborhood'`

Expected: all selected tests pass.

### Task 2: Generate connected fixed-edge random graphs

**Files:**
- Modify: `tests/test_generate_data_extended.py`
- Modify: `generators/generate_data.py:184-230`

- [ ] **Step 1: Write failing tests for exact edge count and connected regular graphs**

```python
def test_fixed_edge_neighborhood_is_connected_and_exact():
  limits = {"neighborhood": {"m": 30}}
  _, graph = generate_neighborhood(20, limits, _rng(3))
  assert nx.is_connected(graph)
  assert graph.number_of_edges() == 30


def test_regular_neighborhood_is_connected():
  limits = {"neighborhood": {"k": 3}}
  _, graph = generate_neighborhood(20, limits, _rng(3))
  assert nx.is_connected(graph)
```

- [ ] **Step 2: Run the tests and verify the fixed-edge test fails**

Run: `uv run pytest -q tests/test_generate_data_extended.py -k 'fixed_edge or regular_neighborhood_is_connected'`

Expected: the `m` configuration fails because no graph is created.

- [ ] **Step 3: Add bounded connected rejection sampling**

Generate `G(n,m)` and random-regular candidates with integer seeds drawn from the supplied NumPy generator. Retry at most 1,000 times and raise a descriptive `ValueError` when no connected graph is obtained. Keep the existing connected `G(n,p)` behavior.

- [ ] **Step 4: Run the focused tests**

Run: `uv run pytest -q tests/test_generate_data_extended.py -k 'neighborhood'`

Expected: all neighborhood tests pass.

### Task 3: Derive latency from Euclidean edge length

**Files:**
- Modify: `tests/test_generate_data_extended.py`
- Modify: `generators/generate_data.py:14-28`

- [ ] **Step 1: Write failing latency tests**

```python
def test_add_network_latency_from_euclidean_length():
  graph = nx.Graph()
  graph.add_edge(0, 1, edge_length=2.5)
  limits = {"weights": {"edge_network_latency": {
    "mode": "euclidean", "base": 1.0, "distance_factor": 2.0,
    "jitter": {"min": 0.0, "max": 0.0},
  }}}
  result = add_network_latency(graph, limits, _rng(1))
  assert result.edges[0, 1]["network_latency"] == 6.0


def test_permuted_euclidean_latency_preserves_values():
  graph = nx.path_graph(4)
  for index, edge in enumerate(graph.edges(), start=1):
    graph.edges[edge]["edge_length"] = float(index)
  base = {"base": 0.0, "distance_factor": 1.0,
          "jitter": {"min": 0.0, "max": 0.0}}
  direct = add_network_latency(graph.copy(), {"weights": {
    "edge_network_latency": {**base, "mode": "euclidean"}}}, _rng(4))
  permuted = add_network_latency(graph.copy(), {"weights": {
    "edge_network_latency": {**base, "mode": "euclidean_permuted"}}}, _rng(4))
  assert sorted(nx.get_edge_attributes(direct, "network_latency").values()) == sorted(
    nx.get_edge_attributes(permuted, "network_latency").values()
  )
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `uv run pytest -q tests/test_generate_data_extended.py -k 'euclidean_latency'`

Expected: failure because the current function expects random `min` and `max` bounds.

- [ ] **Step 3: Extend the shared latency function**

For Euclidean modes, require `edge_length` and compute `base + distance_factor * edge_length + jitter`. Permute the completed latency vector only for `euclidean_permuted`. Preserve constant latency when no configuration is present and the existing bounded-random mode for configurations without a `mode` key. Assign bandwidth only when `edge_bandwidth` is configured.

- [ ] **Step 4: Run latency and weight tests**

Run: `uv run pytest -q tests/test_generate_data_extended.py -k 'latency or weights'`

Expected: all selected tests pass.

### Task 4: Align the paper experiment suites

**Files:**
- Modify: `remote_experiments/definitions/paper.py`
- Modify: `tests/test_paper_experiment_suites.py`

- [ ] **Step 1: Write failing suite tests**

Update E3 to expect six topology cells: Euclidean planar, random regular, and connected `G(n,m)`, each at target mean degree 3 or 5. Add E8 registration and assert its default count is 720, its node counts are `{20, 50, 100}`, and its latency modes are `{"euclidean", "euclidean_permuted"}`.

- [ ] **Step 2: Run the suite tests and verify failure**

Run: `uv run pytest -q tests/test_paper_experiment_suites.py`

Expected: E3 count/topology assertions fail and E8 is missing.

- [ ] **Step 3: Replace planar configurations and add E8**

Use `{"type": "euclidean_planar", "mean_degree": 3, "density": 1.0}` for the default planar topology. In E3, use mean degrees 3 and 5 and compute `m = round(50 * mean_degree / 2)` for `G(n,m)`. Add `paper-e8-spatial-latency` for 20, 50, and 100 nodes, the two Euclidean latency modes, four trade-off-compatible algorithms, and 30 seeds.

- [ ] **Step 4: Run suite tests**

Run: `uv run pytest -q tests/test_paper_experiment_suites.py`

Expected: all tests pass and E3/E8 counts are 1,080 and 720.

### Task 5: Verify the complete change

**Files:**
- Modify: `docs/hierarchical_model_experiment_plan.md` only if configuration names need synchronization

- [ ] **Step 1: Run focused and regression tests**

Run: `uv run pytest -q tests/test_generate_data_extended.py tests/test_paper_experiment_suites.py tests/test_e2e_gurobi_planar.py`

Expected: all available tests pass; Gurobi-dependent tests may skip when no license is available.

- [ ] **Step 2: Run static checks**

Run: `uv run ruff check generators/generate_data.py remote_experiments/definitions/paper.py tests/test_generate_data_extended.py tests/test_paper_experiment_suites.py`

Expected: `All checks passed!`

- [ ] **Step 3: Generate one real paper instance per topology family**

Run: `uv run pytest -q tests/test_paper_experiment_suites.py::test_generated_config_builds_a_real_instance`

Expected: pass with a connected graph and the configured node count.

- [ ] **Step 4: Review affected execution flows**

Run GitNexus change detection over all uncommitted changes and confirm that only instance generation, paper suite definition, tests, and experiment documentation are affected.
