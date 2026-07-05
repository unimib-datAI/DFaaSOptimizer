# PLASMA Quality & Performance Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the quality gap found in the instance trials (64% objective gap vs LMM, 25% rejections, 45 s/step) with three targeted changes: local-first admission, exact per-node DP for Layer B (dSB kept as ablation), and window-vectorized overflow routing. Then re-measure on the same instance.

**Architecture:** All changes stay inside `plasma/`; no new dependencies; the runner/engine/heartbeat contracts are unchanged except the node data-plane API (Task 3). Evidence base: `solutions/fromexisting_plasma*` trials — replicas already match the oracle, the gap is routing mix (40% local vs LMM's 75%) plus pure-Python per-request routing cost.

**Tech Stack:** unchanged (numpy, pytest via `uv run pytest`).

## Global Constraints

- 2-space indent, double quotes, no new deps, Python 3.10.
- Only `plasma/*` and `tests/test_plasma_*.py` may change (no runner CSV format changes, no `run.py` changes).
- Determinism: explicit `numpy.random.Generator` everywhere; same seed → same outputs.
- Locality invariant untouched: nodes still read only own state + neighbor heartbeats/ACKs.
- Existing test semantics: tests may be ADAPTED where the plan says so, never weakened (assertions must still prove the property).
- Commit per task; end commit messages with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Local-first admission — Physarum only on overflow

**Files:**
- Modify: `plasma/core/node.py` (`route_request`)
- Test: `tests/test_plasma_routing.py`

**Rationale (from trials):** sampling LOCAL against neighbors/REJ makes nodes forward/reject traffic they could serve locally (40% local vs LMM's 75%); the reference model semantics (every other method in this repo) is: serve locally up to capacity (`x`), coordinate only the overflow (`omega`).

**Interfaces:** `route_request(f, round_) -> int` keeps its signature and column convention. New semantics: while local capacity remains, return `LOCAL` deterministically; once exhausted, sample ONLY over `[REJ] + neighbors` (LOCAL weight forced to 0). `unsplittable` mode applies to the overflow choice. Conductance updates unchanged (`phi[f, LOCAL]` still counts local admissions).

- [ ] **Step 1: Write the failing test** (append to `tests/test_plasma_routing.py`)

```python
def test_local_first_admission_fills_local_capacity_before_any_forward():
  # capacity 10: the FIRST 10 requests must all go LOCAL, deterministically
  node = _node(r=(2,), u_max=(5.0,))
  node.begin_window()
  cols = [node.route_request(0, round_=0) for _ in range(15)]
  assert cols[:10] == [LOCAL] * 10
  assert LOCAL not in cols[10:]
```

- [ ] **Step 2: Run it** — `uv run pytest tests/test_plasma_routing.py::test_local_first_admission_fills_local_capacity_before_any_forward -v` — expected FAIL (sampling sends some early requests elsewhere).

- [ ] **Step 3: Implement** — in `route_request`, replace the weight/choice block:

```python
  def route_request(self, f: int, round_: int) -> int:
    self._arrivals[f] += 1
    if self._admitted[f] < self._capacity_units(f):
      # local-first: serve own load up to capacity (reference-model x),
      # Physarum coordinates only the overflow
      self._x[f] += 1
      self._admitted[f] += 1
      self._phi[f, LOCAL] += 1
      return LOCAL
    nbr_spare = np.array([
      self.cache.spare(
        j, round_, self.opts.staleness_rounds, self.Nf
      )[f] for j in self.params.nbrs
    ])
    weights = target_weights(
      self.D[f], False, nbr_spare, self.opts.eps_explore
    )
    unsplittable = (
      self.opts.rare_function_mode == "unsplittable"
      and self.lam_hat[f] < self.opts.lambda_split_threshold
    )
    col = choose_target(self.rng, weights, unsplittable)
    if col == LOCAL:  # weight is 0; only reachable if every weight is 0
      col = REJ
    if col == REJ:
      self._z[f] += 1
      self._pull[f] += 1
    return col
```

Note: `choose_target` returns REJ when the total weight is ≤ 0, and with `local_open=False` the LOCAL weight is 0 — the `col == LOCAL` guard covers the argmax path of unsplittable mode (argmax could pick a zero-weight column only if all are zero).

- [ ] **Step 4: Run the routing + clock + e2e suites** — `uv run pytest tests/test_plasma_routing.py tests/test_plasma_clock.py tests/test_plasma_e2e.py -q` — all green. The LP-convergence acceptance test must still pass (local-first gives x = capacity = LP optimum exactly). Seeded engine tests may shift values but their assertions are structural; if a seeded assertion fails, inspect whether the property genuinely still holds under the new (better) routing and adjust the SEED only, never the asserted property.

- [ ] **Step 5: Commit** — `git commit -m "route local-first, physarum on overflow only"`

---

### Task 2: Exact per-node DP for Layer B (`sbm_method: exact|dsb`)

**Files:**
- Modify: `plasma/core/types.py` (new option + validation), `plasma/core/sbm.py` (DP), `plasma/core/node.py` (`sb_pass` dispatch)
- Test: `tests/test_plasma_sbm.py`, `tests/test_plasma_protocol.py` (option validation)

**Rationale:** the per-node Hamiltonian is separable per function with one RAM knapsack constraint — a multi-choice knapsack, exactly solvable by DP in microseconds. dSB (30–40 s/step) stays as ablation (`sbm_method: "dsb"`).

**Interfaces:**
- `PlasmaOptions.sbm_method: str = "exact"`, validated in `__post_init__` (`ValueError` unless in `{"exact", "dsb"}`).
- `sbm.exact_minimize(ctx: HamiltonianContext, r_max: np.ndarray) -> np.ndarray` — returns the exact argmin of the Hamiltonian under the HARD RAM constraint (the soft `A` penalty is irrelevant on the feasible set; no repair needed). Requires integer `ram_req`/`ram_cap` (repo-wide the case); raises `ValueError` telling the user to use `sbm_method: "dsb"` otherwise.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_plasma_sbm.py`; option test to `tests/test_plasma_protocol.py`)

```python
# test_plasma_sbm.py
from plasma.core.sbm import exact_minimize


def test_exact_matches_brute_force_ground_state():
  rng = np.random.default_rng(0)
  for trial in range(50):
    Nf = 3
    ram_req = rng.integers(1, 4, Nf).astype(float)
    ram_cap = float(rng.integers(4, 13))
    r_max = np.floor(ram_cap / ram_req).astype(int)
    ctx = HamiltonianContext(
      benefit=rng.uniform(0, 5, Nf), ram_req=ram_req, ram_cap=ram_cap,
      demand_hat=rng.uniform(0, 10, Nf), margin=rng.uniform(0, 2, Nf),
      u_max=rng.uniform(1, 6, Nf), r_prev=rng.integers(0, 3, Nf),
      A=10.0, B=1.0, C=0.1, switch_cost=1.0,
    )
    r_star = exact_minimize(ctx, r_max)
    assert (ctx.ram_req * r_star).sum() <= ctx.ram_cap + 1e-9
    # brute force over all feasible r
    from itertools import product as iproduct
    best = min(
      (hamiltonian(np.array(rr), ctx)
       for rr in iproduct(*[range(m + 1) for m in r_max])
       if (ctx.ram_req * np.array(rr)).sum() <= ctx.ram_cap + 1e-9)
    )
    assert hamiltonian(r_star, ctx) <= best + 1e-9


def test_exact_rejects_fractional_ram():
  ctx = _ctx(ram_req=np.array([1.5, 2.0]))
  with pytest.raises(ValueError, match="dsb"):
    exact_minimize(ctx, np.array([5, 4]))


def test_sb_pass_exact_is_deterministic():
  a = _sb_node(n_hyst=1)
  b = _sb_node(n_hyst=1)
  for node in (a, b):
    node.demand_hat = np.array([8.0, 3.0])
    node.sb_pass(round_=0)
  assert np.array_equal(a.r, b.r)
```

```python
# test_plasma_protocol.py
def test_options_reject_unknown_sbm_method():
  with pytest.raises(ValueError, match="sbm_method"):
    PlasmaOptions(sbm_method="quantum")
```

- [ ] **Step 2: Run them** — expected FAIL (`ImportError: exact_minimize`, no validation).

- [ ] **Step 3: Implement**

`plasma/core/types.py` — add field `sbm_method: str = "exact"` in the Layer B block and in `__post_init__`:

```python
    if self.sbm_method not in ("exact", "dsb"):
      raise ValueError(f"sbm_method must be 'exact' or 'dsb', got {self.sbm_method!r}")
```

`plasma/core/sbm.py`:

```python
def _level_values(ctx: HamiltonianContext, f: int, r_max_f: int) -> np.ndarray:
  # per-function contribution of r_f = 0..r_max_f (Hamiltonian minus RAM term,
  # which the DP enforces as a hard budget)
  levels = np.arange(r_max_f + 1)
  cap_short = np.maximum(
    0.0, ctx.demand_hat[f] + ctx.margin[f] - levels * ctx.u_max[f]
  )
  return (
    -ctx.benefit[f] * levels
    + ctx.B * cap_short ** 2
    + ctx.C * ctx.switch_cost * np.abs(levels - ctx.r_prev[f])
  )


def exact_minimize(ctx: HamiltonianContext, r_max: np.ndarray) -> np.ndarray:
  # multi-choice knapsack DP over the integer RAM budget: exact argmin of the
  # Hamiltonian under the hard RAM constraint (per-node problem is separable
  # per function; RAM is the only coupling)
  ram_req = np.rint(ctx.ram_req).astype(int)
  budget = int(np.floor(ctx.ram_cap + 1e-9))
  if not np.allclose(ctx.ram_req, ram_req, atol=1e-9) or (ram_req <= 0).any():
    raise ValueError(
      "exact_minimize requires positive integer ram_req; use sbm_method 'dsb'"
    )
  Nf = len(r_max)
  INF = np.inf
  best = np.full(budget + 1, 0.0)  # value of best partial assignment
  choice = np.zeros((Nf, budget + 1), dtype=int)
  for f in range(Nf):
    values = _level_values(ctx, f, int(r_max[f]))
    new_best = np.full(budget + 1, INF)
    for b in range(budget + 1):
      k_hi = min(int(r_max[f]), b // ram_req[f])
      for k in range(k_hi + 1):
        cand = best[b - k * ram_req[f]] + values[k]
        if cand < new_best[b]:
          new_best[b] = cand
          choice[f, b] = k
    best = new_best
  # backtrack from the best final budget
  b = int(np.argmin(best))
  r = np.zeros(Nf, dtype=int)
  for f in range(Nf - 1, -1, -1):
    r[f] = choice[f, b]
    b -= r[f] * ram_req[f]
  return r
```

`plasma/core/node.py` — in `sb_pass`, replace the dSB block:

```python
    if self.opts.sbm_method == "exact":
      r_new = exact_minimize(ctx, r_max)
    else:
      bits = bits_per_fn(r_max)
      n_spins = int(bits.sum())
      if n_spins == 0:
        return False

      def H(s: np.ndarray) -> float:
        return hamiltonian(decode_spins(s, bits, r_max), ctx)

      s = dsb_minimize(H, n_spins, self.opts, self.rng)
      r_new = repair(
        decode_spins(s, bits, r_max), ctx.benefit, self.params.ram_req,
        self.params.ram_cap,
      )
```

(import `exact_minimize` alongside the existing sbm imports; keep everything after — hysteresis, `p_commit` — identical.)

- [ ] **Step 4: Run** — `uv run pytest tests/test_plasma_sbm.py tests/test_plasma_protocol.py tests/test_plasma_e2e.py -q` — all green. Existing `_sb_node` tests now run the exact path (deterministic, still RAM-feasible). dSB tests still call `dsb_minimize` directly and stay green.

- [ ] **Step 5: Commit** — `git commit -m "solve layer B exactly via knapsack DP, keep dSB as ablation"`

---

### Task 3: Window-vectorized overflow routing

**Files:**
- Modify: `plasma/core/node.py` (window API), `plasma/engine.py` (`_route_all`)
- Test: `tests/test_plasma_routing.py`, `tests/test_plasma_clock.py` (adapt data-plane tests)

**Rationale:** the hot loop is per-request Python (`cache.spare` array per request × neighbor): 45→697 s/step in the trials. Weights are CONSTANT within a window (heartbeat cache only changes between rounds; local gate is now deterministic), so sequential sampling ≡ one multinomial draw per (node, f).

**Interfaces (replaces the per-request data plane):**
- `PlasmaNode.route_window(arrivals: np.ndarray, round_: int) -> np.ndarray` — arrivals shape `(Nf,)`; admits local-first internally (fills `x`, `phi[:, LOCAL]`, `admitted`), samples the overflow once per function (multinomial over `[REJ]+nbrs`, or argmax for unsplittable), records REJ into `z`/`pull`, and returns desired forward counts, shape `(Nf, deg)`.
- `PlasmaNode.accept_forwards(f: int, n: int) -> int` — bulk admission: returns `k = min(n, remaining_units)`, adds to `admitted`/`xi`.
- `PlasmaNode.record_forward_results(f: int, k: int, attempted: int, accepted: int)` — `y[k, f] += accepted`, `phi[f, 2+k] += accepted`, `z[f] += attempted - accepted`, `pull[f] += attempted`.
- `route_request`/`admit_forward`/`record_forward_result` are DELETED (engine is the only caller; tests migrate to the window API).

Engine `_route_all` becomes:

```python
  def _route_all(self, round_: int, arrivals: np.ndarray) -> None:
    for i, node in enumerate(self.nodes):
      if not node.alive:
        continue
      desired = node.route_window(arrivals[i], round_)
      for k, j in enumerate(node.params.nbrs):
        for f in range(node.Nf):
          n = int(desired[f, k])
          if n == 0:
            continue
          self.msg_count += n
          accepted = self.nodes[j].accept_forwards(f, n)
          node.record_forward_results(f, k, n, accepted)
```

`route_window` core (per function `f` with `overflow > 0`):

```python
    nbr_spare = self._nbr_spare(round_)  # (deg, Nf), ONE cache read per window
    weights = target_weights(self.D[f], False, nbr_spare[:, f], self.opts.eps_explore)
    weights[LOCAL] = 0.0
    if unsplittable:
      counts = np.zeros(len(weights), dtype=int)
      counts[int(np.argmax(weights))] = overflow
    else:
      total = weights.sum()
      if total <= 0.0:
        counts = np.zeros(len(weights), dtype=int)
        counts[REJ] = overflow
      else:
        counts = self.rng.multinomial(overflow, weights / total)
    counts[REJ] += counts[LOCAL]  # zero-weight LOCAL can only be hit by argmax ties
    self._z[f] += counts[REJ]
    self._pull[f] += overflow
    desired[f, :] = counts[2:]
```

with `_nbr_spare(round_)` building the `(deg, Nf)` spare matrix once per window via `cache.spare`.

**Distributional note (binding):** multinomial over fixed weights is exactly the law of the per-request categorical sequence, so the acceptance tests (LP convergence band) must still pass — that is the regression witness for this task.

- [ ] **Step 1: Write the new-API tests first** (adapt in place in `tests/test_plasma_routing.py`): rewrite `test_capacity_gate_never_admits_beyond_r_umax`, `test_incoming_forwards_share_the_same_capacity`, `test_nack_counts_as_origin_rejection_and_pull`, `test_zero_replicas_rejects_or_forwards_everything`, `test_dead_node_admits_nothing`, `test_end_window_reinforces_local_conductance`, `test_local_first_admission_...`, `test_spare_advertises_floored_capacity` to the window API, preserving each asserted property. Example conversions:

```python
def test_capacity_gate_never_admits_beyond_r_umax():
  node = _node(r=(2,), u_max=(5.0,))
  node.begin_window()
  desired = node.route_window(np.array([100]), round_=0)
  counts = node.end_window()
  assert counts.x[0] == 10                       # local-first fills capacity
  assert counts.z[0] + desired[0].sum() == 90    # overflow rejected or forwarded


def test_incoming_forwards_share_the_same_capacity():
  node = _node(r=(1,), u_max=(3.0,))
  node.begin_window()
  assert node.accept_forwards(0, 10) == 3
  desired = node.route_window(np.array([5]), round_=0)
  counts = node.end_window()
  assert counts.x[0] == 0                        # capacity consumed by forwards


def test_nack_counts_as_origin_rejection_and_pull():
  node = _node()
  node.begin_window()
  node.record_forward_results(0, k=0, attempted=2, accepted=1)
  counts = node.end_window()
  assert counts.z[0] == 1 and counts.y[0, 0] == 1
  assert node.make_heartbeat().pull[0] == 2
```

- [ ] **Step 2: Run** — new/adapted tests FAIL (`route_window` missing).

- [ ] **Step 3: Implement** node + engine as specified. Delete the per-request methods.

- [ ] **Step 4: Run everything** — `uv run pytest tests/test_plasma_routing.py tests/test_plasma_clock.py tests/test_plasma_protocol.py tests/test_plasma_sbm.py tests/test_plasma_e2e.py -q` — all green, INCLUDING the LP-convergence band (distributional equivalence witness). Then timing check: `uv run python -m pytest tests/test_plasma_e2e.py::test_runner_produces_lspc_artifacts -q --durations=1` — the e2e should be visibly faster than before.

- [ ] **Step 5: Commit** — `git commit -m "vectorize window routing with bulk forward resolution"`

---

### Task 4: Re-measure on the instance (controller-run)

- [ ] Re-run: `uv run python run.py -c config_files/config_fromexisting_plasma.json --methods plasma --n_experiments 1 -j 2` (centralized results already exist in `solutions/fromexisting_plasma/`; the runner appends a new plasma folder).
- [ ] Compare vs LMM and vs the pre-fix trials: objective gap per step, rejection %, local/fwd mix, runtime/step. Success bar: rejections ≤ 10%, mean gap ≤ 25%, runtime ≤ 2 s/step. Report the table either way — if the bar is missed, report the residual bottleneck (this is measurement, not tuning; M5 sweeps remain future work).
- [ ] Full regression: `uv run pytest tests/ -q`.
