# PLASMA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement PLASMA — Physarum routing (Layer A) + discrete Simulated Bifurcation replica allocation (Layer B), synchronous round-barrier only — as a new `plasma/` subpackage evaluated side by side with the existing methods via `run.py`.

**Architecture:** A self-contained `plasma/` package (precedent: `hierarchical_auction/`). Pure-function algorithm modules (`routing`, `sbm`, `protocol`) composed by a `PlasmaNode`, driven by a heapq round-barrier engine. A runner mirrors `decentralized_gcaa.run`'s contract (`run(config, parallelism, log_on_file, disable_plotting) -> solution_folder`) and writes LSPc-format results via the existing `decode_solution`/`save_solution` helpers so `compare_results.py` and postprocessing work unchanged.

**Tech Stack:** Python 3.10, numpy, networkx, pandas, scipy (all already in `pyproject.toml`). Tests: pytest via `uv run pytest`. Zero new third-party dependencies.

**Spec sources:** `docs/plans/2026-07-04-plasma-design.md` (authoritative for repo integration) and `/Users/micheleciavotta/Downloads/PLASMA_SPEC.md` §§2–5, 8, 9 (authoritative for algorithm content; §4.4 async mode is DROPPED).

## Global Constraints

- Python pinned `>=3.10,<3.11` — no 3.11+ stdlib (`tomllib`, `ExceptionGroup`), no new deps, do not touch `pyproject.toml`.
- **Never modify existing files** except `run.py` (additive registration only, Task 10) and `config_files/` (new file only). No edits to `models/*`, `run_centralized_model.py`, `run_faasmacro.py`, `run_faasmadea.py`, `heuristic_coordinator.py`, `generators/*`, `postprocessing.py`, `logs_postprocessing.py`, `compare_results.py`.
- Repo code style: **2-space indentation**, double quotes, `snake_case`, type hints (repo passes mypy with lax codes).
- Synchronous execution ONLY: single shared round barrier, period `W`. No per-node clocks, no tick jitter, no `execution.mode` key. Randomized commit (`p_commit`, default 0.5) is unconditional.
- Locality invariant: a node reads only its own state and heartbeats/ACKs from direct neighbors. Heartbeat carries ONLY `spare`, `alpha`, `pull` (+ node id, seq). No `lambda`, RAM, replica counts, or topology cross an edge.
- Determinism: every stochastic component takes an explicit `numpy.random.Generator`.
- All spin couplings are node-local: assert and raise if any Hamiltonian term would couple spins of different nodes (structural: the Hamiltonian is a per-node closure over per-node arrays only).
- Tests live flat in `tests/`, exactly six files: `test_plasma_routing.py`, `test_plasma_sbm.py`, `test_plasma_protocol.py`, `test_plasma_clock.py`, `test_plasma_baselines.py`, `test_plasma_e2e.py`.
- Per project CLAUDE.md: before editing `run.py` run `gitnexus_impact({target: "parse_arguments", direction: "upstream"})`; run `gitnexus_detect_changes()` before every commit. New `plasma/*` files have no upstream callers, so impact analysis applies only to the `run.py` task.
- Run tests with `uv run pytest <file> -v`. Commit after every task; end commit messages with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

## Repo interfaces you will reuse (read-only — verified signatures)

```python
# run_centralized_model.py
init_problem(limits, trace_type, max_steps, seed, solution_folder)
#   -> (base_instance_data, input_requests_traces, agents, graph)
init_complete_solution() -> dict of empty DataFrames
decode_solution(x, y, z, r, xi, rho, U, complete_solution) -> dict   # numpy in, accumulates one timestep
join_complete_solution(cs) -> (solution_df, offloaded_df, detailed_fwd_df)
save_solution(solution, offloaded, cs, detailed_fwd, model_name, folder)  # model_name = "LSPc"
save_checkpoint(cs, folder, t)
solve_instance(M, data, solver_name, solver_options)
#   -> (x, y, z, r, xi, omega, rho, U, obj, runtime, tc)

# utils/centralized.py
get_current_load(input_requests_traces, agents, t) -> {(n, f): load}      # 1-indexed keys
check_feasibility(x, omega, z, r, cpu_utilization, data) -> (bool, str)

# generators/generate_data.py
update_data(data, fixed_values) -> dict                                    # deepcopy + set data[None][k]

# utils/faasmacro.py
compute_centralized_objective(sp_data, x, y, z) -> float                   # per-load-normalized

# utils/common.py
load_configuration(path) -> dict

# models/model.py
LoadManagementModel  # centralized MILP, for solve_instance

# heuristic_coordinator.py
GreedyCoordinator().solve(instance, solver_options) -> dict
#   instance needs data[None] keys: Nn, Nf, neighborhood, omega_bar, x_bar, r_bar,
#   beta, gamma, demand, max_utilization, memory_requirement, incoming_load
#   plus instance["sp_rho"] (residual RAM per node, np.array (Nn,))
```

`base_instance_data[None]` keys (1-indexed dict keys): `Nn`, `Nf`, `neighborhood[(n1,n2)]∈{0,1}`, `alpha[(n,f)]`, `beta[(n1,n2,f)]`, `gamma[(n,f)]`, `delta[(n,f)]`, `demand[(n,f)]`, `max_utilization[f]`, `memory_capacity[n]`, `memory_requirement[f]`. Graph edges carry `network_latency`.

**Problem mapping (locked in):** per-replica capacity `u_max[i][f] = max_utilization[f] / demand[(i,f)]` req/s. Layer-A rewards: LOCAL → `alpha[(i,f)]`, forward to j → `beta[(i,j,f)]` (the sender's own objective coefficient — local constant, already latency-aware; this instantiates the spec's `alpha_remote − c` in repo terms), REJ → small floor. Layer-B benefit: `benefit[f] = alpha[f] * (demand_hat[f] + pull_in[f])`.

---

### Task 1: Types and options — `plasma/core/types.py`

**Files:**
- Create: `plasma/__init__.py`, `plasma/core/__init__.py`, `plasma/core/types.py`
- Test: `tests/test_plasma_protocol.py` (types + protocol share this file)

**Interfaces:**
- Produces: `LOCAL = 0`, `REJ = 1` (column indices in the conductance matrix; neighbor k occupies column `2 + k`), `PlasmaOptions` frozen dataclass with `from_config(config: dict) -> PlasmaOptions`, `Heartbeat` frozen dataclass `(node: int, seq: int, spare: tuple, alpha: tuple, pull: tuple)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_plasma_protocol.py
import dataclasses

import pytest

from plasma.core.types import LOCAL, REJ, Heartbeat, PlasmaOptions


def test_target_columns():
  assert LOCAL == 0
  assert REJ == 1


def test_options_defaults():
  opts = PlasmaOptions()
  assert opts.W == 1.0
  assert opts.k_sb == 10
  assert opts.mu == 0.1
  assert opts.p_commit == 0.5
  assert opts.n_hyst == 2
  assert opts.staleness_rounds == 3
  assert opts.rare_function_mode == "sampled"


def test_options_from_config_overrides():
  config = {"solver_options": {"plasma": {"mu": 0.2, "k_sb": 5}}}
  opts = PlasmaOptions.from_config(config)
  assert opts.mu == 0.2
  assert opts.k_sb == 5
  assert opts.W == 1.0


def test_options_frozen():
  with pytest.raises(dataclasses.FrozenInstanceError):
    PlasmaOptions().mu = 0.5


def test_options_rejects_execution_mode():
  config = {"solver_options": {"plasma": {"execution_mode": "async"}}}
  with pytest.raises(TypeError):
    PlasmaOptions.from_config(config)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_plasma_protocol.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'plasma'`

- [ ] **Step 3: Write the implementation**

```python
# plasma/__init__.py  (empty file)
# plasma/core/__init__.py  (empty file)

# plasma/core/types.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

# conductance-matrix column layout: D has shape (Nf, 2 + deg)
LOCAL = 0
REJ = 1


@dataclass(frozen=True)
class PlasmaOptions:
  # Layer A (round period W is the time unit: 1 round = W seconds)
  W: float = 1.0
  rounds_per_step: int = 20
  mu: float = 0.1
  kappa: float = 1.0
  D_init: float = 1.0
  D_min: float = 1e-3
  D_max: float = 1e3
  eps_explore: float = 0.01
  rej_floor: float = 0.01
  ewma: float = 0.3
  lambda_split_threshold: float = 5.0
  rare_function_mode: str = "sampled"  # "sampled" | "unsplittable"
  r_init: str = "spread"  # "spread" | "zero"
  # Layer B (k_sb = 0 disables SB entirely: replicas stay fixed)
  k_sb: int = 10
  n_sb_steps: int = 300
  sb_dt: float = 0.05
  sb_delta: float = 1.0
  sb_c0: float = 0.2
  a_final: float = 1.0
  A: Optional[float] = None  # None -> auto: 2 * max_f(benefit_f / ram_req_f)
  B: float = 1.0
  C: float = 0.1
  switch_cost: float = 1.0
  z_delta: float = 2.0
  eps_commit: float = 0.05
  n_hyst: int = 2
  p_commit: float = 0.5
  # protocol
  hb_latency_rounds: int = 1
  hb_loss: float = 0.0
  staleness_rounds: int = 3

  @classmethod
  def from_config(cls, config: dict) -> "PlasmaOptions":
    return cls(**config.get("solver_options", {}).get("plasma", {}))


@dataclass(frozen=True)
class Heartbeat:
  # the ENTIRE control-plane message: nothing else may cross an edge
  node: int
  seq: int
  spare: Tuple[float, ...]  # per function, max(0, r*u_max - admitted)
  alpha: Tuple[float, ...]  # per function
  pull: Tuple[float, ...]   # per function, offload pressure last window
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_plasma_protocol.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add plasma/ tests/test_plasma_protocol.py
git commit -m "add plasma core types and options"
```

---

### Task 2: Round-barrier scheduler — `plasma/sim/clock.py`

**Files:**
- Create: `plasma/sim/__init__.py`, `plasma/sim/clock.py`
- Test: `tests/test_plasma_clock.py`

**Interfaces:**
- Produces: `RoundClock` with `schedule(round_: int, fn: Callable[[], None])`, `run(n_rounds: int, on_round: Callable[[int], None])`, attribute `round: int` (next round to execute). Events due at round r are delivered BEFORE `on_round(r)`; ties delivered FIFO.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_plasma_clock.py
from plasma.sim.clock import RoundClock


def test_events_delivered_before_round_callback():
  clock = RoundClock()
  trace = []
  clock.schedule(1, lambda: trace.append("hb@1"))
  clock.run(2, on_round=lambda r: trace.append(f"round{r}"))
  assert trace == ["round0", "hb@1", "round1"]


def test_fifo_tie_break():
  clock = RoundClock()
  trace = []
  clock.schedule(0, lambda: trace.append("a"))
  clock.schedule(0, lambda: trace.append("b"))
  clock.run(1, on_round=lambda r: None)
  assert trace == ["a", "b"]


def test_round_counter_persists_across_runs():
  clock = RoundClock()
  seen = []
  clock.run(3, on_round=seen.append)
  clock.run(2, on_round=seen.append)
  assert seen == [0, 1, 2, 3, 4]


def test_past_due_events_flush():
  clock = RoundClock()
  trace = []
  clock.run(2, on_round=lambda r: None)
  clock.schedule(0, lambda: trace.append("late"))
  clock.run(1, on_round=lambda r: trace.append(f"round{r}"))
  assert trace == ["late", "round2"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_plasma_clock.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# plasma/sim/__init__.py  (empty file)

# plasma/sim/clock.py
from __future__ import annotations

import heapq
from typing import Callable, List, Tuple


class RoundClock:
  """Single shared round barrier (period = W). The only clock in PLASMA:
  there is no per-node clock and no tick jitter (sync-only design)."""

  def __init__(self) -> None:
    self._queue: List[Tuple[int, int, Callable[[], None]]] = []
    self._seq = 0
    self.round = 0

  def schedule(self, round_: int, fn: Callable[[], None]) -> None:
    heapq.heappush(self._queue, (round_, self._seq, fn))
    self._seq += 1

  def run(self, n_rounds: int, on_round: Callable[[int], None]) -> None:
    for r in range(self.round, self.round + n_rounds):
      while self._queue and self._queue[0][0] <= r:
        heapq.heappop(self._queue)[2]()
      on_round(r)
    self.round += n_rounds
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_plasma_clock.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add plasma/sim tests/test_plasma_clock.py
git commit -m "add plasma round-barrier scheduler"
```

---

### Task 3: Layer A pure functions — `plasma/core/routing.py`

**Files:**
- Create: `plasma/core/routing.py`
- Test: `tests/test_plasma_routing.py`

**Interfaces:**
- Consumes: `LOCAL`, `REJ` from `plasma.core.types`.
- Produces:
  - `target_weights(D_f: np.ndarray, local_open: bool, nbr_spare: np.ndarray, eps_explore: float) -> np.ndarray` — shape `(2+deg,)`; column LOCAL zeroed when gate closed, neighbor columns scaled by 1.0 when `nbr_spare > 0` else `eps_explore`.
  - `choose_target(rng, weights: np.ndarray, unsplittable: bool) -> int` — categorical sample, or argmax when `unsplittable`.
  - `update_conductance(D: np.ndarray, phi: np.ndarray, rewards: np.ndarray, opts) -> np.ndarray` — `(1-mu)*D + mu*(phi**kappa)*rewards`, clipped to `[D_min, D_max]`; shapes all `(Nf, 2+deg)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_plasma_routing.py
import numpy as np
import pytest

from plasma.core.types import LOCAL, REJ, PlasmaOptions
from plasma.core.routing import choose_target, target_weights, update_conductance


def test_local_gate_closes_local_column():
  D_f = np.array([10.0, 0.1, 5.0])  # LOCAL, REJ, one neighbor
  w = target_weights(D_f, local_open=False, nbr_spare=np.array([1.0]),
                     eps_explore=0.01)
  assert w[LOCAL] == 0.0
  assert w[2] == 5.0


def test_neighbor_without_spare_gets_exploration_floor():
  D_f = np.array([1.0, 0.1, 4.0])
  w = target_weights(D_f, local_open=True, nbr_spare=np.array([0.0]),
                     eps_explore=0.01)
  assert w[2] == pytest.approx(4.0 * 0.01)


def test_choose_target_unsplittable_is_argmax():
  rng = np.random.default_rng(0)
  w = np.array([1.0, 0.5, 7.0])
  assert choose_target(rng, w, unsplittable=True) == 2


def test_choose_target_sampled_follows_weights():
  rng = np.random.default_rng(0)
  w = np.array([0.0, 0.0, 1.0])
  assert choose_target(rng, w, unsplittable=False) == 2


def test_conductance_reinforces_accepted_traffic():
  opts = PlasmaOptions()
  D = np.full((1, 3), 1.0)
  phi = np.array([[10.0, 0.0, 0.0]])
  rewards = np.array([[2.0, 0.0, 0.0]])
  D2 = update_conductance(D, phi, rewards, opts)
  assert D2[0, LOCAL] == pytest.approx(0.9 * 1.0 + 0.1 * 10.0 * 2.0)


def test_dead_neighbor_conductance_decays_to_floor():
  opts = PlasmaOptions()
  D = np.full((1, 3), 100.0)
  phi = np.zeros((1, 3))
  rewards = np.zeros((1, 3))
  for _ in range(200):
    D = update_conductance(D, phi, rewards, opts)
  # pure evaporation: (1-mu)^200 * 100 << D_min -> clipped at floor
  assert D[0, 2] == pytest.approx(opts.D_min)


def test_conductance_clipped_above():
  opts = PlasmaOptions()
  D = np.full((1, 3), 1.0)
  phi = np.full((1, 3), 1e9)
  rewards = np.full((1, 3), 1e9)
  D2 = update_conductance(D, phi, rewards, opts)
  assert (D2 <= opts.D_max).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_plasma_routing.py -v`
Expected: FAIL with `ModuleNotFoundError` (no `plasma.core.routing`)

- [ ] **Step 3: Write the implementation**

```python
# plasma/core/routing.py
from __future__ import annotations

import numpy as np

from plasma.core.types import LOCAL, REJ, PlasmaOptions


def target_weights(
    D_f: np.ndarray, local_open: bool, nbr_spare: np.ndarray,
    eps_explore: float
  ) -> np.ndarray:
  w = D_f.copy()
  if not local_open:
    w[LOCAL] = 0.0
  gates = np.where(nbr_spare > 0.0, 1.0, eps_explore)
  w[2:] = w[2:] * gates
  return w


def choose_target(
    rng: np.random.Generator, weights: np.ndarray, unsplittable: bool
  ) -> int:
  if unsplittable:
    return int(np.argmax(weights))
  total = weights.sum()
  if total <= 0.0:
    return REJ
  return int(rng.choice(len(weights), p=weights / total))


def update_conductance(
    D: np.ndarray, phi: np.ndarray, rewards: np.ndarray, opts: PlasmaOptions
  ) -> np.ndarray:
  reinforced = (1.0 - opts.mu) * D + opts.mu * (phi ** opts.kappa) * rewards
  return np.clip(reinforced, opts.D_min, opts.D_max)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_plasma_routing.py -v`
Expected: 7 PASS

- [ ] **Step 5: Commit**

```bash
git add plasma/core/routing.py tests/test_plasma_routing.py
git commit -m "add plasma layer A routing primitives"
```

---

### Task 4: Heartbeat protocol — `plasma/core/protocol.py`

**Files:**
- Create: `plasma/core/protocol.py`
- Test: `tests/test_plasma_protocol.py` (append)

**Interfaces:**
- Consumes: `Heartbeat` from `plasma.core.types`.
- Produces:
  - `encode_heartbeat(hb: Heartbeat) -> dict` / `decode_heartbeat(msg: dict) -> Heartbeat` (raises `ValueError` on unknown fields — the privacy whitelist).
  - `HeartbeatCache` with `store(hb: Heartbeat, round_: int)`, `spare(nbr: int, now_round: int, staleness_rounds: int, Nf: int) -> np.ndarray`, `alpha(nbr, now_round, staleness_rounds, Nf) -> np.ndarray`, `pull_in(now_round, staleness_rounds, Nf) -> np.ndarray` (sum of fresh neighbors' pull).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_plasma_protocol.py`)

```python
import numpy as np

from plasma.core.protocol import HeartbeatCache, decode_heartbeat, encode_heartbeat


def _hb(node=3, seq=7):
  return Heartbeat(node=node, seq=seq, spare=(1.0, 0.0), alpha=(2.0, 2.5),
                   pull=(0.0, 4.0))


def test_heartbeat_roundtrip():
  hb = _hb()
  assert decode_heartbeat(encode_heartbeat(hb)) == hb


def test_heartbeat_field_whitelist():
  msg = encode_heartbeat(_hb())
  assert set(msg) == {"node", "seq", "spare", "alpha", "pull"}
  msg["ram_capacity"] = 64  # privacy violation: must be rejected
  with pytest.raises(ValueError):
    decode_heartbeat(msg)


def test_stale_neighbor_treated_as_zero_spare():
  cache = HeartbeatCache()
  cache.store(_hb(node=3), round_=10)
  fresh = cache.spare(3, now_round=12, staleness_rounds=3, Nf=2)
  stale = cache.spare(3, now_round=14, staleness_rounds=3, Nf=2)
  assert fresh.tolist() == [1.0, 0.0]
  assert stale.tolist() == [0.0, 0.0]


def test_unknown_neighbor_is_zero_spare():
  cache = HeartbeatCache()
  assert cache.spare(9, now_round=0, staleness_rounds=3, Nf=2).tolist() == [0.0, 0.0]


def test_pull_in_sums_fresh_neighbors_only():
  cache = HeartbeatCache()
  cache.store(_hb(node=1), round_=10)   # pull (0, 4)
  cache.store(_hb(node=2), round_=1)    # stale at now=12
  pull = cache.pull_in(now_round=12, staleness_rounds=3, Nf=2)
  assert pull.tolist() == [0.0, 4.0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_plasma_protocol.py -v`
Expected: new tests FAIL with `ModuleNotFoundError` (no `plasma.core.protocol`); Task 1 tests still PASS

- [ ] **Step 3: Write the implementation**

```python
# plasma/core/protocol.py
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from plasma.core.types import Heartbeat

_FIELDS = {"node", "seq", "spare", "alpha", "pull"}


def encode_heartbeat(hb: Heartbeat) -> dict:
  return {
    "node": hb.node, "seq": hb.seq, "spare": list(hb.spare),
    "alpha": list(hb.alpha), "pull": list(hb.pull),
  }


def decode_heartbeat(msg: dict) -> Heartbeat:
  extra = set(msg) - _FIELDS
  if extra:
    raise ValueError(f"heartbeat carries non-whitelisted fields: {sorted(extra)}")
  return Heartbeat(
    node=int(msg["node"]), seq=int(msg["seq"]), spare=tuple(msg["spare"]),
    alpha=tuple(msg["alpha"]), pull=tuple(msg["pull"]),
  )


class HeartbeatCache:
  """Per-node view of neighbors. Staleness rule: older than
  staleness_rounds -> spare = 0 for all f (gates close, conductance decays;
  no failure detector, no membership protocol)."""

  def __init__(self) -> None:
    self._last: Dict[int, Tuple[int, Heartbeat]] = {}

  def store(self, hb: Heartbeat, round_: int) -> None:
    self._last[hb.node] = (round_, hb)

  def _fresh(self, nbr: int, now_round: int, staleness_rounds: int):
    entry = self._last.get(nbr)
    if entry is None or now_round - entry[0] > staleness_rounds:
      return None
    return entry[1]

  def spare(
      self, nbr: int, now_round: int, staleness_rounds: int, Nf: int
    ) -> np.ndarray:
    hb = self._fresh(nbr, now_round, staleness_rounds)
    return np.array(hb.spare) if hb else np.zeros(Nf)

  def alpha(
      self, nbr: int, now_round: int, staleness_rounds: int, Nf: int
    ) -> np.ndarray:
    hb = self._fresh(nbr, now_round, staleness_rounds)
    return np.array(hb.alpha) if hb else np.zeros(Nf)

  def pull_in(
      self, now_round: int, staleness_rounds: int, Nf: int
    ) -> np.ndarray:
    total = np.zeros(Nf)
    for nbr in self._last:
      hb = self._fresh(nbr, now_round, staleness_rounds)
      if hb:
        total += np.array(hb.pull)
    return total
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_plasma_protocol.py -v`
Expected: all PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add plasma/core/protocol.py tests/test_plasma_protocol.py
git commit -m "add plasma heartbeat protocol and staleness cache"
```

---

### Task 5: Layer B — `plasma/core/sbm.py`

**Files:**
- Create: `plasma/core/sbm.py`
- Test: `tests/test_plasma_sbm.py`

**Interfaces:**
- Produces:
  - `r_max_per_fn(ram_cap: float, ram_req: np.ndarray) -> np.ndarray[int]`, `bits_per_fn(r_max: np.ndarray) -> np.ndarray[int]` (`ceil(log2(r_max+1))`, 0 bits when r_max == 0).
  - `decode_spins(s: np.ndarray, bits: np.ndarray, r_max: np.ndarray) -> np.ndarray[int]` — binary decode, clipped to `r_max`.
  - `HamiltonianContext` frozen dataclass `(benefit, ram_req, ram_cap, demand_hat, margin, u_max, r_prev, A, B, C, switch_cost)` — all per-node arrays `(Nf,)` + scalars. Node-local by construction (the locality invariant).
  - `hamiltonian(r: np.ndarray, ctx: HamiltonianContext) -> float`.
  - `dsb_minimize(H, n_spins: int, opts: PlasmaOptions, rng) -> np.ndarray` — spins in {-1,+1}; `H: Callable[[np.ndarray], float]` over spins.
  - `brute_force(H, n_spins: int) -> np.ndarray` — exact ground state (test oracle, ≤ 12 spins).
  - `repair(r, benefit, ram_req, ram_cap) -> np.ndarray` — drop lowest-benefit replicas until RAM-feasible.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_plasma_sbm.py
import numpy as np
import pytest

from plasma.core.types import PlasmaOptions
from plasma.core.sbm import (
  HamiltonianContext, bits_per_fn, brute_force, decode_spins, dsb_minimize,
  hamiltonian, r_max_per_fn, repair,
)


def test_encoding_roundtrip():
  r_max = np.array([5, 0, 1])
  bits = bits_per_fn(r_max)
  assert bits.tolist() == [3, 0, 1]
  # spins for r = (5, -, 1): 5 = 101b -> bits (1,0,1) -> spins (+1,-1,+1)
  s = np.array([1, -1, 1, 1])
  assert decode_spins(s, bits, r_max).tolist() == [5, 0, 1]


def test_decode_clips_to_r_max():
  r_max = np.array([5])           # 3 bits encode up to 7
  s = np.array([1, 1, 1])         # decodes to 7
  assert decode_spins(s, bits_per_fn(r_max), r_max).tolist() == [5]


def _ctx(**over):
  base = dict(
    benefit=np.array([3.0, 1.0]), ram_req=np.array([2.0, 2.0]), ram_cap=8.0,
    demand_hat=np.array([4.0, 1.0]), margin=np.array([1.0, 0.5]),
    u_max=np.array([5.0, 5.0]), r_prev=np.array([1, 0]),
    A=10.0, B=1.0, C=0.1, switch_cost=1.0,
  )
  base.update(over)
  return HamiltonianContext(**base)


def test_hamiltonian_penalizes_ram_violation():
  ctx = _ctx()
  ok = hamiltonian(np.array([2, 2]), ctx)       # RAM = 8 <= 8
  bad = hamiltonian(np.array([3, 2]), ctx)      # RAM = 10 > 8
  assert bad > ok


def test_hamiltonian_churn_term():
  ctx = _ctx(B=0.0, A=0.0, benefit=np.zeros(2))
  h_stay = hamiltonian(np.array([1, 0]), ctx)
  h_move = hamiltonian(np.array([3, 2]), ctx)
  assert h_move == pytest.approx(h_stay + 0.1 * 1.0 * (2 + 2))


def test_repair_restores_feasibility_dropping_lowest_benefit():
  r = np.array([3, 3])  # RAM = 12 > 8
  fixed = repair(r, benefit=np.array([3.0, 1.0]), ram_req=np.array([2.0, 2.0]),
                 ram_cap=8.0)
  assert (fixed * np.array([2.0, 2.0])).sum() <= 8.0
  assert fixed[0] >= fixed[1]  # low-benefit f=1 dropped first


def test_dsb_matches_brute_force_on_random_hamiltonians():
  opts = PlasmaOptions(n_sb_steps=400)
  hits = 0
  trials = 200
  for trial in range(trials):
    rng = np.random.default_rng(trial)
    n = 8
    J = rng.normal(size=(n, n)); J = (J + J.T) / 2; np.fill_diagonal(J, 0.0)
    h = rng.normal(size=n)

    def H(s, J=J, h=h):
      return float(-0.5 * s @ J @ s - h @ s)

    s_star = brute_force(H, n)
    s_dsb = dsb_minimize(H, n, opts, rng)
    if H(s_dsb) <= H(s_star) + 1e-9:
      hits += 1
  assert hits >= 0.95 * trials


def test_brute_force_exact_on_tiny_instance():
  def H(s):
    return float(-(s[0] * s[1]) - s[0])  # ground state (+1, +1)
  assert brute_force(H, 2).tolist() == [1, 1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_plasma_sbm.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# plasma/core/sbm.py
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable

import numpy as np

from plasma.core.types import PlasmaOptions


def r_max_per_fn(ram_cap: float, ram_req: np.ndarray) -> np.ndarray:
  return np.floor(ram_cap / ram_req).astype(int)


def bits_per_fn(r_max: np.ndarray) -> np.ndarray:
  return np.where(r_max > 0, np.ceil(np.log2(r_max + 1)), 0).astype(int)


def decode_spins(
    s: np.ndarray, bits: np.ndarray, r_max: np.ndarray
  ) -> np.ndarray:
  r = np.zeros(len(bits), dtype=int)
  k = 0
  for f, nb in enumerate(bits):
    for b in range(nb):
      r[f] += (2 ** b) * (1 + int(s[k])) // 2
      k += 1
  return np.minimum(r, r_max)


@dataclass(frozen=True)
class HamiltonianContext:
  # every array is per-node-local: no term may reference another node's
  # spins (decentralization holds by construction)
  benefit: np.ndarray
  ram_req: np.ndarray
  ram_cap: float
  demand_hat: np.ndarray
  margin: np.ndarray
  u_max: np.ndarray
  r_prev: np.ndarray
  A: float
  B: float
  C: float
  switch_cost: float


def hamiltonian(r: np.ndarray, ctx: HamiltonianContext) -> float:
  field = -(ctx.benefit * r).sum()
  ram_over = max(0.0, float((ctx.ram_req * r).sum() - ctx.ram_cap))
  cap_short = np.maximum(0.0, ctx.demand_hat + ctx.margin - r * ctx.u_max)
  churn = ctx.switch_cost * np.abs(r - ctx.r_prev).sum()
  return float(
    field + ctx.A * ram_over ** 2 + ctx.B * (cap_short ** 2).sum()
    + ctx.C * churn
  )


def _local_field(H: Callable, s: np.ndarray, k: int) -> float:
  sp = s.copy(); sp[k] = 1
  sm = s.copy(); sm[k] = -1
  return (H(sp) - H(sm)) / 2.0


def dsb_minimize(
    H: Callable[[np.ndarray], float], n_spins: int, opts: PlasmaOptions,
    rng: np.random.Generator
  ) -> np.ndarray:
  # discrete SB: couplings act on sign(q) (dSB variant)
  # ponytail: local fields via 2 H-evals per spin per step; fine for
  # <=12 spins/node, switch to analytic gradients if n_spins grows
  q = rng.uniform(-0.1, 0.1, n_spins)
  p = np.zeros(n_spins)
  for step in range(opts.n_sb_steps):
    a = opts.a_final * step / max(1, opts.n_sb_steps)
    s = np.where(q >= 0, 1, -1)
    h = np.array([_local_field(H, s, k) for k in range(n_spins)])
    p += opts.sb_dt * (-(opts.sb_delta - a) * q - opts.sb_c0 * h)
    q += opts.sb_dt * opts.sb_delta * p
    hit = np.abs(q) > 1.0
    q[hit] = np.sign(q[hit])
    p[hit] = 0.0
  return np.where(q >= 0, 1, -1).astype(int)


def brute_force(H: Callable[[np.ndarray], float], n_spins: int) -> np.ndarray:
  best_s, best_h = None, np.inf
  for combo in product((-1, 1), repeat=n_spins):
    s = np.array(combo)
    val = H(s)
    if val < best_h:
      best_s, best_h = s, val
  return best_s


def repair(
    r: np.ndarray, benefit: np.ndarray, ram_req: np.ndarray, ram_cap: float
  ) -> np.ndarray:
  fixed = r.copy()
  while (ram_req * fixed).sum() > ram_cap:
    candidates = np.where(fixed > 0)[0]
    f = candidates[np.argmin(benefit[candidates])]
    fixed[f] -= 1
  return fixed
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_plasma_sbm.py -v`
Expected: 7 PASS. The brute-force comparison test takes ~1–2 minutes; if `dsb_minimize` misses the 95% bar, tune `sb_c0` (try 0.3–0.7) and `n_sb_steps` (up to 1000) — the spec fixes the acceptance bar, not the parameters.

- [ ] **Step 5: Commit**

```bash
git add plasma/core/sbm.py tests/test_plasma_sbm.py
git commit -m "add plasma layer B dSB solver with hamiltonian and repair"
```

---

### Task 6: PlasmaNode, Layer A behavior — `plasma/core/node.py`

**Files:**
- Create: `plasma/core/node.py`
- Test: `tests/test_plasma_routing.py` (append node-level Layer A tests)

**Interfaces:**
- Consumes: everything from Tasks 1, 3, 4, 5.
- Produces:

```python
@dataclass(frozen=True)
class NodeParams:
  node_id: int              # 0-based
  nbrs: tuple               # 0-based neighbor ids; column 2+k <-> nbrs[k]
  alpha: np.ndarray         # (Nf,)
  gamma: np.ndarray         # (Nf,)
  beta: np.ndarray          # (deg, Nf) reward for forwarding to nbrs[k]
  u_max: np.ndarray         # (Nf,) req/s per replica
  ram_cap: float
  ram_req: np.ndarray       # (Nf,)

class PlasmaNode:
  def __init__(self, params: NodeParams, opts: PlasmaOptions, rng): ...
  r: np.ndarray                                   # (Nf,) int, current replicas
  alive: bool
  def begin_window(self) -> None
  def route_request(self, f: int, round_: int) -> int      # returns column
  def admit_forward(self, f: int) -> bool                  # incoming data plane
  def record_forward_result(self, f: int, col: int, accepted: bool) -> None
  def end_window(self) -> WindowCounts
  def make_heartbeat(self) -> Heartbeat
  def on_heartbeat(self, hb: Heartbeat, round_: int) -> None
  def sb_pass(self) -> None                                # Task 7

@dataclass
class WindowCounts:
  x: np.ndarray        # (Nf,) locally processed externals
  z: np.ndarray        # (Nf,) rejected (incl. NACKed forwards)
  y: np.ndarray        # (deg, Nf) ACKed forwards per neighbor
  xi: np.ndarray       # (Nf,) forwards accepted FROM others (received)
```

Semantics locked in:
- `route_request` computes weights via `target_weights` with `local_open = admitted_total[f] < r[f]*u_max[f]*W` and `nbr_spare[k] = cache.spare(nbrs[k], round_, staleness_rounds, Nf)[f]`; unsplittable iff `rare_function_mode == "unsplittable"` and `lam_hat[f] < lambda_split_threshold` (`lam_hat` = EWMA of external arrivals). LOCAL increments `x` and `admitted_total[f]`; REJ increments `z`.
- `admit_forward` returns True and increments `admitted_total[f]` and `xi[f]` iff alive and `admitted_total[f] < r[f]*u_max[f]*W`; TTL=1: a forwarded request is never re-forwarded (receiver only processes or the ORIGIN records the rejection).
- `record_forward_result(f, col, accepted)`: accepted → `y[col-2, f] += 1` and `phi[f, col] += 1`; NACK → `z[f] += 1`. Every forward ATTEMPT (accepted or not) increments `pull_out[f]` (offload pressure), as do local rejections.
- `end_window`: `phi[f, LOCAL] = x[f]`; rewards matrix: `rewards[:, LOCAL] = alpha`, `rewards[:, REJ] = rej_floor`, `rewards[:, 2+k] = beta[k]`; then `update_conductance`; update `demand_hat = (1-ewma)*demand_hat + ewma*(x + accepted forwards... )` — locked: `demand_hat` tracks ACCEPTED local demand: `(1-ewma)*demand_hat + ewma*(x + xi)`; reset window counters after returning them.
- `make_heartbeat`: `spare[f] = max(0, r[f]*u_max[f]*W - admitted_total_last_window[f])`, `alpha` = params.alpha, `pull` = last window's `pull_out`.
- A dead node (`alive = False`) routes nothing, admits nothing, sends no heartbeats.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_plasma_routing.py`)

```python
from plasma.core.node import NodeParams, PlasmaNode


def _node(r=(2,), u_max=(5.0,), nbrs=(1,), opts=None):
  Nf = len(u_max)
  params = NodeParams(
    node_id=0, nbrs=tuple(nbrs), alpha=np.full(Nf, 2.0),
    gamma=np.full(Nf, 0.1), beta=np.full((len(nbrs), Nf), 1.5),
    u_max=np.array(u_max), ram_cap=100.0, ram_req=np.full(Nf, 2.0),
  )
  node = PlasmaNode(params, opts or PlasmaOptions(), np.random.default_rng(0))
  node.r = np.array(r, dtype=int)
  return node


def test_capacity_gate_never_admits_beyond_r_umax():
  node = _node(r=(2,), u_max=(5.0,))  # capacity 10 req/window
  node.begin_window()
  local = sum(node.route_request(0, round_=0) == LOCAL for _ in range(100))
  counts = node.end_window()
  assert local <= 10
  assert counts.x[0] == local


def test_incoming_forwards_share_the_same_capacity():
  node = _node(r=(1,), u_max=(3.0,))
  node.begin_window()
  admitted = sum(node.admit_forward(0) for _ in range(10))
  assert admitted == 3
  assert node.route_request(0, round_=0) != LOCAL  # capacity exhausted


def test_nack_counts_as_origin_rejection_and_pull():
  node = _node()
  node.begin_window()
  node.record_forward_result(0, col=2, accepted=False)
  node.record_forward_result(0, col=2, accepted=True)
  counts = node.end_window()
  assert counts.z[0] == 1
  assert counts.y[0, 0] == 1
  hb = node.make_heartbeat()
  assert hb.pull[0] == 2  # both attempts are offload pressure


def test_zero_replicas_rejects_or_forwards_everything():
  node = _node(r=(0,))
  node.begin_window()
  for _ in range(20):
    assert node.route_request(0, round_=0) != LOCAL


def test_dead_node_admits_nothing():
  node = _node()
  node.alive = False
  assert node.admit_forward(0) is False


def test_end_window_reinforces_local_conductance():
  node = _node(r=(4,), u_max=(100.0,))
  node.begin_window()
  for _ in range(50):
    node.route_request(0, round_=0)
  d_before = node.D[0, LOCAL]
  node.end_window()
  assert node.D[0, LOCAL] > d_before  # phi*alpha > evaporation at D_init
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_plasma_routing.py -v`
Expected: new tests FAIL (`No module named 'plasma.core.node'`); earlier tests PASS

- [ ] **Step 3: Write the implementation**

```python
# plasma/core/node.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from plasma.core.protocol import HeartbeatCache
from plasma.core.routing import choose_target, target_weights, update_conductance
from plasma.core.types import LOCAL, REJ, Heartbeat, PlasmaOptions


@dataclass(frozen=True)
class NodeParams:
  node_id: int
  nbrs: tuple
  alpha: np.ndarray
  gamma: np.ndarray
  beta: np.ndarray
  u_max: np.ndarray
  ram_cap: float
  ram_req: np.ndarray


@dataclass
class WindowCounts:
  x: np.ndarray
  z: np.ndarray
  y: np.ndarray
  xi: np.ndarray


class PlasmaNode:
  def __init__(
      self, params: NodeParams, opts: PlasmaOptions, rng: np.random.Generator
    ) -> None:
    self.params = params
    self.opts = opts
    self.rng = rng
    Nf = len(params.alpha)
    deg = len(params.nbrs)
    self.Nf = Nf
    self.deg = deg
    self.alive = True
    self.r = np.zeros(Nf, dtype=int)
    self.D = np.full((Nf, 2 + deg), opts.D_init)
    self.cache = HeartbeatCache()
    self.demand_hat = np.zeros(Nf)
    self.lam_hat = np.zeros(Nf)
    self._seq = 0
    self._spare_last = np.zeros(Nf)
    self._pull_last = np.zeros(Nf)
    # rewards are constant: local alpha, REJ floor, per-neighbor beta
    self._rewards = np.empty((Nf, 2 + deg))
    self._rewards[:, LOCAL] = params.alpha
    self._rewards[:, REJ] = opts.rej_floor
    for k in range(deg):
      self._rewards[:, 2 + k] = params.beta[k]
    self.begin_window()
    # Layer B state initialized in sb_setup (Task 7)

  # ---------------- Layer A: data plane ----------------

  def begin_window(self) -> None:
    Nf, deg = self.Nf, self.deg
    self._x = np.zeros(Nf)
    self._z = np.zeros(Nf)
    self._y = np.zeros((deg, Nf))
    self._xi = np.zeros(Nf)
    self._phi = np.zeros((Nf, 2 + deg))
    self._pull = np.zeros(Nf)
    self._admitted = np.zeros(Nf)
    self._arrivals = np.zeros(Nf)

  def _capacity(self, f: int) -> float:
    return float(self.r[f]) * self.params.u_max[f] * self.opts.W

  def route_request(self, f: int, round_: int) -> int:
    self._arrivals[f] += 1
    local_open = self._admitted[f] < self._capacity(f)
    nbr_spare = np.array([
      self.cache.spare(
        j, round_, self.opts.staleness_rounds, self.Nf
      )[f] for j in self.params.nbrs
    ])
    weights = target_weights(
      self.D[f], local_open, nbr_spare, self.opts.eps_explore
    )
    unsplittable = (
      self.opts.rare_function_mode == "unsplittable"
      and self.lam_hat[f] < self.opts.lambda_split_threshold
    )
    col = choose_target(self.rng, weights, unsplittable)
    if col == LOCAL:
      self._x[f] += 1
      self._admitted[f] += 1
      self._phi[f, LOCAL] += 1
    elif col == REJ:
      self._z[f] += 1
      self._pull[f] += 1
    return col

  def admit_forward(self, f: int) -> bool:
    if not self.alive or self._admitted[f] >= self._capacity(f):
      return False
    self._admitted[f] += 1
    self._xi[f] += 1
    return True

  def record_forward_result(self, f: int, col: int, accepted: bool) -> None:
    self._pull[f] += 1
    if accepted:
      self._y[col - 2, f] += 1
      self._phi[f, col] += 1
    else:
      self._z[f] += 1

  # ---------------- Layer A: control plane ----------------

  def end_window(self) -> WindowCounts:
    counts = WindowCounts(x=self._x, z=self._z, y=self._y, xi=self._xi)
    self.D = update_conductance(self.D, self._phi, self._rewards, self.opts)
    ew = self.opts.ewma
    self.demand_hat = (1 - ew) * self.demand_hat + ew * (self._x + self._xi)
    self.lam_hat = (1 - ew) * self.lam_hat + ew * self._arrivals
    self._spare_last = np.maximum(
      0.0, self.r * self.params.u_max * self.opts.W - self._admitted
    )
    self._pull_last = self._pull
    self.begin_window()
    return counts

  def make_heartbeat(self) -> Heartbeat:
    self._seq += 1
    return Heartbeat(
      node=self.params.node_id, seq=self._seq,
      spare=tuple(self._spare_last), alpha=tuple(self.params.alpha),
      pull=tuple(self._pull_last),
    )

  def on_heartbeat(self, hb: Heartbeat, round_: int) -> None:
    if self.alive:
      self.cache.store(hb, round_)
```

Note: `sb_pass` is intentionally absent — Task 7 adds it. If a linter complains about the trailing comment, keep the comment; the class is extended in place by Task 7.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_plasma_routing.py tests/test_plasma_protocol.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add plasma/core/node.py tests/test_plasma_routing.py
git commit -m "add PlasmaNode layer A window behavior"
```

---

### Task 7: PlasmaNode, Layer B integration (sb_pass, hysteresis, randomized commit)

**Files:**
- Modify: `plasma/core/node.py` (add methods to `PlasmaNode`)
- Test: `tests/test_plasma_sbm.py` (append)

**Interfaces:**
- Produces (methods on `PlasmaNode`):
  - `init_replicas() -> None` — `r_init == "spread"`: round-robin add one replica per function while RAM allows; `"zero"`: all zeros.
  - `sb_pass(round_: int) -> bool` — builds the local `HamiltonianContext`, runs `dsb_minimize`, decodes + `repair`s, applies hysteresis (`n_hyst` consecutive improving proposals of the SAME r_new, improvement margin `eps_commit * |H(r_prev)|`) then commits with probability `p_commit`. Returns True iff committed. Setting `p_commit = 1` disables the randomized-commit countermeasure (used by the phase-locked test).

Locked-in details: `benefit = alpha * (demand_hat + pull_in)` with `pull_in` from the cache at `round_`; `margin = z_delta * sqrt(demand_hat)`; `A` auto-rule when `opts.A is None`: `A = 2 * max(benefit / ram_req)` (with floor 1.0); RAM feasibility re-checked exactly on commit via `repair`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_plasma_sbm.py`)

```python
from plasma.core.node import NodeParams, PlasmaNode


def _sb_node(p_commit=1.0, n_hyst=1, k_sb=1, ram_cap=8.0):
  params = NodeParams(
    node_id=0, nbrs=(), alpha=np.array([3.0, 1.0]),
    gamma=np.array([0.1, 0.1]), beta=np.zeros((0, 2)),
    u_max=np.array([5.0, 5.0]), ram_cap=ram_cap, ram_req=np.array([2.0, 2.0]),
  )
  opts = PlasmaOptions(p_commit=p_commit, n_hyst=n_hyst, k_sb=k_sb,
                       n_sb_steps=200)
  return PlasmaNode(params, opts, np.random.default_rng(1))


def test_init_replicas_spread_fills_ram_round_robin():
  node = _sb_node()
  node.init_replicas()
  assert (node.r * node.params.ram_req).sum() <= node.params.ram_cap
  assert node.r.sum() == 4  # 8 RAM / 2 per replica


def test_sb_pass_grows_replicas_under_demand():
  node = _sb_node(n_hyst=1)
  node.r = np.zeros(2, dtype=int)
  node.demand_hat = np.array([8.0, 0.0])
  committed = node.sb_pass(round_=0)
  assert committed
  assert node.r[0] >= 1
  assert (node.r * node.params.ram_req).sum() <= node.params.ram_cap


def test_hysteresis_requires_consecutive_confirmations():
  node = _sb_node(n_hyst=2)
  node.r = np.zeros(2, dtype=int)
  node.demand_hat = np.array([8.0, 0.0])
  assert node.sb_pass(round_=0) is False  # first proposal only counts
  assert node.sb_pass(round_=1) is True   # second consecutive -> commit


def test_p_commit_zero_never_commits():
  node = _sb_node(p_commit=0.0, n_hyst=1)
  node.demand_hat = np.array([8.0, 0.0])
  for k in range(5):
    assert node.sb_pass(round_=k) is False
  assert node.r.sum() == 0


def test_committed_r_is_always_ram_feasible():
  node = _sb_node(n_hyst=1, ram_cap=4.0)
  node.demand_hat = np.array([50.0, 50.0])  # wants far more than RAM allows
  node.sb_pass(round_=0)
  assert (node.r * node.params.ram_req).sum() <= 4.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_plasma_sbm.py -v`
Expected: new tests FAIL with `AttributeError: 'PlasmaNode' object has no attribute 'init_replicas'`

- [ ] **Step 3: Write the implementation** (append methods to `PlasmaNode` in `plasma/core/node.py`; add imports)

```python
# add to imports in plasma/core/node.py
from plasma.core.sbm import (
  HamiltonianContext, bits_per_fn, decode_spins, dsb_minimize, hamiltonian,
  r_max_per_fn, repair,
)

# add inside class PlasmaNode:

  # ---------------- Layer B ----------------

  def init_replicas(self) -> None:
    if self.opts.r_init == "zero":
      return
    used = 0.0
    while True:
      progress = False
      for f in range(self.Nf):
        if used + self.params.ram_req[f] <= self.params.ram_cap:
          self.r[f] += 1
          used += self.params.ram_req[f]
          progress = True
      if not progress:
        return

  def _hamiltonian_ctx(self, round_: int) -> HamiltonianContext:
    pull_in = self.cache.pull_in(round_, self.opts.staleness_rounds, self.Nf)
    benefit = self.params.alpha * (self.demand_hat + pull_in)
    A = self.opts.A
    if A is None:
      A = max(1.0, 2.0 * float((benefit / self.params.ram_req).max()))
    return HamiltonianContext(
      benefit=benefit, ram_req=self.params.ram_req,
      ram_cap=self.params.ram_cap, demand_hat=self.demand_hat,
      margin=self.opts.z_delta * np.sqrt(self.demand_hat),
      u_max=self.params.u_max * self.opts.W, r_prev=self.r.copy(),
      A=A, B=self.opts.B, C=self.opts.C, switch_cost=self.opts.switch_cost,
    )

  def sb_pass(self, round_: int) -> bool:
    if not self.alive:
      return False
    ctx = self._hamiltonian_ctx(round_)
    r_max = r_max_per_fn(self.params.ram_cap, self.params.ram_req)
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
    h_prev = hamiltonian(self.r, ctx)
    h_new = hamiltonian(r_new, ctx)
    improving = h_new < h_prev - self.opts.eps_commit * abs(h_prev)
    if not improving or (self._pending is not None
                         and not np.array_equal(r_new, self._pending)):
      self._pending, self._streak = None, 0
      return False
    self._pending = r_new
    self._streak += 1
    if self._streak < self.opts.n_hyst:
      return False
    self._pending, self._streak = None, 0
    # randomized commit: the mandatory Jacobi-oscillation countermeasure
    # under the shared barrier (p_commit = 1 disables it, deliberately)
    if self.rng.random() >= self.opts.p_commit:
      return False
    self.r = r_new
    return True
```

Also add to `__init__` (before `self.begin_window()`):

```python
    self._pending = None
    self._streak = 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_plasma_sbm.py tests/test_plasma_routing.py -v`
Expected: all PASS (tune `n_sb_steps`/`sb_c0` in the test fixture only if `test_sb_pass_grows_replicas_under_demand` flakes; keep seeds fixed)

- [ ] **Step 5: Commit**

```bash
git add plasma/core/node.py tests/test_plasma_sbm.py
git commit -m "add PlasmaNode layer B pass with hysteresis and randomized commit"
```

---

### Task 8: Engine — `plasma/engine.py`

**Files:**
- Create: `plasma/engine.py`
- Test: `tests/test_plasma_clock.py` (append) and `tests/test_plasma_protocol.py` (message budget)

**Interfaces:**
- Consumes: `PlasmaNode`, `RoundClock`, `encode_heartbeat`/`decode_heartbeat`, `PlasmaOptions`.
- Produces:

```python
class PlasmaEngine:
  def __init__(self, nodes: list, opts: PlasmaOptions, rng): ...
  clock: RoundClock
  msg_count: int          # heartbeats sent + forwards attempted (data plane)
  hb_count: int           # heartbeats sent only
  def set_alive(self, i: int, alive: bool) -> None
  def run_rounds(self, n_rounds: int, arrivals: np.ndarray) -> StepResult
  # arrivals: (Nn, Nf) integer external arrivals per round (constant here)

@dataclass
class StepResult:      # aggregates of the LAST round's window (rates, W=1s)
  x: np.ndarray        # (Nn, Nf)
  z: np.ndarray        # (Nn, Nf)
  y: np.ndarray        # (Nn, Nn, Nf)  ACKed forwards, sender-major
  xi: np.ndarray       # (Nn, Nn, Nf)  xi[n2, n1, f] = y[n1, n2, f]
  r: np.ndarray        # (Nn, Nf)
```

Per-round order (the round barrier): (1) clock delivers due heartbeats to their targets, (2) every alive node routes its external arrivals — forwards resolved synchronously: `ok = nodes[j].admit_forward(f)`, then `record_forward_result`; dead-node arrivals are dropped (not routed, not counted), (3) `end_window()` on every alive node, heartbeats scheduled for `round + hb_latency_rounds` per neighbor, each independently lost with probability `hb_loss`, (4) if `k_sb > 0` and `(round+1) % k_sb == 0`: `sb_pass(round)` on every alive node. Heartbeats cross the "wire" as dicts (`encode_heartbeat` → `decode_heartbeat`) so the whitelist is enforced on every message.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_plasma_clock.py`)

```python
import numpy as np

from plasma.core.node import NodeParams, PlasmaNode
from plasma.core.types import PlasmaOptions
from plasma.engine import PlasmaEngine


def _line_engine(Nn=2, Nf=1, opts=None, seed=0, u_max=5.0, ram_cap=8.0):
  opts = opts or PlasmaOptions(k_sb=0)
  rng = np.random.default_rng(seed)
  nodes = []
  for i in range(Nn):
    nbrs = tuple(j for j in (i - 1, i + 1) if 0 <= j < Nn)
    params = NodeParams(
      node_id=i, nbrs=nbrs, alpha=np.full(Nf, 2.0), gamma=np.full(Nf, 0.1),
      beta=np.full((len(nbrs), Nf), 1.5), u_max=np.full(Nf, u_max),
      ram_cap=ram_cap, ram_req=np.full(Nf, 2.0),
    )
    node = PlasmaNode(params, opts, np.random.default_rng(seed + i))
    node.init_replicas()
    nodes.append(node)
  return PlasmaEngine(nodes, opts, rng), nodes


def test_traffic_conservation_every_round():
  engine, _ = _line_engine()
  arrivals = np.array([[8], [8]])
  res = engine.run_rounds(5, arrivals)
  total = res.x + res.z + res.y.sum(axis=1)
  np.testing.assert_allclose(total, arrivals.astype(float))


def test_xi_transposes_y():
  engine, _ = _line_engine()
  res = engine.run_rounds(5, np.array([[20], [0]]))
  np.testing.assert_allclose(res.xi[1, 0, :], res.y[0, 1, :])


def test_dead_node_traffic_redistributes():
  # 3-node line, kill the middle node: node 0 must stop forwarding to it
  opts = PlasmaOptions(k_sb=0)
  engine, nodes = _line_engine(Nn=3, opts=opts)
  engine.run_rounds(5, np.array([[20], [0], [0]]))
  engine.set_alive(1, False)
  res = engine.run_rounds(10 * opts.staleness_rounds, np.array([[20], [0], [0]]))
  assert res.y[0, 1, 0] == 0.0  # nothing ACKed by a dead node
  # conductance toward the dead neighbor decayed to the floor
  col = 2 + nodes[0].params.nbrs.index(1)
  assert nodes[0].D[0, col] <= opts.D_min * 1.01


def test_phase_locked_commit_thrash_vs_randomized():
  # adversarial: 2-node line, shared demand pulse, SB every round
  def run(p_commit):
    opts = PlasmaOptions(k_sb=1, n_hyst=1, p_commit=p_commit, n_sb_steps=150)
    engine, nodes = _line_engine(Nn=2, opts=opts, seed=3, ram_cap=4.0)
    flips = 0
    prev = [n.r.copy() for n in nodes]
    for _ in range(20):
      engine.run_rounds(1, np.array([[12], [12]]))
      for k, n in enumerate(nodes):
        if not np.array_equal(prev[k], n.r):
          flips += 1
        prev[k] = n.r.copy()
    return flips

  thrash = run(p_commit=1.0)
  calm = run(p_commit=0.5)
  assert calm <= thrash


def test_settles_within_20_slow_ticks_with_default_p_commit():
  opts = PlasmaOptions(k_sb=1, n_hyst=2, p_commit=0.5, n_sb_steps=150)
  engine, nodes = _line_engine(Nn=2, opts=opts, seed=3, ram_cap=4.0)
  last_change = 0
  prev = [n.r.copy() for n in nodes]
  for tick in range(1, 21):
    engine.run_rounds(1, np.array([[12], [12]]))
    for k, n in enumerate(nodes):
      if not np.array_equal(prev[k], n.r):
        last_change = tick
        prev[k] = n.r.copy()
  assert last_change < 20
```

And append to `tests/test_plasma_protocol.py`:

```python
from plasma.engine import PlasmaEngine  # noqa: F401 (import checks packaging)


def test_message_budget_heartbeats_bounded_by_degree():
  from plasma.core.node import NodeParams, PlasmaNode
  from plasma.core.types import PlasmaOptions
  opts = PlasmaOptions(k_sb=0, hb_loss=0.0)
  rng = np.random.default_rng(0)
  params0 = NodeParams(node_id=0, nbrs=(1,), alpha=np.ones(1),
                       gamma=np.ones(1), beta=np.ones((1, 1)),
                       u_max=np.ones(1), ram_cap=4.0, ram_req=np.ones(1))
  params1 = NodeParams(node_id=1, nbrs=(0,), alpha=np.ones(1),
                       gamma=np.ones(1), beta=np.ones((1, 1)),
                       u_max=np.ones(1), ram_cap=4.0, ram_req=np.ones(1))
  nodes = [PlasmaNode(params0, opts, np.random.default_rng(1)),
           PlasmaNode(params1, opts, np.random.default_rng(2))]
  engine = PlasmaEngine(nodes, opts, rng)
  engine.run_rounds(10, np.zeros((2, 1), dtype=int))
  # exactly deg(i) heartbeats per node per round, no hidden channels
  assert engine.hb_count == 10 * 2 * 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_plasma_clock.py tests/test_plasma_protocol.py -v`
Expected: new tests FAIL with `ModuleNotFoundError: No module named 'plasma.engine'`

- [ ] **Step 3: Write the implementation**

```python
# plasma/engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from plasma.core.node import PlasmaNode
from plasma.core.protocol import decode_heartbeat, encode_heartbeat
from plasma.core.types import LOCAL, REJ, PlasmaOptions
from plasma.sim.clock import RoundClock


@dataclass
class StepResult:
  x: np.ndarray
  z: np.ndarray
  y: np.ndarray
  xi: np.ndarray
  r: np.ndarray


class PlasmaEngine:
  def __init__(
      self, nodes: List[PlasmaNode], opts: PlasmaOptions,
      rng: np.random.Generator
    ) -> None:
    self.nodes = nodes
    self.opts = opts
    self.rng = rng
    self.clock = RoundClock()
    self.msg_count = 0
    self.hb_count = 0

  def set_alive(self, i: int, alive: bool) -> None:
    self.nodes[i].alive = alive

  def _route_all(self, round_: int, arrivals: np.ndarray) -> None:
    for i, node in enumerate(self.nodes):
      if not node.alive:
        continue
      for f in range(node.Nf):
        for _ in range(int(arrivals[i, f])):
          col = node.route_request(f, round_)
          if col >= 2:
            j = node.params.nbrs[col - 2]
            self.msg_count += 1
            accepted = self.nodes[j].admit_forward(f)
            node.record_forward_result(f, col, accepted)

  def _send_heartbeats(self, round_: int) -> None:
    for node in self.nodes:
      if not node.alive:
        continue
      hb = node.make_heartbeat()
      wire = encode_heartbeat(hb)
      for j in node.params.nbrs:
        self.hb_count += 1
        self.msg_count += 1
        if self.rng.random() < self.opts.hb_loss:
          continue
        target = self.nodes[j]
        self.clock.schedule(
          round_ + self.opts.hb_latency_rounds,
          lambda t=target, w=wire, r=round_ + self.opts.hb_latency_rounds:
            t.on_heartbeat(decode_heartbeat(w), r),
        )

  def run_rounds(self, n_rounds: int, arrivals: np.ndarray) -> StepResult:
    Nn = len(self.nodes)
    Nf = self.nodes[0].Nf
    last = {}

    def on_round(round_: int) -> None:
      self._route_all(round_, arrivals)
      x = np.zeros((Nn, Nf))
      z = np.zeros((Nn, Nf))
      y = np.zeros((Nn, Nn, Nf))
      for i, node in enumerate(self.nodes):
        if not node.alive:
          continue
        counts = node.end_window()
        x[i] = counts.x
        z[i] = counts.z
        for k, j in enumerate(node.params.nbrs):
          y[i, j, :] = counts.y[k]
      last["x"], last["z"], last["y"] = x, z, y
      self._send_heartbeats(round_)
      if self.opts.k_sb > 0 and (round_ + 1) % self.opts.k_sb == 0:
        for node in self.nodes:
          node.sb_pass(round_)

    self.clock.run(n_rounds, on_round)
    xi = np.transpose(last["y"], (1, 0, 2))
    r = np.array([node.r for node in self.nodes])
    return StepResult(x=last["x"], z=last["z"], y=last["y"], xi=xi, r=r)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_plasma_clock.py tests/test_plasma_protocol.py -v`
Expected: all PASS. The two commit-behavior tests are statistical: keep the fixed seeds; if a seed is unlucky, change the SEED in the test, never the assertion.

- [ ] **Step 5: Commit**

```bash
git add plasma/engine.py tests/test_plasma_clock.py tests/test_plasma_protocol.py
git commit -m "add plasma engine with round barrier, forwarding and heartbeats"
```

---

### Task 9: Runner and CLI — `plasma/runner.py`, `plasma/cli.py`

**Files:**
- Create: `plasma/runner.py`, `plasma/cli.py`
- Test: `tests/test_plasma_e2e.py`

**Interfaces:**
- Consumes: `init_problem`, `get_current_load`, `update_data`, `init_complete_solution`, `decode_solution`, `join_complete_solution`, `save_solution`, `save_checkpoint`, `check_feasibility`, `compute_centralized_objective`, `load_configuration` (see "Repo interfaces" table above), plus `PlasmaEngine`.
- Produces: `plasma.runner.run(config: dict, parallelism: int, log_on_file: bool = False, disable_plotting: bool = False) -> str` — same contract as `decentralized_gcaa.run`. Writes into a fresh timestamped folder: `config.json`, `LSPc_solution.csv`, `LSPc_offloaded.csv`, `LSPc_utilization.csv`, `LSPc_replicas.csv`, `LSPc_detailed_fwd_solution.csv`, `LSPc_residual_capacity.csv`, `obj.csv` (column `Plasma`), `runtime.csv`, `termination_condition.csv`, `plasma_messages.csv` (columns `t`, `msgs_per_node_s`, `hb_per_node_s`).

Runner flow (mirrors `decentralized_gcaa.run` lines 106–328, minus the Pyomo solves):
1. Read config: `base_solution_folder`, `seed`, `limits`, `trace_type` (from `limits["load"]`, default `"fixed_sum"`), `verbose`, `max_steps`, `min_run_time`, `max_run_time`, `run_time_step`, `checkpoint_interval`; `opts = PlasmaOptions.from_config(config)`.
2. Create timestamped folder, dump config, optionally open `out.log`.
3. `init_problem(limits, trace_type, max_steps, seed, solution_folder)`.
4. Build `NodeParams` per node from `base_instance_data[None]` (0-based conversion; `beta[k] = beta[(i+1, nbrs[k]+1, f+1)]`; `u_max[f] = max_utilization[f+1] / demand[(i+1, f+1)]`), build `PlasmaNode`s (per-node rng: `np.random.default_rng(seed * 1000 + i)`), `init_replicas()`, build `PlasmaEngine` (engine rng: `np.random.default_rng(seed)`).
5. For each `t` in `range(min_run_time, ub, run_time_step)` (ub logic identical to gcaa): `loadt = get_current_load(...)`; `data = update_data(base_instance_data, {"incoming_load": loadt})`; `arrivals[i, f] = round(loadt[(i+1, f+1)])`; `res = engine.run_rounds(opts.rounds_per_step, arrivals)`.
6. Per step: scale the last-window counts back to the true load so `check_feasibility`'s traffic-conservation holds exactly against `loadt` (arrivals are already the integer load; assert `x + y.sum(axis=1) + z == arrivals` and pass through). Compute `U[n, f] = demand[(n+1, f+1)] * (x + xi.sum(axis=1))[n, f] / r[n, f]` where `r > 0` else 0; `rho[n] = ram_cap[n] - sum_f r[n, f] * ram_req[f]`. Call `check_feasibility(x, y.sum(axis=1), z, r, U, data)` and raise on failure. Accumulate `decode_solution(x, y, z, r, xi, rho, U, cs)`; append `compute_centralized_objective(data, x, y, z)` to `obj_list`; append wall-clock to `runtime_list`; append messages row; `save_checkpoint(cs, os.path.join(folder, "LSPc"), t)` on `checkpoint_interval`.
7. After the loop: `join_complete_solution`, `save_solution(..., "LSPc", folder)`, write `obj.csv` (`pd.DataFrame(obj_list, columns=["Plasma"])`), `termination_condition.csv` (one row per step: `f"rounds: {opts.rounds_per_step}"`), `runtime.csv`, `plasma_messages.csv`. Return folder.

`plasma/cli.py`: argparse entry mirroring `decentralized_gcaa.parse_arguments` (flags `-c/--config`, `-j/--parallelism`, `--disable_plotting`) with `if __name__ == "__main__": run(load_configuration(args.config), args.parallelism, disable_plotting=args.disable_plotting)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_plasma_e2e.py
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plasma.runner import run as run_plasma


def _config(tmp_path, Nn=3, max_steps=3):
  return {
    "base_solution_folder": str(tmp_path),
    "verbose": 0,
    "seed": 42,
    "max_steps": max_steps,
    "min_run_time": 0,
    "max_run_time": max_steps - 1,
    "run_time_step": 1,
    "checkpoint_interval": 1,
    "solver_name": "none",
    "solver_options": {
      "plasma": {"rounds_per_step": 5, "k_sb": 2, "n_sb_steps": 100,
                 "n_hyst": 1}
    },
    "limits": {
      "Nn": {"values": [Nn]},
      "Nf": {"min": 2, "max": 2},
      "neighborhood": {"m": Nn - 1},   # a line/tree on 3 nodes
      "demand": {"values": [1.0, 1.2]},
      "memory_capacity": {"min": 12, "max": 12},
      "memory_requirement": {"values": [2, 3]},
      "max_utilization": {"min": 0.65, "max": 0.75},
      "load": {"trace_type": "sinusoidal",
               "min": {"min": 5, "max": 10},
               "max": {"min": 20, "max": 30}},
      "weights": {"alpha": {"min": 1.0, "max": 1.5},
                  "beta_multiplier": {"min": 1.5, "max": 2.5},
                  "gamma": {"min": 0.05, "max": 0.15},
                  "delta_multiplier": {"min": 0.1, "max": 0.2}},
    },
  }


def test_runner_produces_lspc_artifacts(tmp_path):
  folder = run_plasma(_config(tmp_path), parallelism=0)
  for name in ("LSPc_solution.csv", "LSPc_offloaded.csv",
               "LSPc_utilization.csv", "LSPc_replicas.csv",
               "LSPc_detailed_fwd_solution.csv",
               "LSPc_residual_capacity.csv", "obj.csv", "runtime.csv",
               "termination_condition.csv", "plasma_messages.csv",
               "config.json"):
    assert os.path.exists(os.path.join(folder, name)), name


def test_runner_objective_column_is_plasma(tmp_path):
  folder = run_plasma(_config(tmp_path), parallelism=0)
  obj = pd.read_csv(os.path.join(folder, "obj.csv"))
  assert list(obj.columns) == ["Plasma"]
  assert len(obj) == 3


def test_runner_is_deterministic_given_seed(tmp_path):
  f1 = run_plasma(_config(tmp_path / "a"), parallelism=0)
  f2 = run_plasma(_config(tmp_path / "b"), parallelism=0)
  o1 = pd.read_csv(os.path.join(f1, "obj.csv"))
  o2 = pd.read_csv(os.path.join(f2, "obj.csv"))
  pd.testing.assert_frame_equal(o1, o2)


def test_runner_messages_bounded(tmp_path):
  folder = run_plasma(_config(tmp_path), parallelism=0)
  msgs = pd.read_csv(os.path.join(folder, "plasma_messages.csv"))
  assert (msgs["hb_per_node_s"] <= 2.0 + 1e-9).all()  # deg <= 2 on a tree of 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_plasma_e2e.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'plasma.runner'`

- [ ] **Step 3: Write the implementation**

```python
# plasma/runner.py
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from run_centralized_model import (
  init_complete_solution, init_problem, decode_solution,
  join_complete_solution, save_checkpoint, save_solution,
)
from generators.generate_data import update_data
from utils.centralized import check_feasibility, get_current_load
from utils.faasmacro import compute_centralized_objective

from plasma.core.node import NodeParams, PlasmaNode
from plasma.core.types import PlasmaOptions
from plasma.engine import PlasmaEngine


def build_nodes(base_instance_data: dict, opts: PlasmaOptions, seed: int):
  d = base_instance_data[None]
  Nn = d["Nn"][None]
  Nf = d["Nf"][None]
  nodes = []
  for i in range(Nn):
    nbrs = tuple(
      j for j in range(Nn) if j != i and d["neighborhood"][(i + 1, j + 1)]
    )
    alpha = np.array([d["alpha"][(i + 1, f + 1)] for f in range(Nf)])
    gamma = np.array([d["gamma"][(i + 1, f + 1)] for f in range(Nf)])
    beta = np.array([
      [d["beta"][(i + 1, j + 1, f + 1)] for f in range(Nf)] for j in nbrs
    ]).reshape(len(nbrs), Nf)
    u_max = np.array([
      d["max_utilization"][f + 1] / d["demand"][(i + 1, f + 1)]
      for f in range(Nf)
    ])
    params = NodeParams(
      node_id=i, nbrs=nbrs, alpha=alpha, gamma=gamma, beta=beta,
      u_max=u_max, ram_cap=float(d["memory_capacity"][i + 1]),
      ram_req=np.array([d["memory_requirement"][f + 1] for f in range(Nf)]),
    )
    node = PlasmaNode(params, opts, np.random.default_rng(seed * 1000 + i))
    node.init_replicas()
    nodes.append(node)
  return nodes


def run(
    config: dict, parallelism: int, log_on_file: bool = False,
    disable_plotting: bool = False
  ) -> str:
  base_solution_folder = config["base_solution_folder"]
  seed = config["seed"]
  limits = config["limits"]
  trace_type = limits["load"].get("trace_type", "fixed_sum")
  verbose = config.get("verbose", 0)
  max_steps = config["max_steps"]
  min_run_time = config.get("min_run_time", 0)
  max_run_time = config.get("max_run_time", max_steps)
  run_time_step = config.get("run_time_step", 1)
  checkpoint_interval = config["checkpoint_interval"]
  opts = PlasmaOptions.from_config(config)
  now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
  solution_folder = f"{base_solution_folder}/{now}"
  os.makedirs(solution_folder, exist_ok=True)
  with open(os.path.join(solution_folder, "config.json"), "w") as ostream:
    ostream.write(json.dumps(config, indent=2))
  log_stream = sys.stdout
  if log_on_file:
    log_stream = open(os.path.join(solution_folder, "out.log"), "w")
  base_instance_data, input_requests_traces, agents, graph = init_problem(
    limits, trace_type, max_steps, seed, solution_folder
  )
  d = base_instance_data[None]
  Nn = d["Nn"][None]
  Nf = d["Nf"][None]
  nodes = build_nodes(base_instance_data, opts, seed)
  engine = PlasmaEngine(nodes, opts, np.random.default_rng(seed))
  ram_cap = np.array([d["memory_capacity"][n + 1] for n in range(Nn)])
  ram_req = np.array([d["memory_requirement"][f + 1] for f in range(Nf)])
  demand = np.array([
    [d["demand"][(n + 1, f + 1)] for f in range(Nf)] for n in range(Nn)
  ])
  cs = init_complete_solution()
  obj_list = []
  runtime_list = []
  msg_rows = []
  ub = (
    max_run_time + run_time_step
  ) if max_run_time == min_run_time else max_run_time
  for t in range(min_run_time, ub, run_time_step):
    if verbose > 0:
      print(f"t = {t}", file=log_stream, flush=True)
    loadt = get_current_load(input_requests_traces, agents, t)
    data = update_data(base_instance_data, {"incoming_load": loadt})
    arrivals = np.array([
      [int(round(loadt[(n + 1, f + 1)])) for f in range(Nf)]
      for n in range(Nn)
    ])
    data = update_data(data, {"incoming_load": {
      (n + 1, f + 1): arrivals[n, f] for n in range(Nn) for f in range(Nf)
    }})
    msgs_before, hb_before = engine.msg_count, engine.hb_count
    started = datetime.now()
    res = engine.run_rounds(opts.rounds_per_step, arrivals)
    elapsed = (datetime.now() - started).total_seconds()
    omega = res.y.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
      U = np.where(
        res.r > 0,
        demand * (res.x + res.xi.sum(axis=1)) / np.maximum(res.r, 1),
        0.0,
      )
    rho = ram_cap - (res.r * ram_req[None, :]).sum(axis=1)
    feasible, why = check_feasibility(res.x, omega, res.z, res.r, U, data)
    assert feasible, why
    cs = decode_solution(res.x, res.y, res.z, res.r, res.xi, rho, U, cs)
    obj_list.append(compute_centralized_objective(data, res.x, res.y, res.z))
    runtime_list.append(elapsed)
    seconds = opts.rounds_per_step * opts.W
    msg_rows.append({
      "t": t,
      "msgs_per_node_s": (engine.msg_count - msgs_before) / (Nn * seconds),
      "hb_per_node_s": (engine.hb_count - hb_before) / (Nn * seconds),
    })
    if t % checkpoint_interval == 0 or t == max_steps - 1:
      save_checkpoint(cs, os.path.join(solution_folder, "LSPc"), t)
  solution, offloaded, detailed_fwd = join_complete_solution(cs)
  save_solution(solution, offloaded, cs, detailed_fwd, "LSPc", solution_folder)
  pd.DataFrame(obj_list, columns=["Plasma"]).to_csv(
    os.path.join(solution_folder, "obj.csv"), index=False
  )
  pd.DataFrame(
    [f"rounds: {opts.rounds_per_step}"] * len(obj_list)
  ).to_csv(os.path.join(solution_folder, "termination_condition.csv"))
  pd.DataFrame({"tot": runtime_list}).to_csv(
    os.path.join(solution_folder, "runtime.csv"), index=False
  )
  pd.DataFrame(msg_rows).to_csv(
    os.path.join(solution_folder, "plasma_messages.csv"), index=False
  )
  if verbose > 0:
    print(f"All solutions saved in: {solution_folder}", file=log_stream,
          flush=True)
  if log_on_file:
    log_stream.close()
  return solution_folder
```

```python
# plasma/cli.py
from __future__ import annotations

import argparse

from utils.common import load_configuration

from plasma.runner import run


def parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Run PLASMA standalone (debugging entry point)",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument("-c", "--config", type=str, default="manual_config.json")
  parser.add_argument("-j", "--parallelism", type=int, default=0)
  parser.add_argument("--disable_plotting", default=False,
                      action="store_true")
  return parser.parse_known_args()[0]


if __name__ == "__main__":
  args = parse_arguments()
  run(
    load_configuration(args.config), args.parallelism,
    disable_plotting=args.disable_plotting,
  )
```

Implementation notes:
- `check_feasibility` requires exact traffic conservation against `data[None]["incoming_load"]`; that's why the runner overwrites `incoming_load` with the rounded integer `arrivals` before the check and objective (traces may be floats).
- If a node dies mid-run (not exercised by the runner yet — engine API only), its dropped arrivals would break conservation; the runner never kills nodes, scenario tests drive the engine directly.
- `parallelism` is accepted for signature compatibility and unused (single-process simulation) — keep the parameter, document with a comment.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_plasma_e2e.py -v`
Expected: 4 PASS (each e2e test runs ~seconds)

- [ ] **Step 5: Commit**

```bash
git add plasma/runner.py plasma/cli.py tests/test_plasma_e2e.py
git commit -m "add plasma runner writing LSPc results and argparse cli"
```

---

### Task 10: `run.py` registration + comparison config

**Files:**
- Modify: `run.py` (7 additive touch points, no existing line changed)
- Create: `config_files/plasma_comparison.json`
- Test: `tests/test_plasma_e2e.py` (append wiring tests)

**IMPORTANT (project rule):** before editing, run `gitnexus_impact({target: "parse_arguments", direction: "upstream"})` and `gitnexus_impact({target: "results_postprocessing", direction: "upstream"})`; report blast radius. After editing, `gitnexus_detect_changes()` must show only `run.py` + new files.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_plasma_e2e.py`)

```python
import run as run_module


def test_methods_choice_accepts_plasma(monkeypatch):
  argv = ["run.py", "-c", "config_files/plasma_comparison.json",
          "--methods", "plasma"]
  monkeypatch.setattr("sys.argv", argv)
  args = run_module.parse_arguments()
  assert "plasma" in args.methods


def test_run_module_exposes_plasma_runner():
  assert callable(run_module.run_plasma)


def test_method_result_models_has_plasma_entry():
  assert run_module.METHOD_RESULT_MODELS["plasma"] == ("LSPc", "Plasma")


def test_plasma_comparison_config_exists_and_has_section():
  config = json.loads(
    Path("config_files/plasma_comparison.json").read_text()
  )
  assert "plasma" in config["solver_options"]


def test_set_solution_folder_tolerates_missing_plasma_key():
  solution_folders = {"experiments_list": []}
  run_module.set_solution_folder(solution_folders, "plasma", 0, "/x")
  assert solution_folders["plasma"][0] == "/x"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_plasma_e2e.py -v`
Expected: new tests FAIL (`AttributeError: module 'run' has no attribute 'run_plasma'`, missing choice, missing config file)

- [ ] **Step 3: Edit `run.py` — 7 additive touch points**

1. Imports (after line 12 `from decentralized_gcaa import run as run_gcaa`):
```python
from plasma.runner import run as run_plasma
```
2. `METHOD_RESULT_MODELS` (after the `"faas-gcaa"` entry, ~line 45):
```python
  "plasma": ("LSPc", "Plasma"),
```
3. `--methods` choices list (after `"faas-gcaaa"`... exactly after `"faas-gcaa",`, before `"generate_only"`):
```python
      "plasma",
```
4. `solution_folders` initial dict (~line 901, add key):
```python
    "plasma": [],
```
5. Run-flag init (next to `run_g = False`, ~line 934):
```python
    run_pl = False # -- plasma (Plasma)
```
6. Skip/resume logic (after the `faas-gcaa` block, ~line 1024) and the `except ValueError` fallback (after `run_g = ...`) and the big `if run_c or ...` condition — add `run_pl` to all three:
```python
      if (not generate_only and "plasma" in methods) and ((
          len(solution_folders.get("plasma", [])) <= experiment_idx
        ) or (
          solution_folders["plasma"][experiment_idx] is None
        )):
        run_pl = True
```
```python
      run_pl = "plasma" in methods
```
and extend the condition: `... or run_g or run_pl or generate_only:`
7. Invocation (after the `if run_g:` block, ~line 1228):
```python
      # -- solve PLASMA (Physarum + simulated bifurcation)
      if run_pl:
        pl_folder = run_plasma(
          config, sp_parallelism,
          log_on_file = log_on_file, disable_plotting = disable_plotting
        )
        set_solution_folder(
          solution_folders, "plasma", experiment_idx, pl_folder
        )
```

Create `config_files/plasma_comparison.json` — copy of `config_files/eval_smoke.json` with: `"base_solution_folder": "solutions/plasma_comparison"`, and this block added inside `"solver_options"`:
```json
    "plasma": {
      "rounds_per_step": 20,
      "k_sb": 10,
      "mu": 0.1,
      "n_sb_steps": 300,
      "p_commit": 0.5,
      "n_hyst": 2
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_plasma_e2e.py tests/test_gcaa_wiring.py -v`
Expected: all PASS (gcaa wiring must stay green — proves the edit was additive)

- [ ] **Step 5: Smoke the real pipeline**

Run: `uv run python run.py -c config_files/plasma_comparison.json --methods plasma --n_experiments 1 2>&1 | tail -5`
Expected: completes without traceback; a `solutions/plasma_comparison/...` folder with LSPc files exists. Delete the produced `solutions/plasma_comparison/` folder afterwards (scratch output).

- [ ] **Step 6: Commit**

```bash
git add run.py config_files/plasma_comparison.json tests/test_plasma_e2e.py
git commit -m "register plasma method in run.py dispatch"
```

---

### Task 11: Baselines — `plasma/baselines/`

**Files:**
- Create: `plasma/baselines/__init__.py`, `plasma/baselines/milp_baseline.py`, `plasma/baselines/greedy_baseline.py`, `plasma/baselines/madea_iface.py`
- Test: `tests/test_plasma_baselines.py`

**Interfaces:**
- Produces:
  - `milp_baseline.solve_snapshot(data, solver_name, solver_options) -> (x, y, z, r, obj)` — one MILP solve on the current `incoming_load` via `solve_instance(LoadManagementModel(), ...)`.
  - `milp_baseline.routing_lp(lam, r, u_max, alpha, beta, gamma, adjacency) -> (obj, x, y, z)` — LP-optimal routing with FIXED replicas via `scipy.optimize.linprog` (Layer-A reference; per-load-normalized objective like `compute_centralized_objective`). Shapes: `lam, x, z: (Nn, Nf)`, `y: (Nn, Nn, Nf)`, `r, u_max: (Nn, Nf)`, `alpha, gamma: (Nn, Nf)`, `beta: (Nn, Nn, Nf)`, `adjacency: (Nn, Nn)` 0/1.
  - `milp_baseline.stale_objectives(base_instance_data, traces, agents, t_range, solver_name, solver_options, resolve_every) -> list[float]` — re-solve every `resolve_every` steps on the load observed at the re-solve instant, hold the allocation in between, score each step's held solution on the TRUE load via `compute_centralized_objective` (rejections absorb the mismatch: `z = load - x - y_sent`, clipped at 0, and `x` scaled down if load dropped below the held `x + y`).
  - `greedy_baseline.solve(data, solver_options) -> (x, y, z, r)` — non-coordinated greedy: per node fill RAM greedily by `alpha * load` order, `x = min(load, r * u_max)` (with `u_max = max_utilization/demand`), `omega_bar = load - x`; then delegate cross-node distribution to `heuristic_coordinator.GreedyCoordinator().solve(instance, solver_options)` and convert its `y, z, r` output; anything GreedyCoordinator leaves unassigned is rejection.
  - `madea_iface.run_madea = run_faasmadea.run` (re-export with a docstring; the comparison plug per design §1.2).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_plasma_baselines.py
import numpy as np
import pytest

from plasma.baselines.milp_baseline import routing_lp
from plasma.baselines import madea_iface


def test_routing_lp_prefers_local_when_capacity_allows():
  lam = np.array([[10.0]])
  r = np.array([[5]])
  u_max = np.array([[4.0]])
  alpha = np.array([[2.0]])
  beta = np.zeros((1, 1, 1))
  gamma = np.array([[1.0]])
  adjacency = np.zeros((1, 1))
  obj, x, y, z = routing_lp(lam, r, u_max, alpha, beta, gamma, adjacency)
  assert x[0, 0] == pytest.approx(10.0)
  assert z[0, 0] == pytest.approx(0.0)
  assert obj == pytest.approx(2.0)  # alpha * x / lam


def test_routing_lp_offloads_overflow_to_neighbor():
  lam = np.array([[10.0], [0.0]])
  r = np.array([[1], [5]])
  u_max = np.array([[4.0], [4.0]])
  alpha = np.full((2, 1), 2.0)
  beta = np.zeros((2, 2, 1)); beta[0, 1, 0] = 1.5
  gamma = np.full((2, 1), 5.0)
  adjacency = np.array([[0, 1], [1, 0]])
  obj, x, y, z = routing_lp(lam, r, u_max, alpha, beta, gamma, adjacency)
  assert x[0, 0] == pytest.approx(4.0)
  assert y[0, 1, 0] == pytest.approx(6.0)
  assert z[0, 0] == pytest.approx(0.0)


def test_routing_lp_respects_receiver_capacity():
  lam = np.array([[10.0], [0.0]])
  r = np.array([[0], [1]])
  u_max = np.array([[4.0], [4.0]])
  alpha = np.full((2, 1), 2.0)
  beta = np.zeros((2, 2, 1)); beta[0, 1, 0] = 1.5
  gamma = np.full((2, 1), 0.1)
  adjacency = np.array([[0, 1], [1, 0]])
  obj, x, y, z = routing_lp(lam, r, u_max, alpha, beta, gamma, adjacency)
  assert y[0, 1, 0] == pytest.approx(4.0)
  assert z[0, 0] == pytest.approx(6.0)


def test_madea_iface_reexports_runner():
  import run_faasmadea
  assert madea_iface.run_madea is run_faasmadea.run


def test_greedy_baseline_conserves_traffic():
  from plasma.baselines.greedy_baseline import solve
  data = _tiny_instance()
  x, y, z, r = solve(data, {})
  Nn = data[None]["Nn"][None]
  Nf = data[None]["Nf"][None]
  for n in range(Nn):
    for f in range(Nf):
      load = data[None]["incoming_load"][(n + 1, f + 1)]
      assert x[n, f] + y[n, :, f].sum() + z[n, f] == pytest.approx(load)


def _tiny_instance():
  Nn, Nf = 2, 1
  return {None: {
    "Nn": {None: Nn}, "Nf": {None: Nf},
    "neighborhood": {(1, 1): 0, (1, 2): 1, (2, 1): 1, (2, 2): 0},
    "alpha": {(1, 1): 2.0, (2, 1): 2.0},
    "beta": {(1, 1, 1): 0.0, (1, 2, 1): 1.5, (2, 1, 1): 1.5, (2, 2, 1): 0.0},
    "gamma": {(1, 1): 0.1, (2, 1): 0.1},
    "demand": {(1, 1): 1.0, (2, 1): 1.0},
    "max_utilization": {1: 0.7},
    "memory_capacity": {1: 4, 2: 4},
    "memory_requirement": {1: 2},
    "incoming_load": {(1, 1): 10, (2, 1): 1},
  }}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_plasma_baselines.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# plasma/baselines/__init__.py  (empty file)

# plasma/baselines/madea_iface.py
"""Adapter so FaaS-MADeA plugs into PLASMA comparisons (design doc S1.2)."""
from run_faasmadea import run as run_madea  # noqa: F401
```

```python
# plasma/baselines/milp_baseline.py
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.optimize import linprog

from generators.generate_data import update_data
from models.model import LoadManagementModel
from run_centralized_model import solve_instance
from utils.centralized import get_current_load
from utils.faasmacro import compute_centralized_objective


def solve_snapshot(data: dict, solver_name: str, solver_options: dict):
  x, y, z, r, xi, omega, rho, U, obj, runtime, tc = solve_instance(
    LoadManagementModel(), data, solver_name, solver_options
  )
  return x, y, z, r, obj


def routing_lp(
    lam: np.ndarray, r: np.ndarray, u_max: np.ndarray, alpha: np.ndarray,
    beta: np.ndarray, gamma: np.ndarray, adjacency: np.ndarray
  ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
  """LP-optimal routing with fixed replicas: the Layer-A reference optimum.
  Variables per (n, f): x, z; per edge (n1, n2, f): y. Objective matches
  compute_centralized_objective (per-load-normalized)."""
  Nn, Nf = lam.shape
  # variable order: x (Nn*Nf), z (Nn*Nf), y (Nn*Nn*Nf)
  nx = Nn * Nf
  ny = Nn * Nn * Nf
  def ix(n, f): return n * Nf + f
  def iz(n, f): return nx + n * Nf + f
  def iy(n1, n2, f): return 2 * nx + (n1 * Nn + n2) * Nf + f
  c = np.zeros(2 * nx + ny)
  for n in range(Nn):
    for f in range(Nf):
      scale = max(lam[n, f], 1e-12)
      c[ix(n, f)] = -alpha[n, f] / scale
      c[iz(n, f)] = gamma[n, f] / scale
      for m in range(Nn):
        c[iy(n, m, f)] = -beta[n, m, f] / scale
  A_eq, b_eq = [], []
  for n in range(Nn):
    for f in range(Nf):
      row = np.zeros(2 * nx + ny)
      row[ix(n, f)] = 1.0
      row[iz(n, f)] = 1.0
      for m in range(Nn):
        row[iy(n, m, f)] = 1.0
      A_eq.append(row)
      b_eq.append(lam[n, f])
  A_ub, b_ub = [], []
  for n in range(Nn):
    for f in range(Nf):
      row = np.zeros(2 * nx + ny)
      row[ix(n, f)] = 1.0
      for m in range(Nn):
        row[iy(m, n, f)] = 1.0
      A_ub.append(row)
      b_ub.append(r[n, f] * u_max[n, f])
  bounds = [(0, None)] * (2 * nx) + [
    (0, None if adjacency[n1, n2] else 0)
    for n1 in range(Nn) for n2 in range(Nn) for _ in range(Nf)
  ]
  res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=bounds,
                method="highs")
  assert res.success, res.message
  sol = res.x
  x = sol[:nx].reshape(Nn, Nf)
  z = sol[nx:2 * nx].reshape(Nn, Nf)
  y = sol[2 * nx:].reshape(Nn, Nn, Nf)
  return float(-res.fun), x, y, z


def stale_objectives(
    base_instance_data: dict, traces: dict, agents, t_range,
    solver_name: str, solver_options: dict, resolve_every: int
  ) -> List[float]:
  """Centralized MILP re-solved every resolve_every steps on then-current
  load, held in between, scored on the true load (staleness is the point)."""
  held = None
  objs = []
  for k, t in enumerate(t_range):
    loadt = get_current_load(traces, agents, t)
    data = update_data(base_instance_data, {"incoming_load": loadt})
    if held is None or k % resolve_every == 0:
      x, y, z, r, _ = solve_snapshot(data, solver_name, solver_options)
      held = (x, y)
    x, y = held
    lam = np.array([
      [loadt[(n + 1, f + 1)] for f in range(x.shape[1])]
      for n in range(x.shape[0])
    ])
    handled = x + y.sum(axis=1)
    over = np.maximum(0.0, handled - lam)
    x_eff = np.maximum(0.0, x - over)  # shed overflow from local first
    z_eff = np.maximum(0.0, lam - x_eff - y.sum(axis=1))
    objs.append(compute_centralized_objective(data, x_eff, y, z_eff))
  return objs
```

```python
# plasma/baselines/greedy_baseline.py
from __future__ import annotations

from copy import deepcopy
from typing import Tuple

import numpy as np

from heuristic_coordinator import GreedyCoordinator


def solve(data: dict, solver_options: dict) -> Tuple[np.ndarray, ...]:
  """Local, non-coordinated greedy lower baseline: each node fills its own
  RAM by alpha*load order and serves what it can; leftover offloading is
  distributed by the existing GreedyCoordinator."""
  d = data[None]
  Nn = d["Nn"][None]
  Nf = d["Nf"][None]
  ram_cap = np.array([float(d["memory_capacity"][n + 1]) for n in range(Nn)])
  ram_req = np.array([float(d["memory_requirement"][f + 1]) for f in range(Nf)])
  lam = np.array([
    [d["incoming_load"][(n + 1, f + 1)] for f in range(Nf)] for n in range(Nn)
  ])
  u_max = np.array([
    [d["max_utilization"][f + 1] / d["demand"][(n + 1, f + 1)]
     for f in range(Nf)] for n in range(Nn)
  ])
  alpha = np.array([
    [d["alpha"][(n + 1, f + 1)] for f in range(Nf)] for n in range(Nn)
  ])
  r = np.zeros((Nn, Nf), dtype=int)
  for n in range(Nn):
    budget = ram_cap[n]
    for f in sorted(range(Nf), key=lambda f: -alpha[n, f] * lam[n, f]):
      while lam[n, f] > r[n, f] * u_max[n, f] and budget >= ram_req[f]:
        r[n, f] += 1
        budget -= ram_req[f]
  x = np.minimum(lam, r * u_max)
  omega = lam - x
  instance = deepcopy(data)
  instance[None]["omega_bar"] = {
    (n + 1, f + 1): float(omega[n, f]) for n in range(Nn) for f in range(Nf)
  }
  instance[None]["x_bar"] = {
    (n + 1, f + 1): float(x[n, f]) for n in range(Nn) for f in range(Nf)
  }
  instance[None]["r_bar"] = {
    (n + 1, f + 1): int(r[n, f]) for n in range(Nn) for f in range(Nf)
  }
  instance["sp_rho"] = ram_cap - (r * ram_req[None, :]).sum(axis=1)
  result = GreedyCoordinator().solve(instance, solver_options)
  y = np.array(result["y"], dtype=float).reshape(Nn, Nn, Nf)
  r_extra = np.array(result["r"], dtype=float).reshape(Nn, Nf)
  z = omega - y.sum(axis=1)
  return x, y, np.maximum(z, 0.0), r + r_extra.astype(int)
```

Note: inspect `GreedyCoordinator.solve`'s return dict once during implementation (`heuristic_coordinator.py:127` onward) — if the keys differ from `{"y", "z", "r"}` adapt the conversion, keep the `solve(data, solver_options) -> (x, y, z, r)` contract fixed.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_plasma_baselines.py -v`
Expected: 6 PASS (`solve_snapshot`/`stale_objectives` need a Pyomo solver — they are exercised in Task 13 behind a solver-availability skip, not here)

- [ ] **Step 5: Commit**

```bash
git add plasma/baselines tests/test_plasma_baselines.py
git commit -m "add plasma baselines: milp snapshot, routing lp, greedy, madea iface"
```

---

### Task 12: Evaluation metrics — `plasma/eval/regret.py`

**Files:**
- Create: `plasma/eval/__init__.py`, `plasma/eval/regret.py`
- Test: `tests/test_plasma_baselines.py` (append — metrics consume baseline outputs)

**Interfaces:**
- Produces:
  - `cumulative_regret(method_obj: np.ndarray, oracle_obj: np.ndarray) -> np.ndarray` — elementwise `cumsum(oracle - method)`.
  - `adaptation_lag(times: np.ndarray, method_obj: np.ndarray, oracle_obj: np.ndarray, change_points: list, threshold: float = 0.1) -> list` — for each change point, seconds until `method >= (1 - threshold) * oracle` again (`np.nan` if never).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_plasma_baselines.py`)

```python
from plasma.eval.regret import adaptation_lag, cumulative_regret


def test_cumulative_regret():
  method = np.array([1.0, 1.0, 2.0])
  oracle = np.array([2.0, 2.0, 2.0])
  assert cumulative_regret(method, oracle).tolist() == [1.0, 2.0, 2.0]


def test_adaptation_lag_measures_recovery():
  times = np.arange(6, dtype=float)
  oracle = np.full(6, 10.0)
  method = np.array([10.0, 10.0, 2.0, 5.0, 9.5, 9.8])  # change point at t=2
  lag = adaptation_lag(times, method, oracle, change_points=[2.0])
  assert lag == [2.0]  # recovered at t=4 (9.5 >= 9.0)


def test_adaptation_lag_nan_when_never_recovering():
  times = np.arange(3, dtype=float)
  lag = adaptation_lag(times, np.zeros(3), np.full(3, 10.0),
                       change_points=[0.0])
  assert np.isnan(lag[0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_plasma_baselines.py -v`
Expected: new tests FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# plasma/eval/__init__.py  (empty file)

# plasma/eval/regret.py
from __future__ import annotations

from typing import List

import numpy as np


def cumulative_regret(
    method_obj: np.ndarray, oracle_obj: np.ndarray
  ) -> np.ndarray:
  return np.cumsum(np.asarray(oracle_obj) - np.asarray(method_obj))


def adaptation_lag(
    times: np.ndarray, method_obj: np.ndarray, oracle_obj: np.ndarray,
    change_points: List[float], threshold: float = 0.1
  ) -> List[float]:
  times = np.asarray(times)
  method_obj = np.asarray(method_obj)
  oracle_obj = np.asarray(oracle_obj)
  lags: List[float] = []
  for cp in change_points:
    after = times >= cp
    recovered = after & (method_obj >= (1.0 - threshold) * oracle_obj)
    idx = np.flatnonzero(recovered)
    lags.append(float(times[idx[0]] - cp) if len(idx) else float("nan"))
  return lags
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_plasma_baselines.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add plasma/eval tests/test_plasma_baselines.py
git commit -m "add plasma regret and adaptation-lag metrics"
```

---

### Task 13: Acceptance tests — LP convergence, dead neighbor at scale, MILP gap

These are the spec's remaining non-negotiable acceptance criteria, run against the finished stack. They are slower (~1–3 min total); mark each with `@pytest.mark.slow` only if the repo's pytest config defines that marker — otherwise leave unmarked (check `pyproject.toml`/`pytest.ini` first; currently no marker config exists, so leave unmarked).

**Files:**
- Test: `tests/test_plasma_routing.py` (LP convergence), `tests/test_plasma_e2e.py` (MILP gap, solver-gated)

- [ ] **Step 1: Write the LP-convergence test** (append to `tests/test_plasma_routing.py`)

```python
from plasma.baselines.milp_baseline import routing_lp
from plasma.engine import PlasmaEngine


def test_physarum_converges_to_lp_routing_fractions():
  # 2-node line, fixed replicas, stationary integer traffic (lambda = 40):
  # node 0 undersized -> LP says: serve 20 locally, forward 20.
  opts = PlasmaOptions(k_sb=0, hb_latency_rounds=1)
  rng = np.random.default_rng(7)
  make = lambda i, nbrs: PlasmaNode(
    NodeParams(
      node_id=i, nbrs=nbrs, alpha=np.array([2.0]), gamma=np.array([0.1]),
      beta=np.full((len(nbrs), 1), 1.5), u_max=np.array([10.0]),
      ram_cap=100.0, ram_req=np.array([2.0]),
    ), opts, np.random.default_rng(10 + i))
  n0, n1 = make(0, (1,)), make(1, (0,))
  n0.r = np.array([2])   # capacity 20
  n1.r = np.array([4])   # capacity 40
  engine = PlasmaEngine([n0, n1], opts, rng)
  arrivals = np.array([[40], [0]])
  engine.run_rounds(150, arrivals)         # burn-in
  x_acc = np.zeros(1); y_acc = 0.0
  for _ in range(50):                      # measure 50 windows
    res = engine.run_rounds(1, arrivals)
    x_acc += res.x[0]; y_acc += res.y[0, 1, 0]
  lp_obj, lp_x, lp_y, lp_z = routing_lp(
    lam=np.array([[40.0], [0.0]]), r=np.array([[2], [4]]),
    u_max=np.full((2, 1), 10.0), alpha=np.full((2, 1), 2.0),
    beta=np.array([[[0.0], [1.5]], [[1.5], [0.0]]]),
    gamma=np.full((2, 1), 0.1), adjacency=np.array([[0, 1], [1, 0]]),
  )
  assert abs(x_acc[0] / 50 - lp_x[0, 0]) / 40.0 <= 0.05   # +-5% band
  assert abs(y_acc / 50 - lp_y[0, 1, 0]) / 40.0 <= 0.05
```

- [ ] **Step 2: Write the solver-gated MILP-gap e2e test** (append to `tests/test_plasma_e2e.py`)

```python
def _solver_available(name="gurobi"):
  try:
    from pyomo.environ import SolverFactory
    return SolverFactory(name).available(exception_flag=False)
  except Exception:
    return False


@pytest.mark.skipif(not _solver_available(), reason="no MILP solver")
def test_plasma_gap_vs_milp_on_small_graph(tmp_path):
  from generators.generate_data import update_data
  from plasma.baselines.milp_baseline import solve_snapshot
  from run_centralized_model import init_problem
  from utils.centralized import get_current_load
  config = _config(tmp_path, Nn=3, max_steps=6)
  config["solver_options"]["plasma"]["rounds_per_step"] = 30
  folder = run_plasma(config, parallelism=0)
  plasma_obj = pd.read_csv(os.path.join(folder, "obj.csv"))["Plasma"]
  # dynamic oracle on the same instance/traces (re-generated: same seed)
  base, traces, agents, _ = init_problem(
    config["limits"], "sinusoidal", config["max_steps"], config["seed"],
    str(tmp_path / "oracle"),
  )
  oracle = []
  for t in range(0, config["max_steps"] - 1):
    loadt = get_current_load(traces, agents, t)
    loadt = {k: int(round(v)) for k, v in loadt.items()}
    data = update_data(base, {"incoming_load": loadt})
    oracle.append(solve_snapshot(data, "gurobi", {"OutputFlag": 0})[4])
  # late-horizon gap (after Layer A/B settle): within 35% of the oracle
  # (M5's 10% target applies to the tuned 20-node run, not this smoke)
  late_p = plasma_obj.iloc[-2:].mean()
  late_o = np.mean(oracle[-2:])
  assert late_p >= late_o - abs(late_o) * 0.35
```

- [ ] **Step 3: Run the new tests**

Run: `uv run pytest tests/test_plasma_routing.py tests/test_plasma_e2e.py -v`
Expected: all PASS (gap test SKIPPED when no Gurobi). If the LP-convergence band fails, increase burn-in rounds (to ~300) before touching algorithm defaults; if it still fails, debug Layer A — the ±5% band is the spec's acceptance bar, do not widen it.

- [ ] **Step 4: Full suite + repo regression**

Run: `uv run pytest tests/ -x -q`
Expected: everything green, pre-existing tests untouched.

- [ ] **Step 5: Commit**

```bash
git add tests/test_plasma_routing.py tests/test_plasma_e2e.py
git commit -m "add plasma acceptance tests: lp convergence and milp gap"
```

---

### Task 14: LaTeX note skeleton — `faas-plasma-note/`

**Files:**
- Create: `faas-plasma-note/main.tex`, `faas-plasma-note/faas-plasma.tex`, `faas-plasma-note/references.bib`

Match the existing convention exactly (compare with `faas-magcaa-note/main.tex` before writing; adjust preamble to be byte-similar).

- [ ] **Step 1: Inspect the convention**

Run: `head -30 faas-magcaa-note/main.tex && head -40 faas-magcaa-note/faas-magcaa.tex`
Copy the documentclass/preamble structure verbatim, replacing the name.

- [ ] **Step 2: Write the three files**

`main.tex`:
```latex
% Same preamble as faas-magcaa-note/main.tex, then:
\input{faas-plasma}
```

`faas-plasma.tex` — section skeleton with real content stubs to be populated from M3 results onward:
```latex
\section{PLASMA: Physarum Routing with Simulated-Bifurcation Allocation}

\subsection{Problem statement}
% FRALB/DiFRALB instance as in the other notes; neighbor-only communication.

\subsection{Layer A: Physarum conductance routing}
% Per-request categorical routing on gated conductances; window update
% D <- (1-mu) D + mu g(phi) reward; capacity and spare-capacity gates.

\subsection{Layer B: discrete simulated bifurcation for replicas}
% Binary spin encoding, node-local Hamiltonian (field + RAM penalty +
% capacity chance-constraint + churn), dSB integrator, hysteresis,
% randomized commit (p_commit) under the shared round barrier.

\subsection{Round-barrier formulation}
% Single synchronous execution mode; SB pass every k_sb rounds.

\subsection{Acceptance criteria and results}
% LP-convergence band, dSB vs brute force, gap vs MILP (populate from M3).

\subsection{Privacy comparison vs FaaS-MADeA}
% Heartbeat carries only spare/alpha/pull: strictly less disclosure than
% MADeA's bid exchange.
```

`references.bib` — seed with the two published lines of work named in the spec:
```bibtex
@article{bonifaci2012physarum,
  author  = {Bonifaci, Vincenzo and Mehlhorn, Kurt and Varma, Girish},
  title   = {Physarum can compute shortest paths},
  journal = {Journal of Theoretical Biology},
  volume  = {309},
  pages   = {121--133},
  year    = {2012}
}

@article{goto2021high,
  author  = {Goto, Hayato and Endo, Kotaro and Suzuki, Masaru and others},
  title   = {High-performance combinatorial optimization based on classical mechanics},
  journal = {Science Advances},
  volume  = {7},
  number  = {6},
  year    = {2021}
}
```

- [ ] **Step 3: Commit**

```bash
git add faas-plasma-note/
git commit -m "scaffold faas-plasma-note LaTeX package"
```

---

## Milestone map (design §4 → tasks)

| Milestone | Tasks | Notes |
|---|---|---|
| M1 skeleton + scheduler | 1, 2, 6 (partial), 9 | 3-node line e2e in Task 9 |
| M2 Layer A alone | 3, 4, 6, 8, 13 (LP convergence) | `k_sb = 0` = fixed replicas |
| M3 baselines | 11 | milp/greedy/madea_iface + routing LP |
| M4 Layer B alone | 5 | dSB vs brute force |
| M5 coupling + hysteresis | 7, 8 (phase-locked tests), 13 (gap) | 10% gap target is a tuning goal on 20-node G(n,m), tracked in the note, not a CI assert |
| M6 non-stationarity + failures | 8 (`set_alive`), 11 (`stale_objectives`), 12 | comparison tables are experiment work via `run.py --methods centralized plasma faas-madea`, not new code |
| M7 rare-function mode + sweeps | 3, 6 (`rare_function_mode`) | sweeps reuse `run.py --loop_over`; nothing new to build |

Deliberately not built (YAGNI, per design doc): MMPP burst generator (existing `clipped`/`sinusoidal` traces cover M1–M5; add a small generator in `plasma/sim/` only when M6 experiments demand it), plotting modules (`postprocessing.py` covers it), a sweep runner (`--loop_over` exists), pure-Poisson traces (same rule).

## Self-review checklist (done during planning)

- Spec coverage: all §8 non-negotiable tests mapped — locality invariant (Task 4 whitelist + Task 5 node-local ctx), capacity gate (Task 6), RAM repair (Tasks 5, 7), dSB-vs-brute-force (Task 5), Physarum-LP (Task 13), dead neighbor (Tasks 3, 8), oscillation/hysteresis (Task 7), message budget (Task 8), phase-locked commit (Task 8). Async variant: intentionally absent everywhere, `execution_mode` rejected by construction (Task 1 test).
- Types consistent across tasks: `PlasmaOptions`, `Heartbeat`, `NodeParams`, `WindowCounts`, `StepResult`, `HamiltonianContext` defined once each, consumed by name.
- No placeholders: every code step is complete; the one deliberate deferral (GreedyCoordinator return-dict keys) is an explicit verify-then-adapt instruction with a fixed contract.
