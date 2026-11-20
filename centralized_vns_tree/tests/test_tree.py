from itertools import product
from time import monotonic
import numpy as np
import pytest
from centralized_vns.core import Options
from centralized_vns_frontier.regions import Regions
from centralized_vns_frontier.proof import Ledger
from centralized_vns_frontier.tests.test_frontier import disjoint
from centralized_vns_tree.frontier import Tree
from centralized_vns_tree.moves import TreePool, interval
from centralized_vns_tree.core import solve


def test_sparse_cover_inherits_bounds_and_keeps_waiting_sibling():
  p = disjoint()
  full = Regions([0, 1], [0, 0])
  coarse = Regions([], [])
  zero = np.zeros(2, np.int64)
  book = Ledger(p, coarse, zero, 10.0, 0.0, monotonic())
  tree = Tree(full, book)
  tree.advance(0, zero)
  key = tree.reserve()
  assert key == (1, 0)
  tree.advance(0, zero)
  assert tree.active == [key]
  tree.finish_proof(key, 4.0)
  assert book.upper[0] == 10
  tree.advance(0, zero)
  assert tree.bounds[0] == {(1, 1): 10.0, (2, 0): 4.0, (2, 2): 4.0}
  for full_code in range(4):
    assert sum(full_code % 2**d == c for d, c in tree.bounds[0]) == 1
  for _ in range(4):
    d, c = tree.active[0]
    seed = tree.prefix[d].seed(p, c, zero)
    tree.advance(0, seed)
  assert tree.switches > 0


@pytest.mark.parametrize('backend', ['cpu', 'cuda'])
def test_generated_moves_respect_prefix_and_are_real(backend):
  if backend == 'cuda':
    cuda = pytest.importorskip('numba.cuda')
    if not cuda.is_available():
      pytest.skip('CUDA unavailable')
  p = disjoint()
  full = Regions([0, 1], [1, 0])
  o = Options(backend=backend, pool_size=2, candidates=96)
  pool = TreePool(p, o, full, 1)
  pool.branches[:] = [0, 3]
  pool.depths[:] = [1, 2]
  pool.assign()
  for level in range(4):
    ys = [np.array([0, 1]), np.array([2, 1])]
    for slot, y in enumerate(ys):
      pool.seed(slot, y, p.check(y))
    e = pool.engine
    if backend == 'cuda':
      e.level.copy_to_device(np.full(2, level, dtype=np.int64))
    else:
      e.level[:] = level
    pool.step(level)
    moves = e.moves.copy_to_host() if backend == 'cuda' else e.moves
    for slot in range(2):
      for move in moves[slot]:
        new = ys[slot].copy()
        for edge, v in move:
          if edge >= 0:
            assert v != 0
            new[edge] += v
        if np.any(move[:, 0] >= 0):
          assert np.any(new != ys[slot])
          assert full.identify(new) % (1 << int(pool.depths[slot])) == pool.branches[slot]
          p.check(new)
  assert interval(0, np.array([0]), np.array([6]), 1, 1, np.array([14])) == (7, 14)


def test_solver_bound_covers_enumerated_optimum():
  p = disjoint()
  optimum = max(p.check(np.array(y))['welfare'] for y in product(range(3), repeat=2))
  r = solve(
    p, Options(pool_size=2, candidates=8, iterations=20, time_limit=2), depth=20, refine_every=2
  )
  assert r['coordinator_objective'] == pytest.approx(optimum)
  assert r['coordinator_upper'] >= optimum - 1e-8


def test_generates_each_neighborhood_with_shared_receiver_ram():
  from centralized_vns.core import Problem
  from centralized_vns_tree.moves import generate_cpu

  p = Problem(
    load=[[4, 0], [0, 4], [0, 0], [0, 0]],
    x=[[0, 0]] * 4,
    capacity=[[2.5, 2.5]] * 4,
    memory=[1, 1],
    ram=[2] * 4,
    alpha=[[0, 0]] * 4,
    gamma=[[0, 0]] * 4,
    edges=[(0, 2, 0), (0, 3, 0), (1, 2, 1)],
    beta=[1, 1, 1],
  )
  reg = Regions([0], [1])
  o = Options(pool_size=1, candidates=64)
  pool = TreePool(p, o, reg, 1)
  engine = pool.engine
  for level, values in enumerate(([0, 0, 0], [1, 2, 0], [0, 0, 4], [1, 2, 0])):
    parent = np.array(values)
    pool.seed(0, parent, p.check(parent))
    emitted = 0
    for c in range(o.candidates):
      if generate_cpu(
        o.seed,
        0,
        0,
        c,
        level,
        engine.moves,
        engine.state.y,
        engine.state.incoming,
        engine.state.rejected,
        p.load,
        p.x,
        p.capacity,
        p.memory,
        p.ram,
        p.src,
        p.dst,
        p.fun,
        p.starts,
        engine.in_edges,
        engine.in_starts,
        engine.bit,
        engine.cut,
        0,
        1,
        pool.eligible,
        pool.cap,
      ):
        child = parent.copy()
        for edge, qty in engine.moves[0, c]:
          if edge >= 0:
            assert qty
            child[edge] += qty
        assert reg.identify(child) == 0 and np.any(child != parent)
        p.check(child)
        emitted += 1
    assert emitted > 0
