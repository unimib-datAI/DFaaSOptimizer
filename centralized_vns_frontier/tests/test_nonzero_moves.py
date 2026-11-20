import numpy as np
import pytest

from centralized_vns.core import Options
from centralized_vns_frontier.moves import changes_flow
from centralized_vns_frontier.pool import BoundedCPU, Pool
from centralized_vns_frontier.regions import Regions
from centralized_vns_frontier.tests.test_frontier import disjoint


def test_net_change_is_per_edge_and_precedes_scoring(monkeypatch):
  moves = np.array(
    [
      [
        [[0, 0], [-1, 0], [-1, 0], [-1, 0]],
        [[0, -2], [0, 2], [-1, 0], [-1, 0]],
        [[0, -2], [1, 2], [-1, 0], [-1, 0]],
        [[0, -2], [0, 2], [1, 1], [1, -2]],
      ]
    ]
  )
  assert [changes_flow(moves, 0, c) for c in range(4)] == [False, False, True, True]
  monkeypatch.setattr('centralized_vns_frontier.pool.score_move', lambda *_: 0.0)
  assert [BoundedCPU.score(0, c, moves) for c in range(4)] == [-np.inf, -np.inf, 0.0, 0.0]


@pytest.mark.parametrize('backend', ['cpu', 'metal', 'cuda'])
def test_zero_transfer_and_self_relocation_are_ineligible(backend):
  if backend == 'cuda':
    cuda = pytest.importorskip('numba.cuda')
    if not cuda.is_available():
      pytest.skip('CUDA unavailable')
  if backend == 'metal':
    pytest.importorskip('mlx.core')
  p = disjoint()
  pool = Pool(p, Options(backend=backend, pool_size=1, candidates=16), Regions([], []))
  e = pool.engine
  captured = []
  if backend != 'cuda':
    accept = e.accept

    def capture(index, level, gains):
      captured.append(np.asarray(gains).copy())
      accept(index, level, gains)

    e.accept = capture
  for level, traffic in ((0, [0, 0]), (0, [2, 2]), (1, [1, 1]), (3, [1, 1])):
    y = np.array(traffic, dtype=np.int64)
    pool.seed(0, y, p.check(y))
    if backend == 'cuda':
      e.level.copy_to_device(np.array([level], dtype=np.int64))
    else:
      e.level[0] = level
    e.step(0)
    gains = e.gains.copy_to_host()[0] if backend == 'cuda' else captured[-1]
    moves = e.moves.copy_to_host() if backend == 'cuda' else e.moves
    valid = np.array([changes_flow(moves, 0, c) for c in range(16)])
    if level == 1:
      assert np.any(moves[0, :, :, 1] != 0) and not valid.any()
    np.testing.assert_array_equal(np.isfinite(gains), valid)
    assert valid.any() == (traffic == [0, 0] or level == 3)
