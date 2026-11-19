"""Best point of a path replacing independent, node-disjoint blocks of flows."""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def relink(p, start, guide):
  edges = np.flatnonzero(start != guide)
  graph = csr_matrix(
    (np.ones(len(edges)), (p.src[edges], p.dst[edges])), shape=(len(p.load), len(p.load))
  )
  count, labels = connected_components(graph, directed=False)
  blocks = labels[p.src[edges]]
  gains = np.bincount(blocks, weights=(guide[edges] - start[edges]) * p.reward[edges], minlength=count)
  # On the path ordered by decreasing gain, the maximum follows all positive blocks.
  change = edges[gains[blocks] > 0]
  child = start.copy()
  child[change] = guide[change]
  return child


def cross_regions(book, limit, cuda_relink=None):
  """One frozen leader per region, unordered pairs, original feasibility and bounds."""
  from itertools import combinations, islice

  leaders = sorted(
    ((b, archive[0]) for b, archive in enumerate(book.elite) if archive),
    key=lambda item: (-item[1].welfare, item[0]),
  )[:limit]
  pairs = combinations(leaders, 2)
  before = book.best
  count = improved = better_than_before = 0
  winner = None
  while jobs := list(islice(pairs, 32)):
    if cuda_relink is None:
      children = [relink(book.p, a.y, g.y) for (_, a), (_, g) in jobs]
    else:
      children = cuda_relink(
        np.stack([a.y for (_, a), _ in jobs]), np.stack([g.y for _, (_, g) in jobs])
      )
    for ((b, a), (c, g)), child in zip(jobs, children, strict=True):
      # A cross-region child can fall in either parent region or a third region.
      r = book.offer('final_path_relinking', child)
      count += 1
      improved += int(r.welfare > max(a.welfare, g.welfare) + 1e-10)
      better_than_before += int(r.welfare > before.welfare + 1e-10)
      if r is book.best:
        winner = dict(parents=[b, c], child_region=int(book.regions.identify(child)))
  return dict(
    parent_regions=[b for b, _ in leaders],
    pairs=count,
    improved_children=improved,
    better_than_pre_final=better_than_before,
    winning_pair=winner,
    backend='cuda' if cuda_relink is not None else 'cpu',
  ), np.stack([r.y for _, r in leaders])
