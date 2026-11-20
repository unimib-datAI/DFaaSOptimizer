"""A sparse covering frontier; only active prefixes own GPU trajectories."""

from collections import deque
from time import monotonic
import numpy as np

from centralized_vns_frontier.proof import Ledger, certify_branch
from centralized_vns_frontier.regions import Regions
from centralized_vns_highs.core import run_highs


class Tree:
  def __init__(self, full, book):
    self.book = book
    self.prefix = [Regions(full.edges[:d], full.cuts[:d]) for d in range(len(full.edges) + 1)]
    coarse = book.regions
    self.base = len(coarse.edges)
    self.mask = coarse.count - 1
    self.active = [(self.base, b) for b in range(coarse.count)]
    self.bounds = [{key: float(book.upper[b])} for b, key in enumerate(self.active)]
    self.waiting = [deque() for _ in self.active]
    self.visits = np.zeros(coarse.count, np.int64)
    self.splits = self.switches = 0
    self.pinned = None
    self.probed = set()

  def limit(self, key):
    d, code = key
    b = code & self.mask
    return min(self.book.upper[b], self.bounds[b][key])

  def threshold(self):
    lower = self.book.best.welfare
    return lower + self.book.gap * max(1.0, abs(lower))

  def advance(self, b, y):
    """Split around the trajectory record or rotate to a queued sibling."""
    with self.book.lock:
      key = self.active[b]
      if key is None or key == self.pinned:
        return
      depth, code = key
      if self.prefix[depth].identify(y) != code:
        raise ArithmeticError('trajectory escaped active prefix')
      self.visits[b] += 1
      can_split = depth + 1 < len(self.prefix)
      live = self.limit(key) > self.threshold()
      if live and can_split and (self.visits[b] % 3 or not self.waiting[b]):
        upper = self.limit(key)
        left, right = (depth + 1, code), (depth + 1, code + (1 << depth))
        new = int(self.prefix[depth + 1].identify(y))
        self.active[b] = (depth + 1, new)
        self.waiting[b].append(right if new == code else left)
        del self.bounds[b][key]
        self.bounds[b].update({left: upper, right: upper})
        self.splits += 1
      else:
        while self.waiting[b]:
          other = self.waiting[b].popleft()
          if self.limit(other) > self.threshold():
            if live:
              self.waiting[b].append(key)
            self.active[b] = other
            self.switches += 1
            break
        else:
          if not live:
            self.active[b] = None

  def reserve(self):
    with self.book.lock:
      threshold = self.threshold()
      # Prefer a deep, active prefix at equal upper bound; rotate fresh slices.
      best = None
      priority = None
      for b, leaves in enumerate(self.bounds):
        if self.book.closed[b]:
          continue
        for key, upper in leaves.items():
          cap = min(self.book.upper[b], upper)
          if cap <= threshold:
            continue
          rank = (cap, key not in self.probed, key == self.active[b], key[0])
          if priority is None or rank > priority:
            priority, best = rank, key
      self.pinned = best
      if best is not None:
        self.probed.add(best)
      return best

  def finish_proof(self, key, upper):
    with self.book.lock:
      if key != self.pinned:
        raise ArithmeticError('prefix changed during regional proof')
      b = key[1] & self.mask
      self.bounds[b][key] = min(upper, self.limit(key))
      # A single child certificate cannot close siblings waiting for a trajectory.
      self.book.tighten(max(self.bounds[b].values()), b)
      self.pinned = None


class Slice:
  """Expose one prefix as the sole region of a local proof ledger."""

  count = 1

  def __init__(self, regions, code):
    self.regions, self.code = regions, code

  def identify(self, y):
    if self.regions.identify(y) != self.code:
      raise ArithmeticError('witness outside proof prefix')
    return 0

  def constrain(self, solver, p, b):
    self.regions.constrain(solver, p, self.code)


def prove(tree, deadline, rows):
  book = tree.book
  now = monotonic()
  root = run_highs(
    book.p, book, now + max(0.0, deadline - now) * 0.8, tighten_receive=True, rows=rows
  )
  reports = []
  while monotonic() < deadline and not book.stop.is_set():
    key = tree.reserve()
    if key is None:
      break
    depth, code = key
    prefix = tree.prefix[depth]
    parent = book.witness(code & tree.mask)
    y = prefix.seed(book.p, code, parent.y)
    local = Ledger(book.p, Slice(prefix, code), y, tree.limit(key), 0.0, now)
    r = certify_branch(
      local, 0, min(deadline, monotonic() + 3), tighten_receive=True, pair_rows=rows
    )
    book.offer('branch_highs', local.best.y, code & tree.mask)
    tree.finish_proof(key, float(local.upper[0]))
    reports.append(dict(depth=depth, code=code, upper=float(local.upper[0]), status=r['status']))
  return root, reports
