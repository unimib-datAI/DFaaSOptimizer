"""A Frontier neighbor must change at least one integer edge flow."""


def changes_flow(moves, slot, candidate):
  for a in range(4):
    delta = moves[slot, candidate, a, 1]
    if delta == 0:
      continue
    edge = moves[slot, candidate, a, 0]
    if edge < 0:
      continue
    for b in range(4):
      if a != b and moves[slot, candidate, b, 0] == edge:
        delta += moves[slot, candidate, b, 1]
    if delta != 0:
      return True
  return False
