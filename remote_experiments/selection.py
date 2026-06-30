"""Parse free-text experiment selection ("1,2,5-7" / "all") and compute resume defaults."""

from __future__ import annotations

from .manifest import Manifest, SUCCEEDED


def parse_selection(text: str, n: int) -> list[int]:
  text = text.strip()
  if text == "" or text.lower() == "all":
    return list(range(n))
  indices: set[int] = set()
  for part in text.split(","):
    part = part.strip()
    if not part:
      continue
    if "-" in part:
      start_s, end_s = part.split("-", 1)
      indices.update(range(int(start_s), int(end_s) + 1))
    else:
      indices.add(int(part))
  for i in indices:
    if not (0 <= i < n):
      raise ValueError(f"selection index out of range: {i} (valid: 0..{n - 1})")
  return sorted(indices)


def default_selection(experiment_ids: list[str], manifest: Manifest) -> list[int]:
  return [i for i, eid in enumerate(experiment_ids) if manifest.status(eid) != SUCCEEDED]
