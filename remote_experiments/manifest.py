"""Per-experiment status persisted to disk, for cross-process stop/resume."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

NEVER_RUN = "never_run"
SUCCEEDED = "succeeded"


@dataclass
class ManifestEntry:
  status: str = NEVER_RUN
  host: str | None = None
  duration_s: float | None = None


class Manifest:
  def __init__(self, path: str | Path) -> None:
    self._path = Path(path)
    self._entries: dict[str, ManifestEntry] = {}
    if self._path.exists():
      raw = json.loads(self._path.read_text())
      self._entries = {k: ManifestEntry(**v) for k, v in raw.items()}

  def status(self, experiment_id: str) -> str:
    entry = self._entries.get(experiment_id)
    return entry.status if entry else NEVER_RUN

  def host(self, experiment_id: str) -> str | None:
    entry = self._entries.get(experiment_id)
    return entry.host if entry else None

  def duration(self, experiment_id: str) -> float | None:
    entry = self._entries.get(experiment_id)
    return entry.duration_s if entry else None

  def record(self, experiment_id: str, **fields) -> None:
    current = self._entries.get(experiment_id, ManifestEntry())
    updated = ManifestEntry(**{**asdict(current), **fields})
    self._entries[experiment_id] = updated
    self._save()

  def pending_ids(self, experiment_ids: list[str]) -> list[str]:
    return [eid for eid in experiment_ids if self.status(eid) != SUCCEEDED]

  def _save(self) -> None:
    self._path.parent.mkdir(parents=True, exist_ok=True)
    self._path.write_text(
      json.dumps({k: asdict(v) for k, v in self._entries.items()}, indent=2)
    )
