"""Pure computation of batch/VM progress stats for the TUI — no rendering here."""

from __future__ import annotations

from dataclasses import dataclass

from ray_dispatcher import Inventory

from .manifest import Manifest, SUCCEEDED

_RUNNING = "running"
_NEVER_RUN = "never_run"
_PENDING = "pending"


@dataclass(frozen=True)
class HostStats:
  host: str
  slots_total: int
  slots_busy: int
  jobs_completed: int
  current_jobs: tuple[str, ...]


@dataclass(frozen=True)
class BatchStats:
  total: int
  succeeded: int
  failed: int
  running: int
  pending: int
  elapsed_s: float
  throughput_per_min: float
  eta_s: float | None
  hosts: tuple[HostStats, ...]


def estimate_eta(avg_duration_s: float, remaining: int, total_slots: int) -> float:
  if total_slots <= 0:
    raise ValueError("total_slots must be > 0")
  if remaining <= 0:
    return 0.0
  return avg_duration_s * remaining / total_slots


def summarize(
    experiment_ids: list[str],
    manifest: Manifest,
    inventory: Inventory,
    running_hosts: dict[str, str],
    elapsed_s: float,
    succeeded_at_start: int = 0,
  ) -> BatchStats:
  statuses = {eid: manifest.status(eid) for eid in experiment_ids}
  succeeded = sum(1 for s in statuses.values() if s == SUCCEEDED)
  running = sum(1 for s in statuses.values() if s == _RUNNING)
  pending = sum(1 for s in statuses.values() if s in (_NEVER_RUN, _PENDING))
  failed = len(statuses) - succeeded - running - pending

  durations = [
    manifest.duration(eid) for eid in experiment_ids
    if statuses[eid] == SUCCEEDED and manifest.duration(eid) is not None
  ]
  avg_duration = sum(durations) / len(durations) if durations else 0.0
  total_slots = sum(h.slots for h in inventory.hosts)
  remaining = running + pending
  eta_s = estimate_eta(avg_duration, remaining, total_slots) if durations else None
  session_succeeded = max(succeeded - succeeded_at_start, 0)
  throughput_per_min = (session_succeeded / elapsed_s * 60) if elapsed_s > 0 else 0.0

  host_stats = []
  for h in inventory.hosts:
    current_jobs = tuple(
      eid for eid, host in running_hosts.items() if host == h.host and eid in statuses
    )
    completed = sum(
      1 for eid in experiment_ids
      if statuses[eid] == SUCCEEDED and manifest.host(eid) == h.host
    )
    host_stats.append(HostStats(
      host=h.host, slots_total=h.slots, slots_busy=len(current_jobs),
      jobs_completed=completed, current_jobs=current_jobs,
    ))

  return BatchStats(
    total=len(statuses), succeeded=succeeded, failed=failed, running=running,
    pending=pending, elapsed_s=elapsed_s, throughput_per_min=throughput_per_min,
    eta_s=eta_s, hosts=tuple(host_stats),
  )
