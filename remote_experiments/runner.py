"""Submit/poll loop driving a batch through ray_dispatcher, with stop-on-interrupt."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from ray_dispatcher import Job, JobHandle, JobStatus

from .manifest import Manifest

_TERMINAL = frozenset({
  JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.TIMED_OUT,
})


def run_batch(
    dispatcher,
    jobs: Sequence[Job],
    manifest: Manifest,
    on_tick: Callable[[dict[str, str]], None],
    *,
    poll_interval_s: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
  ) -> bool:
  """Submit jobs, poll until all terminal or interrupted.

  Returns True if the batch ran to completion, False if stopped (Ctrl-C):
  in-flight jobs are cancelled and the manifest reflects their last known
  state, so a later run_batch() call on the same experiment ids resumes
  from there.
  """
  handles: list[JobHandle] = dispatcher.submit(jobs)
  started_at = {h.job_id: now() for h in handles}
  last_known_host: dict[str, str] = {}
  for handle in handles:
    manifest.record(handle.job_id, status="pending")

  remaining = list(handles)
  try:
    while remaining:
      running_hosts = dispatcher.running_hosts()
      last_known_host.update(running_hosts)
      next_remaining = []
      for handle in remaining:
        status = dispatcher.status(handle)
        host = last_known_host.get(handle.job_id)
        if status in _TERMINAL:
          duration = now() - started_at[handle.job_id]
          manifest.record(handle.job_id, status=status.value, host=host, duration_s=duration)
        else:
          manifest.record(handle.job_id, status=status.value, host=host)
          next_remaining.append(handle)
      remaining = next_remaining
      on_tick(running_hosts)
      if remaining:
        sleep(poll_interval_s)
    return True
  except KeyboardInterrupt:
    for handle in remaining:
      dispatcher.cancel(handle)
      manifest.record(handle.job_id, status=JobStatus.CANCELLED.value)
    return False
