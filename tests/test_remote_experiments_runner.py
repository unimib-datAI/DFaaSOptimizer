from ray_dispatcher import Job, JobHandle, JobStatus

from remote_experiments.manifest import Manifest
from remote_experiments.runner import run_batch


class FakeDispatcher:
  def __init__(self, status_sequences, hosts=None):
    self._sequences = {k: list(v) for k, v in status_sequences.items()}
    self._hosts = hosts or {}
    self.cancelled: list[str] = []

  def submit(self, jobs):
    return [JobHandle(batch_id="b1", job_id=j.id, token=j.id) for j in jobs]

  def status(self, handle):
    seq = self._sequences[handle.job_id]
    return seq.pop(0) if len(seq) > 1 else seq[0]

  def cancel(self, handle):
    self.cancelled.append(handle.job_id)

  def running_hosts(self):
    return dict(self._hosts)


def _jobs(*ids):
  return [Job(id=i, command=("echo",)) for i in ids]


def test_run_batch_records_succeeded_status_and_host(tmp_path):
  dispatcher = FakeDispatcher(
    {"e1": [JobStatus.RUNNING, JobStatus.SUCCEEDED]}, hosts={"e1": "vm1"}
  )
  manifest = Manifest(tmp_path / "m.json")
  ticks = []
  completed = run_batch(
    dispatcher, _jobs("e1"), manifest, on_tick=ticks.append, sleep=lambda s: None
  )
  assert completed is True
  assert manifest.status("e1") == "succeeded"
  assert manifest.host("e1") == "vm1"
  assert len(ticks) >= 1


def test_run_batch_records_duration_from_submit_to_terminal(tmp_path):
  times = iter([100.0, 105.0])
  dispatcher = FakeDispatcher({"e1": [JobStatus.SUCCEEDED]}, hosts={})
  manifest = Manifest(tmp_path / "m.json")
  completed = run_batch(
    dispatcher, _jobs("e1"), manifest, on_tick=lambda h: None,
    sleep=lambda s: None, now=lambda: next(times),
  )
  assert completed is True
  assert manifest.duration("e1") == 5.0


def test_run_batch_stops_and_cancels_on_keyboard_interrupt(tmp_path):
  dispatcher = FakeDispatcher({"e1": [JobStatus.RUNNING] * 5}, hosts={"e1": "vm1"})
  manifest = Manifest(tmp_path / "m.json")

  def _raising_sleep(_seconds):
    raise KeyboardInterrupt

  completed = run_batch(
    dispatcher, _jobs("e1"), manifest, on_tick=lambda h: None, sleep=_raising_sleep
  )
  assert completed is False
  assert "e1" in dispatcher.cancelled
  assert manifest.status("e1") == "cancelled"


def test_run_batch_preserves_host_after_lease_released_before_terminal_is_observed(tmp_path):
  hosts_per_tick = [{"e1": "vm1"}, {}]

  class _Dispatcher(FakeDispatcher):
    def running_hosts(self):
      return hosts_per_tick.pop(0) if hosts_per_tick else {}

  dispatcher = _Dispatcher({"e1": [JobStatus.RUNNING, JobStatus.SUCCEEDED]})
  manifest = Manifest(tmp_path / "m.json")
  run_batch(dispatcher, _jobs("e1"), manifest, on_tick=lambda h: None, sleep=lambda s: None)
  assert manifest.host("e1") == "vm1"
