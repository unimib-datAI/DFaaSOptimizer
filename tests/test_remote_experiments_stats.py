import pytest
from ray_dispatcher import Inventory, RemoteHost

from remote_experiments.manifest import Manifest
from remote_experiments.stats import estimate_eta, summarize


def test_estimate_eta_zero_remaining_is_zero():
  assert estimate_eta(avg_duration_s=10.0, remaining=0, total_slots=2) == 0.0


def test_estimate_eta_divides_by_total_slots():
  assert estimate_eta(avg_duration_s=10.0, remaining=4, total_slots=2) == 20.0


def test_estimate_eta_rejects_zero_slots():
  with pytest.raises(ValueError):
    estimate_eta(avg_duration_s=10.0, remaining=1, total_slots=0)


def _inventory():
  return Inventory((
    RemoteHost("vm1", user="ubuntu", slots=2),
    RemoteHost("vm2", user="ubuntu", slots=1),
  ))


def test_summarize_counts_by_status(tmp_path):
  manifest = Manifest(tmp_path / "m.json")
  manifest.record("e1", status="succeeded", host="vm1", duration_s=10.0)
  manifest.record("e2", status="running", host="vm1")
  manifest.record("e3", status="never_run")
  stats = summarize(["e1", "e2", "e3"], manifest, _inventory(), {"e2": "vm1"}, elapsed_s=60.0)
  assert stats.total == 3
  assert stats.succeeded == 1
  assert stats.running == 1
  assert stats.pending == 1
  assert stats.failed == 0


def test_summarize_eta_uses_succeeded_average_duration(tmp_path):
  manifest = Manifest(tmp_path / "m.json")
  manifest.record("e1", status="succeeded", host="vm1", duration_s=10.0)
  manifest.record("e2", status="running", host="vm1")
  stats = summarize(["e1", "e2"], manifest, _inventory(), {"e2": "vm1"}, elapsed_s=60.0)
  assert stats.eta_s == 10.0 / 3  # avg_duration=10, remaining=1, total_slots=3


def test_summarize_eta_none_without_any_succeeded(tmp_path):
  manifest = Manifest(tmp_path / "m.json")
  manifest.record("e1", status="running", host="vm1")
  stats = summarize(["e1"], manifest, _inventory(), {"e1": "vm1"}, elapsed_s=60.0)
  assert stats.eta_s is None


def test_summarize_attributes_current_jobs_per_host(tmp_path):
  manifest = Manifest(tmp_path / "m.json")
  manifest.record("e1", status="running")
  manifest.record("e2", status="running")
  stats = summarize(
    ["e1", "e2"], manifest, _inventory(), {"e1": "vm1", "e2": "vm2"}, elapsed_s=10.0
  )
  vm1 = next(h for h in stats.hosts if h.host == "vm1")
  vm2 = next(h for h in stats.hosts if h.host == "vm2")
  assert vm1.current_jobs == ("e1",)
  assert vm1.slots_busy == 1
  assert vm2.current_jobs == ("e2",)


def test_summarize_throughput_per_minute(tmp_path):
  manifest = Manifest(tmp_path / "m.json")
  manifest.record("e1", status="succeeded", duration_s=5.0)
  stats = summarize(["e1"], manifest, _inventory(), {}, elapsed_s=30.0)
  assert stats.throughput_per_min == 2.0  # 1 succeeded in 30s -> 2/min
