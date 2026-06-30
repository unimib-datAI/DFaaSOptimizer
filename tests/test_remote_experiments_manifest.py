from remote_experiments.manifest import Manifest


def test_unknown_experiment_status_is_never_run(tmp_path):
  manifest = Manifest(tmp_path / "m.json")
  assert manifest.status("e1") == "never_run"


def test_record_updates_status_and_persists(tmp_path):
  path = tmp_path / "m.json"
  manifest = Manifest(path)
  manifest.record("e1", status="running", host="vm1")
  assert manifest.status("e1") == "running"
  assert manifest.host("e1") == "vm1"
  assert path.exists()


def test_record_partial_update_preserves_other_fields(tmp_path):
  manifest = Manifest(tmp_path / "m.json")
  manifest.record("e1", status="running", host="vm1")
  manifest.record("e1", status="succeeded", duration_s=12.5)
  assert manifest.host("e1") == "vm1"
  assert manifest.duration("e1") == 12.5


def test_manifest_reloads_from_disk(tmp_path):
  path = tmp_path / "m.json"
  Manifest(path).record("e1", status="succeeded")
  reloaded = Manifest(path)
  assert reloaded.status("e1") == "succeeded"


def test_pending_ids_excludes_succeeded(tmp_path):
  manifest = Manifest(tmp_path / "m.json")
  manifest.record("e1", status="succeeded")
  manifest.record("e2", status="failed")
  assert manifest.pending_ids(["e1", "e2", "e3"]) == ["e2", "e3"]
