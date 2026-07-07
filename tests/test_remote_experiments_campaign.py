import argparse
import json

import remote_experiments.campaign as campaign


def _fake_args(tmp_path):
  return argparse.Namespace(
    inventory=str(tmp_path / "inventory.yaml"),
    results_dir=str(tmp_path / "results"),
    instances=str(tmp_path / "instances"),
    gurobi_license=None,
    project_path=".",
    python_version="3.10.19",
    uv_version="0.11.25",
  )


def test_campaign_runs_screening_then_selects_then_confirms(tmp_path, monkeypatch):
  calls = []

  def fake_run_suite(suite, args, state):
    calls.append(suite)

  def fake_select(batch, results_dir, **kw):
    return ["faas-madea", "faas-diffuse", "faas-powd", "faas-br-o"]

  monkeypatch.setattr(campaign, "_run_suite", fake_run_suite)
  monkeypatch.setattr(campaign, "select_survivors", fake_select)
  monkeypatch.setattr(campaign, "SURVIVORS_PATH", tmp_path / "survivors.json")
  monkeypatch.setattr(campaign, "STATE_PATH", tmp_path / "campaign-state.json")

  args = _fake_args(tmp_path)
  campaign.run_campaign(args)

  assert calls[0] == "paper-a-screening"
  assert calls[1:] == list(campaign.CONFIRMATORY_SUITES)
  survivors = json.loads((tmp_path / "survivors.json").read_text())["survivors"]
  assert survivors == ["faas-madea", "faas-diffuse", "faas-powd", "faas-br-o"]


def test_campaign_resumes_from_state(tmp_path, monkeypatch):
  calls = []
  monkeypatch.setattr(campaign, "_run_suite", lambda s, a, st: calls.append(s))
  monkeypatch.setattr(campaign, "select_survivors", lambda *a, **k: ["x", "y", "z", "w"])
  monkeypatch.setattr(campaign, "SURVIVORS_PATH", tmp_path / "survivors.json")
  state = tmp_path / "campaign-state.json"
  monkeypatch.setattr(campaign, "STATE_PATH", state)
  (tmp_path / "survivors.json").write_text(json.dumps({"survivors": ["x", "y", "z", "w"]}))
  state.write_text(json.dumps({"stage": "confirm", "done_suites": ["paper-e1-quality-runtime"]}))

  campaign.run_campaign(_fake_args(tmp_path))
  assert "paper-a-screening" not in calls
  assert "paper-e1-quality-runtime" not in calls
  assert calls[0] == "paper-e2-scalability"
