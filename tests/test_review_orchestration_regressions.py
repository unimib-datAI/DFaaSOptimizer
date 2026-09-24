"""Regression contracts from branch review findings 4–6, 8, and 22–24.

Only remote dispatch, algorithm execution, and an injected filesystem failure
are replaced; generation, persistence, selection, and postprocessing are real.
"""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from ray_dispatcher import JobHandle, JobStatus

import compare_results
import remote_experiments.campaign as campaign
import remote_experiments.cli as cli
import remote_experiments.definitions as definitions
import run
from remote_experiments.batch import Batch, Experiment
from remote_experiments.instances import materialize_instance, validate_instance
from remote_experiments.manifest import Manifest
from remote_experiments.results_reader import read_run_objective
from remote_experiments.survivors import SurvivorSelectionError, select_survivors


def test_single_model_default_plots_generic_result_columns(tmp_path, monkeypatch):
  folder = tmp_path / "Nn_3"
  inputs = folder / "postprocessing"
  inputs.mkdir(parents=True)
  pd.DataFrame({"obj": [1., 2.]}).to_csv(inputs / "obj.csv", index=False)
  pd.DataFrame({"runtime": [0.1, 0.2]}).to_csv(inputs / "runtime.csv", index=False)
  monkeypatch.setattr("sys.argv", [
    "compare_results.py", "-i", str(folder), "--run", "compare_single_model",
  ])
  args = compare_results.parse_arguments()
  output = tmp_path / "plots"
  try:
    compare_results.compare_single_model(
      args.postprocessing_folders, "{}_{:d}", "Nodes", args.models, str(output),
    )
    assert pd.read_csv(output / "obj.csv")["LoadManagementModel"].tolist() == [1., 2.]
    assert pd.read_csv(output / "runtime.csv")["LoadManagementModel"].tolist() == [0.1, 0.2]
    assert list(output.glob("*.png"))
  finally:
    plt.close("all")


@pytest.mark.parametrize("mode", [
  "compare_single_model", "compare_results", "compare_across_folders",
])
def test_comparison_explicit_models_override_mode_default(monkeypatch, mode):
  monkeypatch.setattr("sys.argv", [
    "compare_results.py", "-i", "results", "--run", mode,
    "--models", "HierarchicalMADeALevelCycles",
  ])
  assert compare_results.parse_arguments().models == ["HierarchicalMADeALevelCycles"]


def _experiment(algorithm, seed=1, suite="review"):
  config = {
    "seed": seed,
    "verbose": 0,
    "max_steps": 3,
    "min_run_time": 0,
    "max_run_time": 3,
    "run_time_step": 1,
    "base_solution_folder": f"solutions/{suite}-{algorithm}-{seed}",
    "limits": {
      "Nn": {"min": 2, "max": 2},
      "Nf": {"min": 1, "max": 1},
      "neighborhood": {"p": 1.0},
      "demand": {"values": [1.0]},
      "memory_capacity": {"min": 12, "max": 12},
      "memory_requirement": {"values": [2]},
      "max_utilization": {"min": 0.7, "max": 0.7},
      "load": {"trace_type": "fixed_sum", "values": [10.0]},
      "weights": {
        "alpha": {"min": 1.0, "max": 1.5},
        "beta_multiplier": {"min": 1.5, "max": 2.5},
        "gamma": {"min": 0.05, "max": 0.15},
        "delta_multiplier": {"min": 0.1, "max": 0.2},
      },
    },
  }
  return Experiment(
    id=f"{suite}-{algorithm}-{seed}", suite=suite, algorithm=algorithm,
    seed=seed, graph_params=deepcopy(config["limits"]),
    load_params=deepcopy(config["limits"]["load"]), config=config,
  )


def _screening_results(root, values):
  experiments = []
  for algorithm, objectives in values.items():
    for seed, objective in enumerate(objectives, start=1):
      experiment = _experiment(algorithm, seed)
      experiments.append(experiment)
      if objective is None:
        continue
      folder = root / experiment.id / "outputs" / experiment.id / "run"
      folder.mkdir(parents=True)
      (folder / "obj.csv").write_text(f"Model\n{objective}\n")
      (folder / "runtime.csv").write_text("tot\n1.0\n")
  return Batch("review", tuple(experiments))


def _write_run(folder, model_key, label, runtime):
  """Two timesteps with 8 local requests and 2 rejections, no forwarding."""
  folder.mkdir(parents=True, exist_ok=True)
  solution = {}
  for node in range(2):
    solution[f"n{node}_f0_loc"] = [4.0, 4.0]
    solution[f"n{node}_f0_fwd"] = [0.0, 0.0]
    solution[f"n{node}_f0"] = [1.0, 1.0]
  pd.DataFrame(solution).to_csv(folder / f"{model_key}_solution.csv", index=False)
  for suffix in ("replicas", "utilization"):
    pd.DataFrame({"n0_f0": [1.0, 1.0], "n1_f0": [1.0, 1.0]}).to_csv(
      folder / f"{model_key}_{suffix}.csv", index=False,
    )
  pd.DataFrame({
    "n0_f0_n1_tot": [0.0, 0.0], "n1_f0_n0_tot": [0.0, 0.0],
    "n0_f0_n1_accepted": [0.0, 0.0], "n1_f0_n0_accepted": [0.0, 0.0],
  }).to_csv(folder / f"{model_key}_detailed_fwd_solution.csv", index=False)
  pd.DataFrame({label: [100.0, 100.0]}).to_csv(folder / "obj.csv", index=False)
  centralized = model_key == "LoadManagementModel"
  pd.DataFrame({label if centralized else "tot": [runtime, runtime]}).to_csv(
    folder / "runtime.csv", index=False,
  )
  termination = "optimal" if centralized else (
    "converged (it: 2; obj. deviation: 0.0; best it: 1)"
  )
  pd.DataFrame({"0": [termination, termination]}).to_csv(
    folder / "termination_condition.csv", index=not centralized,
  )
  return str(folder)


@pytest.fixture(autouse=True)
def _close_figures():
  yield
  plt.close("all")


@pytest.mark.parametrize("status", [JobStatus.FAILED, JobStatus.TIMED_OUT])
@pytest.mark.parametrize("initial_state", ["confirm", "stale-confirm", "stale-select"])
def test_failed_campaign_suite_remains_pending_for_resume(tmp_path, monkeypatch, status, initial_state):
  monkeypatch.chdir(tmp_path)
  suite = "review-confirm"
  experiment = _experiment("centralized", suite=suite)
  completed_experiment = _experiment("centralized", seed=2, suite=suite)
  experiments = (experiment, completed_experiment)
  monkeypatch.setitem(definitions._REGISTRY, suite, lambda: list(experiments))
  monkeypatch.setattr(campaign, "CONFIRMATORY_SUITES", (suite,))
  monkeypatch.setattr(campaign, "SCREENING_SUITE", suite)
  state_path = tmp_path / "batches" / "campaign-state.json"
  state_path.parent.mkdir()
  state_path.write_text(json.dumps({
    "stage": "select" if initial_state == "stale-select" else "confirm",
    "done_suites": [suite] if initial_state == "stale-confirm" else [],
  }))
  if initial_state.startswith("stale"):
    Batch(suite, experiments).save(state_path.parent / f"{suite}.json")
    manifest = Manifest(state_path.parent / f"{suite}.manifest.json")
    manifest.record(experiment.id, status=status.value)
    manifest.record(completed_experiment.id, status="succeeded")
  inventory = tmp_path / "inventory.yaml"
  inventory.write_text("hosts:\n  - host: example.invalid\n    user: test\n    slots: 1\n")
  submitted = []

  class TerminalDispatcher:
    def __init__(self, inventory, project, *, results_dir):
      pass

    def __enter__(self):
      return self

    def __exit__(self, *exc):
      return False

    def submit(self, jobs, *, batch_id):
      submitted.append([j.id for j in jobs])
      return [JobHandle(batch_id=batch_id, job_id=j.id, token=j.id) for j in jobs]

    def status(self, handle):
      return JobStatus.SUCCEEDED if handle.job_id == completed_experiment.id else status

    def running_hosts(self):
      return {}

  monkeypatch.setattr(cli, "Dispatcher", TerminalDispatcher)
  args = SimpleNamespace(
    instances=str(tmp_path / "instances"), inventory=str(inventory),
    project_path=str(tmp_path), results_dir=str(tmp_path / "results"),
    gurobi_license=None, python_version="3.10.19", uv_version="0.11.25",
  )
  campaign.run_campaign(args)

  manifest = Manifest(state_path.parent / f"{suite}.manifest.json")
  assert manifest.status(experiment.id) == status.value
  assert suite not in json.loads(state_path.read_text())["done_suites"]
  if initial_state == "stale-select":
    assert json.loads(state_path.read_text())["stage"] == "screening"
    campaign.run_campaign(args)
    assert submitted == [[experiment.id], [experiment.id]]
    return
  status = JobStatus.SUCCEEDED
  campaign.run_campaign(args)
  assert Manifest(state_path.parent / f"{suite}.manifest.json").status(experiment.id) == "succeeded"
  assert json.loads(state_path.read_text())["done_suites"] == [suite]
  assert submitted[-1] == [experiment.id]


def test_screening_does_not_promote_nan_objective(tmp_path):
  # Four of five runs are finite: enough for the existing 80% batch threshold.
  batch = _screening_results(tmp_path, {
    "anchor": [100], "invalid": [float("nan")], "best": [99],
    "second": [98], "third": [97],
  })
  assert select_survivors(batch, tmp_path, anchors=("anchor",), n=1) == ["best"]


@pytest.mark.parametrize("anchor,better,worse", [(-10, -11, -20), (0, -1, -10)])
def test_screening_prefers_higher_welfare_when_best_is_nonpositive(
    tmp_path, anchor, better, worse,
  ):
  batch = _screening_results(tmp_path, {
    "anchor": [anchor], "better": [better], "worse": [worse],
  })
  assert select_survivors(batch, tmp_path, anchors=("anchor",), n=1) == ["better"]


def test_screening_does_not_reward_missing_difficult_instances(tmp_path):
  batch = _screening_results(tmp_path, {
    "anchor": [100, 100], "fragile": [100, None], "reliable": [90, 90],
  })
  assert select_survivors(batch, tmp_path, anchors=("anchor",), n=1) == ["reliable"]


@pytest.mark.parametrize("invalid", ["nan", "inf", "-inf"])
def test_screening_reader_rejects_partially_nonfinite_runs(tmp_path, invalid):
  (tmp_path / "obj.csv").write_text(f"Model\n100\n{invalid}\n")
  assert read_run_objective(tmp_path) is None


def test_screening_compares_candidates_on_shared_instances(tmp_path):
  batch = _screening_results(tmp_path, {
    "anchor": [100] * 5,
    "steady": [92, 92, 92, 92, 0],
    "uneven": [100, 89, 89, 89, None],
  })
  # On their four shared instances, steady averages 92 versus uneven's 91.75.
  assert select_survivors(batch, tmp_path, anchors=("anchor",), n=1) == ["steady"]


def test_screening_refuses_insufficient_shared_coverage(tmp_path):
  batch = _screening_results(tmp_path, {
    "anchor": [100] * 5,
    "first": [None, 90, 90, 90, 90],
    "second": [90, None, 90, 90, 90],
  })
  with pytest.raises(SurvivorSelectionError, match="shared"):
    select_survivors(batch, tmp_path, anchors=("anchor",), n=1)


def test_fix_r_uses_newly_generated_centralized_solution(tmp_path, monkeypatch):
  centralized = tmp_path / "centralized"

  def central_runner(config, **kwargs):
    return _write_run(centralized, "LoadManagementModel", "LoadManagementModel", 2)

  def macro_runner(config, parallelism, **kwargs):
    assert config["opt_solution_folder"] == str(centralized)
    return _write_run(tmp_path / "macro", "LSP", "FaaS-MACrO", 1)

  monkeypatch.setattr(run, "run_centralized", central_runner)
  monkeypatch.setattr(run, "run_iterations", macro_runner)
  run.run(_experiment("centralized").config, str(tmp_path), 1,
          ["centralized", "faas-macro"], "centralized", True, 0, False, "Nn")
  folders = json.loads((tmp_path / "experiments.json").read_text())
  assert folders["centralized"] == [str(centralized)]
  assert folders["faas-macro"] == [str(tmp_path / "macro")]


def test_resume_can_add_method_without_a_centralized_reference(tmp_path, monkeypatch):
  previous = _write_run(tmp_path / "macro", "LSP", "FaaS-MACrO", 1)
  (tmp_path / "experiments.json").write_text(json.dumps({
    "experiments_list": [[2, 1]], "centralized": [], "faas-macro": [previous],
  }))

  def powerd_runner(config, parallelism, **kwargs):
    assert "opt_solution_folder" not in config
    return _write_run(tmp_path / "powerd", "LSPc", "FaaS-MAPoD", 1)

  monkeypatch.setattr(run, "run_powerd", powerd_runner)
  run.run(_experiment("faas-powd").config, str(tmp_path), 1,
          ["faas-powd"], "centralized", False, 0, False, "Nn")
  folders = json.loads((tmp_path / "experiments.json").read_text())
  assert folders["faas-macro"] == [previous]
  assert folders["faas-powd"] == [str(tmp_path / "powerd")]


def test_campaign_preparation_creates_batches_directory(tmp_path, monkeypatch):
  monkeypatch.chdir(tmp_path)
  experiment = _experiment("centralized")
  monkeypatch.setitem(definitions._REGISTRY, "review", lambda: [experiment])
  batch = campaign._define_and_materialize("review", str(tmp_path / "instances"))
  stored = Batch.load(tmp_path / "batches" / "review.json")
  assert [e.id for e in stored.experiments] == ["review-centralized-1"]
  assert stored == batch


def test_materialization_can_retry_after_interrupted_payload_write(tmp_path, monkeypatch):
  experiment = _experiment("centralized")
  destination = tmp_path / "instance"
  write_text = Path.write_text

  def interrupted_write(path, *args, **kwargs):
    if path.name == "input_requests_traces.json":
      raise OSError("simulated interrupted disk write")
    return write_text(path, *args, **kwargs)

  with monkeypatch.context() as fault:
    fault.setattr(Path, "write_text", interrupted_write)
    with pytest.raises(OSError, match="interrupted disk write"):
      materialize_instance(experiment, destination)

  assert not destination.exists()
  materialize_instance(experiment, destination)
  metadata = validate_instance(destination)
  assert metadata["generation"]["generation_seed"] == 1
  traces = json.loads((destination / "input_requests_traces.json").read_text())
  assert len(traces["0"]["0"]) == 3


def test_materialization_reuses_valid_instance_without_rewriting(tmp_path, monkeypatch):
  experiment = _experiment("centralized")
  destination = materialize_instance(experiment, tmp_path / "instance")
  original = {path.name: path.read_bytes() for path in destination.iterdir()}

  def no_writes(*args, **kwargs):
    raise OSError("existing instances must not be rewritten")

  monkeypatch.setattr(Path, "write_text", no_writes)
  assert materialize_instance(experiment, destination) == destination
  assert {path.name: path.read_bytes() for path in destination.iterdir()} == original


def test_runtime_comparison_accepts_decentralized_reference(tmp_path):
  centralized = _write_run(
    tmp_path / "centralized", "LoadManagementModel", "LoadManagementModel", 2,
  )
  macro = _write_run(tmp_path / "macro", "LSP", "FaaS-MACrO", 1)
  run.results_postprocessing({
    "experiments_list": [[2, 1]], "centralized": [centralized], "faas-macro": [macro],
  }, str(tmp_path), "Nn", ["centralized", "faas-macro"], "faas-macro")
  runtime = pd.read_csv(tmp_path / "postprocessing" / "runtime.csv")
  assert runtime["LoadManagementModel"].tolist() == [2, 2]
  assert runtime["FaaS-MACrO"].tolist() == [1, 1]
  assert runtime["dev_LoadManagementModel"].tolist() == [2, 2]


def test_custom_baseline_does_not_require_optional_rejections_csv(tmp_path):
  pd.DataFrame({
    "Nn": [2, 2], "LoadManagementModel": [100, 100], "FaaS-MACrO": [90, 90],
  }).to_csv(tmp_path / "obj.csv", index=False)
  pd.DataFrame({
    "Nn": [2, 2], "LoadManagementModel": [2, 2], "FaaS-MACrO": [1, 1],
  }).to_csv(tmp_path / "runtime.csv", index=False)
  objective, rejections, runtime = compare_results.compare_results(
    str(tmp_path), "Nn", "Nodes", ["LoadManagementModel", "FaaS-MACrO"],
    baseline_model="FaaS-MACrO",
  )
  assert rejections is None
  assert objective["FaaS-MACrO"].tolist() == [90, 90]
  assert runtime["LoadManagementModel"].tolist() == [2, 2]
  assert (tmp_path / "box-vs-FaaS-MACrO.png").is_file()
