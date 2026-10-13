"""Regressions for data conservation and physical timestep alignment."""
import json

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import convert_data_format as converter
from generators.generate_data import generate_weights
from generators.load_generator import LoadGenerator
from postprocessing import load_runtime, load_solution
from tests.test_convert_data_format import _sample_instance_data, _sample_requests


@pytest.mark.parametrize("dtype", [int, float])
def test_fixed_sum_conserves_nonnegative_workload_with_arbitrary_node_ids(dtype):
  base = {"a": np.array([0, -2, 1], dtype=dtype),
          "b": np.array([0, 1, 1], dtype=dtype),
          "c": np.array([0, 3, 1], dtype=dtype)}
  out = LoadGenerator()._impose_system_workload(10, base)
  values = np.array(list(out.values()))
  assert list(out) == list(base)
  assert np.all(values >= 0)
  assert values.sum(axis=0) == pytest.approx([10, 10, 10])
  assert np.issubdtype(values.dtype, np.integer) == (dtype is int)


@pytest.mark.parametrize("nodes", [1, 3])
def test_heterogeneous_weights_generate_every_nodes_gamma(nodes):
  limits = {"weights": {"type": "heterogeneous", "alpha": {"min": 1, "max": 2},
            "gamma": {"min": 1, "max": 2}, "beta_multiplier": {"min": 1, "max": 2}, "delta_multiplier": {"min": 1, "max": 2}}}
  _, beta, gamma, delta = generate_weights(nodes, 2, limits, np.random.default_rng(11), nx.complete_graph(nodes))
  assert gamma.shape == (nodes, 2)
  assert np.isfinite(delta).all()
  assert np.all(gamma >= 1)
  assert len(np.unique(gamma, axis=0)) == nodes
  assert np.allclose(beta, beta.transpose(1, 0, 2))


def test_converter_exports_squared_coefficient_of_variation():
  result = converter.update_functions_info(converter.build_base_spec_dict(), _sample_instance_data())
  assert [f["duration_scv"] for f in result["functions"]] == pytest.approx([0.5, 0.5])


def test_converter_main_uses_requested_destination(tmp_path, monkeypatch):
  monkeypatch.chdir(tmp_path)
  monkeypatch.setattr(converter, "load_base_instance", lambda folder: (_sample_instance_data(), {}))
  monkeypatch.setattr(converter, "load_requests_traces", lambda folder: (_sample_requests(), 0, 2, 1))
  destination = tmp_path / "nested" / "output"
  converter.main("unused", str(destination))
  assert (destination / "base_spec.yaml").is_file()
  assert (destination / "spec_t1.yaml").is_file()
  assert not (tmp_path / "spec").exists()


def _write_results(folder, steps):
  for suffix in ("solution", "replicas", "detailed_fwd_solution", "utilization"):
    pd.DataFrame({"value": [10, 20]}).to_csv(folder / f"Model_{suffix}.csv", index=False)
  pd.DataFrame({"Model": [100, 200]}).to_csv(folder / "obj.csv", index=False)
  pd.DataFrame({"step": steps, "algorithm_s": [0.2, 0.4], "wall_s": [1, 2],
                "parallel_phases": [0, 0], "workers": [0, 0]}).to_csv(folder / "runtime.csv", index=False)


def test_solution_and_runtime_keep_real_timesteps(tmp_path):
  _write_results(tmp_path, [2, 5])
  for frame in load_solution(str(tmp_path), "Model"):
    assert frame.index.tolist() == [2, 5]
  runtime = load_runtime(str(tmp_path))
  assert runtime.index.tolist() == [2, 5]
  assert runtime.loc[5, "runtime"] == 0.4


def test_legacy_solution_recovers_steps_from_config(tmp_path):
  _write_results(tmp_path, [2, 5])
  (tmp_path / "runtime.csv").unlink()
  (tmp_path / "config.json").write_text(json.dumps({"min_run_time": 2, "max_run_time": 8,
    "max_steps": 10, "run_time_step": 3}))
  for frame in load_solution(str(tmp_path), "Model"):
    assert frame.index.tolist() == [2, 5]


def test_runtime_and_objective_comparison_aligns_by_timestep(tmp_path):
  from run import load_obj_value, runtime_series
  _write_results(tmp_path, [2, 5])
  objective = load_obj_value(str(tmp_path))
  measured = runtime_series(pd.read_csv(tmp_path / "runtime.csv"), "Model")
  other = runtime_series(pd.DataFrame({"step": [5, 8], "algorithm_s": [2, 4]}), "Other")
  compared = pd.DataFrame({"first": measured, "second": other})
  assert objective.index.tolist() == [2, 5]
  assert compared.index.tolist() == [2, 5, 8]
  assert compared.loc[5].tolist() == [0.4, 2]
  assert pd.isna(compared.loc[2, "second"])


def test_fixed_reference_works_at_nonzero_strided_steps(tmp_path):
  from tests.test_run_frame import _config, _run, FakeStep, OUTPUTS
  from utils.run_frame import Output
  (tmp_path / "ref").mkdir()
  reference = _run(_config(tmp_path / "ref", steps=5, min_run_time=1, max_run_time=5, run_time_step=2),
                   FakeStep(names=("LoadManagementModel",)),
                   outputs={"LoadManagementModel": Output(plot_filename=None)})
  (tmp_path / "target").mkdir()
  step = FakeStep()
  _run(_config(tmp_path / "target", steps=5, min_run_time=1, max_run_time=5, run_time_step=2,
               opt_solution_folder=str(reference)), step, outputs=OUTPUTS)
  assert step.seen == {1: {(1, 1): 1, (2, 1): 1}, 3: {(1, 1): 1, (2, 1): 1}}


def test_hierarchical_madea_name_means_same_variant_in_every_entrypoint():
  import methods
  import run
  from remote_experiments.jobs import SCRIPT_BY_ALGORITHM
  from hierarchical_auction import madea_level_cycles_runner, madea_runner
  assert run.run_hierarchical_madea is methods.HARMONIZED["hierarchical-madea"]
  assert run.run_hierarchical_madea is madea_level_cycles_runner.run
  assert SCRIPT_BY_ALGORITHM["hierarchical-madea"][0] == ("-m", "hierarchical_auction.madea_level_cycles_runner")
  assert run.run_hierarchical_madea_legacy is madea_runner.run
  assert SCRIPT_BY_ALGORITHM["hierarchical-madea-legacy"][0] == ("-m", "hierarchical_auction.madea_runner")


def test_forwarding_analysis_keeps_nonzero_physical_steps(tmp_path):
  from fwd_analysis import count_requests
  from tests.test_run_frame import _config, _run, FakeStep
  folder = _run(_config(tmp_path, steps=5, min_run_time=1, max_run_time=5, run_time_step=2), FakeStep())
  for frame in count_requests(str(folder), "LSPc"):
    assert frame["t"].unique().tolist() == [1, 3]


def test_standalone_postprocessing_accepts_nonzero_steps(tmp_path, monkeypatch):
  from tests.test_coverage_expansion import _write_model_outputs
  import postprocessing
  _write_model_outputs(tmp_path, "ModelA")
  pd.DataFrame({"step": [2, 5], "algorithm_s": [.1, .2]}).to_csv(tmp_path / "runtime.csv", index=False)
  # Rendering is covered by the plot smoke tests; exercise the real aggregation.
  monkeypatch.setattr(postprocessing.plt, "savefig", lambda *a, **k: None)
  out = postprocessing.process_results(str(tmp_path), ["ModelA"])
  assert (tmp_path / "postprocessing" / "by_node_count.csv").is_file()
  assert out == str(tmp_path / "postprocessing")


@pytest.mark.parametrize("label", ["HierarchicalMADeA", "HierarchicalMADeALevelCycles"])
def test_historical_hierarchical_alias_keeps_its_actual_variant(tmp_path, label):
  from run import resolve_result_model
  pd.DataFrame({label: [1.0]}).to_csv(tmp_path / "obj.csv", index=False)
  assert resolve_result_model(str(tmp_path), "hierarchical-madea") == ("LSPc", label)


def test_forwarding_analysis_subtracts_fractional_rejected_rates(tmp_path, monkeypatch):
  import fwd_analysis
  sentrecv = pd.DataFrame({"t": [0], "node": [0], "sent": [.5], "recv": [0.]})
  rejected = pd.DataFrame({"t": [0], "node": [0], "all": [.5], "f0": [.5]})
  monkeypatch.setattr(fwd_analysis, "count_requests", lambda *a: (pd.DataFrame(), sentrecv, rejected))
  monkeypatch.setattr(fwd_analysis, "load_requests_traces", lambda *a: ({0: {0: np.array([1], dtype=int)}}, 0, 1, 1))
  monkeypatch.setattr(fwd_analysis.plt, "savefig", lambda *a, **k: None)
  monkeypatch.setattr(fwd_analysis.LoadGenerator, "plot_input_load", lambda *a, **k: None)
  fwd_analysis.main(str(tmp_path), "LSPc")
  traces = json.loads((tmp_path / "load" / "norej_trace-0_1_1.json").read_text())
  assert traces["0"]["0"] == [.5]


def test_count_comparisons_keep_union_of_physical_steps():
  from run import merge_sol_dict
  first = {"tot": pd.DataFrame({"A": [30]}), "it 2": pd.DataFrame({"A": [10]}),
           "it 5": pd.DataFrame({"A": [20]})}
  second = {"tot": pd.DataFrame({"B": [70]}), "it 5": pd.DataFrame({"B": [30]}),
            "it 8": pd.DataFrame({"B": [40]})}
  result = merge_sol_dict([first, second], ["A", "B"]).set_index("time")
  assert result.index.tolist() == ["tot", 2, 5, 8]
  assert result.loc[5].tolist() == [20, 30]
  assert pd.isna(result.loc[8, "A"])
  assert result.loc[8, "B"] == 40
