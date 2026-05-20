from pathlib import Path

import pandas as pd
import pytest

from benchmark_planar_3reg import (
  aggregate_results,
  build_base_config,
  render_html_report,
  run_benchmark,
  write_report,
)


def test_aggregate_results_computes_mean_and_std():
  raw = pd.DataFrame([
    {"Nn": 20, "seed": 0, "model": "centralized", "objective": 10.0, "wallclock_s": 1.0},
    {"Nn": 20, "seed": 1, "model": "centralized", "objective": 14.0, "wallclock_s": 3.0},
    {"Nn": 20, "seed": 0, "model": "hierarchical", "objective": 8.0, "wallclock_s": 2.0},
    {"Nn": 20, "seed": 1, "model": "hierarchical", "objective": 12.0, "wallclock_s": 4.0},
  ])

  summary = aggregate_results(raw)
  row = summary[(summary["Nn"] == 20) & (summary["model"] == "centralized")].iloc[0]

  assert row["objective_mean"] == 12.0
  assert row["objective_std"] == pytest.approx(2.8284271247)
  assert row["wallclock_mean_s"] == 2.0
  assert row["wallclock_std_s"] == pytest.approx(1.4142135623)
  assert row["runs"] == 2


def test_render_html_report_contains_summary_and_raw_tables(tmp_path: Path):
  raw = pd.DataFrame([
    {"Nn": 20, "seed": 0, "model": "centralized", "objective": 10.0, "wallclock_s": 1.0},
  ])
  summary = pd.DataFrame([
    {
      "Nn": 20,
      "model": "centralized",
      "objective_mean": 10.0,
      "objective_std": 0.0,
      "wallclock_mean_s": 1.0,
      "wallclock_std_s": 0.0,
      "runs": 1,
    },
  ])

  html_path = tmp_path / "report.html"
  render_html_report(
    summary,
    raw,
    html_path,
    meta={"sizes": [20, 40, 50], "seeds": [0, 1, 2, 3, 4]},
  )

  text = html_path.read_text()
  assert "<html" in text.lower()
  assert "Summary" in text
  assert "Raw Results" in text
  assert "centralized" in text


def test_build_base_config_uses_requested_sizes_and_seed(tmp_path: Path):
  config = build_base_config(tmp_path, nn=20, seed=3)
  assert config["limits"]["Nn"]["min"] == 20
  assert config["limits"]["Nn"]["max"] == 20
  assert config["seed"] == 3
  assert config["solver_name"] == "gurobi"
  assert config["limits"]["neighborhood"] == {"type": "planar", "degree": 3}


def test_run_benchmark_collects_three_models(monkeypatch, tmp_path: Path):
  def fake_run(config, **_kwargs):
    folder = tmp_path / f"{Path(config['base_solution_folder']).name}-out"
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"LoadManagementModel": [11.0]}).to_csv(folder / "obj.csv", index=False)
    pd.DataFrame({"runtime": [1.0]}).to_csv(folder / "runtime.csv", index=False)
    pd.DataFrame({"0": ["optimal"]}).to_csv(folder / "termination_condition.csv", index=False)
    return str(folder)

  monkeypatch.setattr("benchmark_planar_3reg.run_centralized", fake_run)
  monkeypatch.setattr("benchmark_planar_3reg.run_distributed", fake_run)
  monkeypatch.setattr("benchmark_planar_3reg.run_hierarchical", fake_run)

  raw = run_benchmark(output_root=tmp_path, sizes=[20], seeds=[0], solver_name="gurobi")
  assert set(raw["model"]) == {"centralized", "distributed", "hierarchical"}
  assert len(raw) == 3


def test_report_writer_creates_csv_and_html(tmp_path: Path):
  raw = pd.DataFrame([
    {"Nn": 20, "seed": 0, "model": "centralized", "objective": 10.0, "wallclock_s": 1.0},
    {"Nn": 20, "seed": 1, "model": "centralized", "objective": 14.0, "wallclock_s": 3.0},
  ])
  summary = aggregate_results(raw)

  write_report(raw, summary, tmp_path, meta={"sizes": [20], "seeds": [0, 1]})

  assert (tmp_path / "raw_results.csv").exists()
  assert (tmp_path / "summary.csv").exists()
  html = (tmp_path / "summary.html").read_text()
  assert "Planar Benchmark" in html
  assert "Objective mean" in html
  assert "Wallclock mean" in html
