import json
from pathlib import Path

import run


def test_methods_choice_accepts_faas_br(monkeypatch):
  argv = ["run.py", "-c", "config_files/planar_comparison.json",
          "--methods", "faas-br-s", "faas-br-r", "faas-br-o"]
  monkeypatch.setattr("sys.argv", argv)
  args = run.parse_arguments()
  assert {"faas-br-s", "faas-br-r", "faas-br-o"}.issubset(set(args.methods))


def test_run_module_exposes_br_runners():
  for name in ("run_br_s", "run_br_r", "run_br_o"):
    assert callable(getattr(run, name))


def test_planar_config_has_br_blocks():
  config = json.loads(Path("config_files/planar_comparison.json").read_text())
  so = config["solver_options"]
  assert "br_s" in so and "br_r" in so and "br_o" in so


def test_compare_results_palette_includes_mabr():
  import inspect
  import compare_results
  src = inspect.getsource(compare_results)
  assert '"FaaS-MABR-S"' in src and '"FaaS-MABR-R"' in src and '"FaaS-MABR-O"' in src


def test_compare_results_defaults_include_mabr(monkeypatch):
  monkeypatch.setattr("sys.argv", ["compare_results.py", "-i", "solutions/demo"])
  import compare_results
  args = compare_results.parse_arguments()
  assert {"FaaS-MABR-S", "FaaS-MABR-R", "FaaS-MABR-O"}.issubset(set(args.models))
