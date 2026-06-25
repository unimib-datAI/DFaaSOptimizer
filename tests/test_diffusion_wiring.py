import json
from pathlib import Path

import run


def test_methods_choice_accepts_faas_diffuse(monkeypatch):
  argv = ["run.py", "-c", "config_files/planar_comparison.json",
          "--methods", "faas-diffuse"]
  monkeypatch.setattr("sys.argv", argv)
  args = run.parse_arguments()
  assert "faas-diffuse" in args.methods


def test_run_module_exposes_diffusion_runner():
  assert hasattr(run, "run_diffusion")
  assert callable(run.run_diffusion)


def test_planar_config_has_diffusion_section():
  config = json.loads(Path("config_files/planar_comparison.json").read_text())
  diffusion = config["solver_options"]["diffusion"]
  assert "latency_weight" in diffusion
  assert "fairness_weight" in diffusion


def test_compare_results_palette_includes_madig():
  import inspect

  import compare_results

  source = inspect.getsource(compare_results)
  assert '"FaaS-MADiG"' in source
