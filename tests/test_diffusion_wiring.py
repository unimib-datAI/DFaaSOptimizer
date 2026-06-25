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
