"""utils/run_frame.py: the lifecycle shared by the decentralized runners, with a fake solve_step."""
import json
import logging
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from run_faasmacro import combine_solutions
from test_review_distributed_regressions import _materialized_config, _two_node_data
from utils import run_frame, step_time

OUTPUTS = {"LSP": run_frame.Output(checkpoint=True, plot_filename="sp.png"),
           "LSPc": run_frame.Output(checkpoint=True, plot_filename=None)}


def _solution(data, replicas):
  """A feasible solution: local processing within one replica, the rest rejected."""
  Nn, Nf = data[None]["Nn"][None], data[None]["Nf"][None]
  load = data[None]["incoming_load"]
  x = np.array([[min(load[(n + 1, 1)], 5.) * replicas for _ in range(Nf)] for n in range(Nn)])
  return combine_solutions(Nn, Nf, data, load, x, np.full((Nn, Nf), replicas), np.zeros(Nn),
                           None, np.zeros((Nn, Nn, Nf)), None, None, None, None)["sp"]


class FakeStep:
  def __init__(self, replicas=lambda t: 1, fail_at=None, names=("LSP", "LSPc")):
    self.replicas, self.fail_at, self.names, self.calls, self.seen = replicas, fail_at, names, [], {}

  def __call__(self, ctx, t, data):
    self.calls.append(t)
    self.seen[t] = deepcopy(ctx.step_data(data, t)[None].get("r_bar"))
    assert step_time.active()
    if t == self.fail_at:
      raise RuntimeError("solver failure")
    solution = _solution(data, self.replicas(t))
    self.returned = solution
    return run_frame.StepResult(solutions={name: solution for name in self.names}, objective=10. + t,
                                termination=f"converged (it: {t}; total runtime: 0.5)")


def _config(tmp_path, steps=2, **extra):
  config = _materialized_config(tmp_path, _two_node_data(), steps=steps)
  config.update(extra)
  return config


def _run(config, step, outputs=OUTPUTS, disable_plotting=True, log_on_file=True, method="FaaS-Fake"):
  return Path(run_frame.run(config, 0, log_on_file=log_on_file, disable_plotting=disable_plotting,
                            method=method, solve_step=step, outputs=outputs))


def test_outputs_have_the_legacy_formats(tmp_path):
  config = _config(tmp_path)
  folder = _run(config, FakeStep())
  assert (folder / "obj.csv").read_text() == "FaaS-Fake\n10.0\n11.0\n"
  assert (folder / "termination_condition.csv").read_text() == (
    ",0\n0,converged (it: 0; total runtime: 0.5)\n1,converged (it: 1; total runtime: 0.5)\n")
  runtime = pd.read_csv(folder / "runtime.csv")
  assert tuple(runtime.columns) == step_time.COLUMNS and runtime["step"].tolist() == [0, 1]
  assert (runtime["wall_s"] >= runtime["algorithm_s"]).all()
  for name in ("LSP", "LSPc"):
    for kind in ("solution", "offloaded", "replicas", "residual_capacity", "utilization", "detailed_fwd_solution"):
      assert (folder / f"{name}_{kind}.csv").exists()
    assert sorted(p.name for p in (folder / name).iterdir()) == ["0", "1"]
  assert json.loads((folder / "config.json").read_text()) == config
  assert (folder / "out.log").exists() and not step_time.active()


def test_returned_arrays_reach_the_decoder_unchanged(tmp_path, monkeypatch):
  import utils.centralized
  decoded = []
  original = utils.centralized.decode_solution

  def spy(x, y, z, r, xi, rho, U, complete):
    decoded.append((x, y, z, r, xi, rho, U))
    return original(x, y, z, r, xi, rho, U, complete)

  monkeypatch.setattr(utils.centralized, "decode_solution", spy)
  step = FakeStep()
  _run(_config(tmp_path, steps=1), step)
  keys = ("x", "y", "z", "r", "xi", "rho", "U")
  assert all(a is step.returned[k] for a, k in zip(decoded[0], keys))


def test_checkpoints_follow_the_output_selection_and_interval(tmp_path):
  outputs = {"LSP": run_frame.Output(checkpoint=False, plot_filename=None),
             "LSPc": run_frame.Output(checkpoint=True, plot_filename=None)}
  folder = _run(_config(tmp_path, steps=3, checkpoint_interval=2), FakeStep(), outputs)
  assert not (folder / "LSP").exists()
  assert sorted(p.name for p in (folder / "LSPc").iterdir()) == ["0", "2"]   # interval and last step


def test_plotting_enabled_and_disabled(tmp_path):
  (tmp_path / "a").mkdir()
  (tmp_path / "b").mkdir()
  shown = _run(_config(tmp_path / "a"), FakeStep(), disable_plotting=False)
  hidden = _run(_config(tmp_path / "b"), FakeStep(), disable_plotting=True)
  assert (shown / "sp.png").exists() and not list(hidden.glob("*.png"))


def test_non_unit_steps(tmp_path):
  step = FakeStep()
  folder = _run(_config(tmp_path, steps=5, min_run_time=1, max_run_time=5, run_time_step=2), step)
  assert step.calls == [1, 3]
  assert pd.read_csv(folder / "runtime.csv")["step"].tolist() == [1, 3]


def test_config_is_not_modified(tmp_path):
  config = _config(tmp_path)
  before = deepcopy(config)
  _run(config, FakeStep())
  assert config == before


def test_fixed_reference_is_indexed_by_step(tmp_path):
  (tmp_path / "ref").mkdir()
  (tmp_path / "run").mkdir()
  reference = _run(_config(tmp_path / "ref", steps=2),
                   FakeStep(replicas=lambda t: 1 - t, names=("LoadManagementModel",)),
                   {"LoadManagementModel": run_frame.Output(checkpoint=False, plot_filename=None)})
  step = FakeStep()
  _run(_config(tmp_path / "run", steps=2, opt_solution_folder=str(reference)), step)
  assert step.seen == {0: {(1, 1): 1, (2, 1): 1}, 1: {(1, 1): 0, (2, 1): 0}}
  plain = FakeStep()
  (tmp_path / "plain").mkdir()
  _run(_config(tmp_path / "plain"), plain)
  assert plain.seen == {0: None, 1: None}


def test_frame_failure_then_success_is_clean(tmp_path, capsys):
  root = logging.getLogger("dfaas")
  handlers = list(root.handlers)
  (tmp_path / "a").mkdir()
  (tmp_path / "b").mkdir()
  with pytest.raises(RuntimeError, match="solver failure"):
    _run(_config(tmp_path / "a", verbose=1), FakeStep(fail_at=1))
  assert not step_time.active() and root.handlers == handlers
  failed = next((tmp_path / "a" / "solutions").iterdir())
  failed_log = (failed / "out.log").read_text()
  assert failed_log.startswith("t = 0\n") and failed_log.endswith("t = 1\n")
  folder = _run(_config(tmp_path / "b", verbose=1), FakeStep())
  log = (folder / "out.log").read_text()
  assert log.startswith("t = 0\n") and "All solutions saved in:" in log
  assert (failed / "out.log").read_text() == failed_log      # the next run's messages stay in its log
  assert capsys.readouterr().out == ""


def test_incumbent_keeps_strict_improvements_as_copies():
  incumbent = run_frame.Incumbent(-np.inf)
  first = {"x": np.zeros(2)}
  assert incumbent.consider(first, -5., 0)                     # a first negative objective counts
  first["x"][0] = 99.                                          # the incumbent is a copy
  assert incumbent.solution["x"][0] == 0. and incumbent.iteration == 0
  assert not incumbent.consider({"x": np.ones(2)}, -5., 1)     # ties keep the incumbent
  assert incumbent.iteration == 0 and incumbent.solution["x"][0] == 0.
  assert incumbent.consider({"x": np.ones(2)}, 3., 2)
  assert incumbent.objective == 3. and incumbent.iteration == 2


def test_incumbent_copy_is_untimed(monkeypatch):
  class Clock:
    t = 0.

    def __call__(self):
      return self.t

  clock = Clock()
  monkeypatch.setattr(step_time, "perf_counter", clock)

  class Slow(dict):
    def __deepcopy__(self, memo):
      clock.t += 2.
      return Slow(self)

  step = step_time.start_step()
  run_frame.Incumbent(-np.inf).consider(Slow(a=1), 1., 0)
  row = step_time.end_step(step)
  assert row["algorithm_s"] == 0. and row["wall_s"] == 2.


def test_snapshot_combines_checks_and_scores():
  data = _two_node_data()
  load = data[None]["incoming_load"]
  args = (2, 1, data, load, np.full((2, 1), 5.), np.ones((2, 1)), np.zeros(2),
          None, np.zeros((2, 2, 1)), None, None, None, None)
  from utils.faasmacro import compute_centralized_objective
  csol, objective = run_frame.snapshot(data, *args)
  assert set(csol["sp"]) >= {"x", "y", "z", "r", "xi", "rho", "U"}
  assert objective == compute_centralized_objective(data, csol["sp"]["x"], csol["sp"]["y"], csol["sp"]["z"])
  infeasible = list(args)
  infeasible[4] = np.full((2, 1), 50.)                      # more than the incoming load
  with pytest.raises((RuntimeError, ValueError)):   # the checker's own exception is kept
    run_frame.snapshot(data, *infeasible)


def test_runner_failure_releases_pool_clock_and_log(tmp_path, monkeypatch):
  import sys
  import decentralized_diffusion
  import run_faasmacro
  sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "harmonization"))
  from capture_baseline import capture_config, instances
  root = logging.getLogger("dfaas")
  handlers, seen = list(root.handlers), {}

  def failing(*args, **kwargs):
    seen["pool"] = run_faasmacro._parallel_session.get()["pool"]
    raise RuntimeError("local solver failure")

  monkeypatch.setattr(decentralized_diffusion, "compute_social_welfare", failing)
  config = capture_config(instances()[0], tmp_path / "failed", "faas-diffuse")
  with pytest.raises(RuntimeError, match="local solver failure"):
    decentralized_diffusion.run(config, 2, log_on_file=True, disable_plotting=True)
  assert seen["pool"] is not None and not any(p.is_alive() for p in seen["pool"]._pool)
  assert run_faasmacro._parallel_session.get() is None
  assert not step_time.active() and root.handlers == handlers
  monkeypatch.undo()
  config = capture_config(instances()[0], tmp_path / "ok", "faas-diffuse")
  folder = Path(decentralized_diffusion.run(config, 0, log_on_file=True, disable_plotting=True))
  assert (folder / "out.log").read_text().startswith("t = 0\n")
  assert pd.read_csv(folder / "runtime.csv")["step"].tolist() == [0, 1]
