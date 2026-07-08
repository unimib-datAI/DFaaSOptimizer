import pytest
from remote_experiments.batch import Batch, Experiment
from remote_experiments.survivors import select_survivors, SurvivorSelectionError


def _exp(algo, nodes, seed):
  return Experiment(
    id=f"paper-a-screening-n{nodes}-f2-planar3-{algo}-s{seed}",
    suite="paper-a-screening", algorithm=algo, seed=seed,
    graph_params={"Nn": {"min": nodes, "max": nodes}, "Nf": {"min": 2, "max": 2}},
    load_params={}, config={},
  )


def _write_result(results_dir, exp, obj, runtime=0.1):
  run = results_dir / exp.id / "2026-07-06_00-00-00.000000"
  run.mkdir(parents=True)
  if obj is not None:
    (run / "obj.csv").write_text(f"Model\n{obj}\n")
  (run / "runtime.csv").write_text(f"tot\n{runtime}\n")


def test_selects_lowest_relative_deficit(tmp_path):
  # per instance obj_best is the max; rank candidates by mean deficit to it
  objs = {  # algo -> obj on the single instance (n50, seed1)
    "hierarchical-madea": 100.0,  # anchor, best, but not promotable
    "faas-madea": 99.0, "faas-diffuse": 98.0, "faas-powd": 97.0,
    "faas-br-o": 96.0, "faas-gcaa": 50.0,
  }
  exps = [_exp(a, 50, 1) for a in objs]
  batch = Batch(suite="paper-a-screening", experiments=tuple(exps))
  for e in exps:
    _write_result(tmp_path, e, objs[e.algorithm])
  survivors = select_survivors(batch, tmp_path, n=4)
  assert survivors == ["faas-madea", "faas-diffuse", "faas-powd", "faas-br-o"]
  assert "hierarchical-madea" not in survivors


def test_runtime_breaks_ties(tmp_path):
  exps = [_exp("faas-madea", 50, 1), _exp("faas-powd", 50, 1),
          _exp("hierarchical-madea", 50, 1)]
  batch = Batch(suite="paper-a-screening", experiments=tuple(exps))
  _write_result(tmp_path, exps[0], 90.0, runtime=0.5)
  _write_result(tmp_path, exps[1], 90.0, runtime=0.1)   # same obj, faster
  _write_result(tmp_path, exps[2], 100.0, runtime=0.9)
  assert select_survivors(batch, tmp_path, n=1) == ["faas-powd"]


def _write_real_layout_result(root, suite, exp, obj, runtime=0.1):
  # real ray_dispatcher output tree: <root>/<suite>/<e.id>/outputs/<e.id>/<ts>/obj.csv
  run = root / suite / exp.id / "outputs" / exp.id / "2026-07-06_00-00-00.000000"
  run.mkdir(parents=True)
  (run / "obj.csv").write_text(f"Model\n{obj}\n")
  (run / "runtime.csv").write_text(f"tot\n{runtime}\n")


def test_selects_survivors_from_real_nested_ray_dispatcher_layout(tmp_path):
  suite = "paper-a-screening"
  objs = {
    "hierarchical-madea": 100.0,
    "faas-madea": 99.0, "faas-diffuse": 98.0, "faas-powd": 97.0,
    "faas-br-o": 96.0, "faas-gcaa": 50.0,
  }
  exps = [_exp(a, 50, 1) for a in objs]
  batch = Batch(suite=suite, experiments=tuple(exps))
  for e in exps:
    _write_real_layout_result(tmp_path, suite, e, objs[e.algorithm])

  survivors = select_survivors(batch, tmp_path / suite, n=4)
  assert survivors == ["faas-madea", "faas-diffuse", "faas-powd", "faas-br-o"]
  assert "hierarchical-madea" not in survivors


def test_guard_trips_on_too_many_failures(tmp_path):
  exps = [_exp("faas-madea", 50, 1), _exp("faas-powd", 50, 1)]
  batch = Batch(suite="paper-a-screening", experiments=tuple(exps))
  _write_result(tmp_path, exps[0], 90.0)
  _write_result(tmp_path, exps[1], None)   # failed
  with pytest.raises(SurvivorSelectionError):
    select_survivors(batch, tmp_path, n=1, min_valid_fraction=0.8)
