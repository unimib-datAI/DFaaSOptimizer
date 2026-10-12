"""Artifact oracle of the runner refactor: capture_baseline.py and compare_artifacts.py."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "harmonization"))

import capture_baseline  # noqa: E402
import compare_artifacts  # noqa: E402
import methods  # noqa: E402

RUN = "fam-n2-s1/2026-01-01_00-00-00.000001"
FILES = {
  "obj.csv": "FaaS-MADiG\n587.7\n581.0\n",
  "termination_condition.csv": (
    ",0\n0,max iterations reached (it: 99; obj. deviation: None; best it: 5; total runtime: 1.41)\n"
    "1,reached time limit: 12.0 >= 10.0 (it: 8; obj. deviation: None; best it: 7; total runtime: 22.5)\n"
  ),
  "runtime.csv": "step,algorithm_s,wall_s,parallel_phases,workers\n0,1.41,7.8,484,0\n1,1.40,7.6,498,0\n",
  "LSP/0/local_processing.csv": "n,f,x\n0,0,3\n1,0,4\n",
  "LSP_solution.csv": "t,n,f,x\n0,0,0,3\n",
  "refinement.csv": "step,sweeps,moves,seconds,reason\n0,2,1,0.03,no improving proposal\n",
  "base_instance_data.json": '{"None": {"Nn": {"None": 2}}}',
  "config.json": json.dumps({"base_solution_folder": "/tmp/a", "seed": 1}),
  "out.log": "t = 0\n",
}


def write_capture(root: Path, files=FILES, method="faas-diffuse", status="ok") -> Path:
  for name, text in files.items():
    path = root / method / RUN / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
  manifest = dict(revision="abc", solver="glpk", seed=1, workers=0, steps=2, instances=["fam-n2-s1"],
                  methods={method: dict(callable="decentralized_diffusion.run", status=status)})
  (root / "manifest.json").write_text(json.dumps(manifest))
  return root


@pytest.fixture
def pair(tmp_path):
  before = write_capture(tmp_path / "before")
  after = tmp_path / "after"
  shutil.copytree(before, after)
  return before, after


def edit(after: Path, name: str, old: str, new: str, run=RUN, method="faas-diffuse"):
  path = after / method / run / name
  text = path.read_text()
  assert old in text
  path.write_text(text.replace(old, new))


def digests(root: Path) -> dict:
  return {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


def test_timing_only_changes_compare_equal(pair):
  before, after = pair
  edit(after, "runtime.csv", "1.41,7.8", "9.99,19.8")
  edit(after, "termination_condition.csv", "total runtime: 1.41", "total runtime: 3.0")
  edit(after, "termination_condition.csv", "12.0 >= 10.0", "13.5 >= 11.0")
  edit(after, "refinement.csv", "0.03", "0.71")
  edit(after, "config.json", "/tmp/a", "/tmp/b")
  moved = after / "faas-diffuse" / "fam-n2-s1"
  (moved / "2026-01-01_00-00-00.000001").rename(moved / "2027-02-02_00-00-00.000002")
  assert compare_artifacts.compare_artifacts(before, after) == []


@pytest.mark.parametrize("name, old, new", [
  ("obj.csv", "581.0", "581.5"),                                       # objective
  ("LSP/0/local_processing.csv", "1,0,4", "1,0,5"),                    # flow
  ("termination_condition.csv", "it: 99", "it: 98"),                   # iteration
  ("termination_condition.csv", "max iterations reached", "no update"),  # reason
  ("refinement.csv", ",2,1,", ",3,1,"),                                # diagnostic non-timing field
  ("config.json", '"seed": 1', '"seed": 2'),                           # algorithm setting
  ("base_instance_data.json", '"Nn": {"None": 2}', '"Nn": {"None": 3}'),  # materialized input
])
def test_decision_changes_fail(pair, name, old, new):
  before, after = pair
  edit(after, name, old, new)
  assert compare_artifacts.compare_artifacts(before, after)


def test_removed_diagnostic_fails(pair):
  before, after = pair
  (after / "faas-diffuse" / RUN / "refinement.csv").unlink()
  assert any("refinement.csv" in m for m in compare_artifacts.compare_artifacts(before, after))


def test_runtime_compared_by_contract(pair):
  before, after = pair
  edit(after, "runtime.csv", "484,0", "485,0")
  assert compare_artifacts.compare_artifacts(before, after)
  edit(after, "runtime.csv", "1,1.40,7.6,498,0\n", "")
  assert compare_artifacts.compare_artifacts(before, after)


def test_historical_tot_runtime_may_become_five_columns(tmp_path):
  before = write_capture(tmp_path / "before", {**FILES, "runtime.csv": "tot\n1.2\n1.3\n"})
  after = write_capture(tmp_path / "after")
  assert compare_artifacts.compare_artifacts(before, after) == []
  assert compare_artifacts.compare_artifacts(after, before)  # never back to an old format


def test_missing_coverage_fails(tmp_path):
  before = write_capture(tmp_path / "before")
  after = write_capture(tmp_path / "after", status="unavailable: glpk not found")
  assert any("coverage" in m for m in compare_artifacts.compare_artifacts(before, after))
  assert any("coverage" in m for m in compare_artifacts.compare_artifacts(before, before, ["MADEA"]))


def test_comparison_leaves_captures_untouched(pair):
  before, after = pair
  edit(after, "obj.csv", "581.0", "581.5")
  snapshot = digests(before.parent)
  compare_artifacts.compare_artifacts(before, after)
  assert digests(before.parent) == snapshot


def test_cli_exit_codes(pair):
  before, after = pair
  assert compare_artifacts.main([str(before), str(after)]) == 0
  edit(after, "obj.csv", "581.0", "581.5")
  assert compare_artifacts.main([str(before), str(after), "--method", "faas-diffuse"]) == 1


@pytest.fixture(scope="module")
def madea_run(tmp_path_factory):
  out = tmp_path_factory.mktemp("madea")
  instance = sorted(p for p in capture_baseline.DATA.iterdir() if (p / "metadata.json").exists())[0]
  config = capture_baseline.config_for(instance, out)
  config["verbose"] = 2
  return Path(methods.HARMONIZED["MADEA"](config, 0, log_on_file=True, disable_plotting=True))


def test_madea_log_compared_through_parser(madea_run, tmp_path):
  instance = madea_run.parent.name
  before, after = (tmp_path / side / "MADEA" / instance / madea_run.name for side in ("b", "a"))
  for side in (before, after):
    shutil.copytree(madea_run, side)
  log = (after / "out.log").read_text()
  timed = log.replace("runtime = ", "runtime = 1", 1)
  assert timed != log
  (after / "out.log").write_text(timed)
  assert compare_artifacts.compare_run(before, after, "MADEA") == []
  lines = log.splitlines(keepends=True)
  index = next(i for i, line in enumerate(lines) if line.startswith("t = 1"))
  lines[index] = "t = 7\n"
  (after / "out.log").write_text("".join(lines))
  assert compare_artifacts.compare_run(before, after, "MADEA")


def test_baseline_runners_are_explicit():
  import decentralized_bestresponse
  import decentralized_potentialgame
  import hierarchical_frankwolfe
  import madea_pg
  runners = capture_baseline.baseline_runners()
  assert set(runners) == set(methods.HARMONIZED) | set(methods.EXCLUDED) | {"centralized"}
  assert runners["FaaS-MAPG-S"] is decentralized_potentialgame.run_pg_s
  assert runners["faas-br-o"] is decentralized_bestresponse.run_br_o
  assert runners["MAFW-PG"] is hierarchical_frankwolfe.run_flat_pg
  assert runners["hierarchical-madea-pg"] is runners["hierarchical-madea-level-cycles-pg"] is madea_pg.run_hierarchical
  assert capture_baseline.ALIASES["hierarchical-madea-level-cycles-pg"] == "hierarchical-madea-pg"
  assert set(methods.EXCLUDED).isdisjoint(methods.HARMONIZED)  # campaign registries untouched


def test_capture_config_has_nonbinding_finite_budgets(tmp_path):
  instance = sorted(p for p in capture_baseline.DATA.iterdir() if (p / "metadata.json").exists())[0]
  config = capture_baseline.capture_config(instance, tmp_path, "faas-br-s", "glpk")
  general = config["solver_options"]["general"]
  assert config["verbose"] == 2 and config["seed"] == 1 and config["max_steps"] == 2
  assert config["solver_name"] == "missing_solver" and general["use_dp"]
  assert 1e4 <= general["algorithm_time_limit"] < float("inf")
  assert 1e4 <= general["TimeLimit"] < float("inf")
  assert 1e2 <= config["solver_options"]["madea_pg"]["time_limit_per_node"] < float("inf")
  central = capture_baseline.capture_config(instance, tmp_path, "centralized", "glpk")
  assert central["solver_name"] == "glpk"
  assert set(central["solver_options"]["general"]) == {"TimeLimit"}
  highs = capture_baseline.capture_config(instance, tmp_path, "centralized")   # default: HiGHS
  assert highs["solver_name"] == "appsi_highs"
  assert highs["solver_options"]["general"] == dict(time_limit=3600, threads=1, random_seed=0, mip_rel_gap=1e-6)


def test_old_welfare_interface_unchanged():
  assert capture_baseline.runners() is methods.HARMONIZED
  instance = sorted(p for p in capture_baseline.DATA.iterdir() if (p / "metadata.json").exists())[0]
  config = capture_baseline.config_for(instance, Path("/tmp/x"))
  assert config["solver_name"] == "missing_solver" and config["max_steps"] == 2
  values = capture_baseline.welfare(methods.HARMONIZED["MADEA"], instance)
  baseline = json.loads((capture_baseline.DATA / "baseline.json").read_text())
  assert values == baseline["MADEA"][instance.name]
