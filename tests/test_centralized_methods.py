"""methods.CENTRALIZED: the centralized references offered to campaigns (catalog_centralized.solve)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments" / "madea_pg_compact"), str(ROOT)]

import methods  # noqa: E402
from catalog_centralized import solve  # noqa: E402

INSTANCE = next(p for p in sorted((ROOT / "tests/data/harmonization").iterdir())
                if (p / "metadata.json").exists())


def test_centralized_methods_are_named_apart_from_the_harmonized_ones():
  assert set(methods.CENTRALIZED) == {"centralized", "selfish-centralized"}
  assert not set(methods.CENTRALIZED) & set(methods.HARMONIZED)


def test_selfish_keeps_local_gains_and_never_beats_the_unconstrained_optimum():
  rows = {name: solve(INSTANCE, 120, **kwargs) for name, kwargs in methods.CENTRALIZED.items()}
  assert rows["centralized"]["model"] == "LoadManagementModel" and rows["centralized"]["status"] == "ok"
  selfish = rows["selfish-centralized"]
  assert selfish["model"] == "SelfishLoadManagementModel" and selfish["status"] == "ok"
  assert selfish["local_reference_seconds"] >= 0
  # the local floors only remove solutions: never more welfare than the unconstrained optimum
  assert 0 < selfish["welfare"] <= rows["centralized"]["welfare"] + 1e-6
  assert selfish["mip_rel_gap"] == 1e-6  # the default


def test_the_gap_is_handed_to_highs_and_recorded():
  assert solve(INSTANCE, 120, gap=0.25)["mip_rel_gap"] == 0.25


def test_an_earlier_highs_solve_does_not_break_the_reference():
  # HiGHS keeps one thread scheduler per process: a solve asking for another thread count failed
  # ("global scheduler has already been initialized"), e.g. after a local floor fell back to HiGHS
  import pyomo.environ as pyo
  from models.model import BaseAbstractModel

  class Tiny(BaseAbstractModel):
    def __init__(self):
      super().__init__()
      self.model.x = pyo.Var([1], within=pyo.NonNegativeIntegers, bounds=(0, 5))
      self.set_objective(lambda m: m.x[1], pyo.maximize)

  tiny = Tiny()
  tiny.solve(tiny.generate_instance({}), {}, "appsi_highs")
  for threads in (1, 2):
    assert solve(INSTANCE, 120, threads=threads)["status"] == "ok"
