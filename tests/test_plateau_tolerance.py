"""plateau_tolerance: stop once the best objective improved less than it, in all, over the patience window
(the default rule needs every single iteration below `tolerance`)."""
import csv
import sys
import tempfile
from collections import deque
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments" / "harmonization"), str(ROOT)]

from run_faasmadea import check_stopping_criteria  # noqa: E402


def check(odevs, patience=3, **kw):
  ones = np.ones((2, 2))
  return check_stopping_criteria(5, 100, ones, ones, ones, deque(odevs, maxlen=patience), None, None, None,
                                 1e-6, 0.0, float("inf"), **kw)


def test_default_rule_is_unchanged():
  assert check([0.0, 0.0, 0.0]) == (True, "UB/LB diff < tol")
  assert check([5e-6, 5e-6, 5e-6])[0] is False      # each step above 1e-6: goes on


def test_plateau_tolerance_bounds_the_improvement_over_the_window():
  stop, why = check([5e-6, 5e-6, 5e-6], plateau_tolerance=1e-4)  # 1.5e-5 in all
  assert stop and why == "best improved < 0.0001 over 3 iterations"
  assert check([5e-5, 5e-5, 5e-5], plateau_tolerance=1e-4)[0] is False  # 1.5e-4 in all
  assert check([0.0], plateau_tolerance=1e-4)[0] is False              # window not full yet


def run_method(name, **overrides):
  from capture_baseline import DATA, config_for
  import methods
  instance = next(p for p in sorted(DATA.iterdir()) if (p / "metadata.json").exists())
  config = config_for(instance, Path(tempfile.mkdtemp()), steps=1)
  config.update(overrides)
  methods.HARMONIZED[name](config, 2, log_on_file=True, disable_plotting=True)


# every module calling the shared stopping check
CALLERS = ("run_faasmadea", "decentralized_auction", "hierarchical_auction.madea_runner",
           "hierarchical_auction.madea_cycles_runner", "hierarchical_auction.madea_level_cycles_runner")
# the methods whose plateau is the improvement of the best objective
PLATEAU_METHODS = ("MADEA", "MADEA2", "MADEA2-b", "MADEA-one-shot", "MADEA2-one-shot", "MADEA2-b-one-shot",
                   "hierarchical-madea", "hierarchical-madea-cycles", "hierarchical-madea-legacy")


@pytest.mark.parametrize("name", PLATEAU_METHODS)
def test_every_plateau_rule_takes_plateau_tolerance_from_the_config(monkeypatch, name):
  import importlib
  import run_faasmadea
  seen = []
  original = run_faasmadea.check_stopping_criteria

  def recording(*args, **kwargs):
    queue = kwargs.get("odev_queue", args[5] if len(args) > 5 else None)
    if queue is not None:
      seen.append(kwargs.get("plateau_tolerance"))
    return original(*args, **kwargs)

  for module in CALLERS:
    mod = importlib.import_module(module)
    if hasattr(mod, "check_stopping_criteria"):
      monkeypatch.setattr(mod, "check_stopping_criteria", recording)
  run_method(name, plateau_tolerance=1e-4)
  assert seen and set(seen) == {1e-4}
