"""A convergence gap must be nonnegative and well-defined at zero welfare."""

import math

import pytest

from utils import faasmacro


@pytest.mark.parametrize("value,reference,expected", [
  (3.0, 2.0, 0.5),
  (-3.0, -2.0, 0.5),
  (-1.0, -1.0, 0.0),
  (0.0, 0.0, 0.0),
  (1.0, 0.0, math.inf),
  (-1.0, 0.0, math.inf),
  (-math.inf, -1.0, math.inf),
  (math.nan, 1.0, math.inf),
])
def test_relative_objective_gap_never_mistakes_undefined_change_for_convergence(
    value, reference, expected,
):
  assert faasmacro.relative_objective_gap(value, reference) == expected
