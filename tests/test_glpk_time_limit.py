import pytest

from models.model import _solver_option_name, _solver_option_value


@pytest.mark.parametrize("value,expected", [(30.0, 30), (0.75, 1), (40.2, 41), (12, 12)])
def test_glpk_time_limit_is_whole_seconds(value, expected):
  assert _solver_option_name("glpk", "TimeLimit") == "tmlim"
  assert _solver_option_value("glpk", "TimeLimit", value) == expected


def test_other_solvers_and_options_unchanged():
  assert _solver_option_value("gurobi", "TimeLimit", 30.5) == 30.5
  assert _solver_option_value("glpk", "mipgap", 1e-9) == 1e-9
