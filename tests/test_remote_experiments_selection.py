import pytest

from remote_experiments.manifest import Manifest
from remote_experiments.selection import default_selection, parse_selection


def test_parse_selection_all_returns_every_index():
  assert parse_selection("all", 5) == [0, 1, 2, 3, 4]


def test_parse_selection_empty_returns_every_index():
  assert parse_selection("", 5) == [0, 1, 2, 3, 4]


def test_parse_selection_comma_list():
  assert parse_selection("0,2,4", 5) == [0, 2, 4]


def test_parse_selection_range():
  assert parse_selection("1-3", 5) == [1, 2, 3]


def test_parse_selection_mixed():
  assert parse_selection("0, 2-3", 5) == [0, 2, 3]


def test_parse_selection_out_of_range_raises():
  with pytest.raises(ValueError):
    parse_selection("9", 5)


def test_default_selection_skips_succeeded(tmp_path):
  manifest = Manifest(tmp_path / "m.json")
  manifest.record("e1", status="succeeded")
  assert default_selection(["e1", "e2", "e3"], manifest) == [1, 2]
