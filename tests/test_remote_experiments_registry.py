import pytest

from remote_experiments.definitions import get_suite, list_suites, register_suite


def test_get_suite_unknown_raises_key_error():
  with pytest.raises(KeyError):
    get_suite("does-not-exist")


def test_register_suite_makes_it_listable():
  @register_suite("__test_registry_suite__")
  def build():
    return []

  assert "__test_registry_suite__" in list_suites()
  assert get_suite("__test_registry_suite__") is build


def test_register_suite_rejects_duplicate_name():
  @register_suite("__test_dup__")
  def build_a():
    return []

  with pytest.raises(ValueError):
    @register_suite("__test_dup__")
    def build_b():
      return []
