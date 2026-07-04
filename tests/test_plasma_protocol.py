import dataclasses

import pytest

from plasma.core.types import LOCAL, REJ, Heartbeat, PlasmaOptions


def test_target_columns():
  assert LOCAL == 0
  assert REJ == 1


def test_options_defaults():
  opts = PlasmaOptions()
  assert opts.W == 1.0
  assert opts.k_sb == 10
  assert opts.mu == 0.1
  assert opts.p_commit == 0.5
  assert opts.n_hyst == 2
  assert opts.staleness_rounds == 3
  assert opts.rare_function_mode == "sampled"


def test_options_from_config_overrides():
  config = {"solver_options": {"plasma": {"mu": 0.2, "k_sb": 5}}}
  opts = PlasmaOptions.from_config(config)
  assert opts.mu == 0.2
  assert opts.k_sb == 5
  assert opts.W == 1.0


def test_options_frozen():
  with pytest.raises(dataclasses.FrozenInstanceError):
    PlasmaOptions().mu = 0.5


def test_options_rejects_execution_mode():
  config = {"solver_options": {"plasma": {"execution_mode": "async"}}}
  with pytest.raises(TypeError):
    PlasmaOptions.from_config(config)
