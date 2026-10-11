"""PG orchestration must preserve time for refinement and leave non-PG unchanged."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from hierarchical_auction import madea_cycles_runner as runner
from hierarchical_auction.iterative_engine import IterativeHierarchicalAuctionEngine
from postprocessing import load_solution
from utils.centralized import encode_solution, validate_centralized_solution
from test_review_distributed_regressions import _materialized_config, _two_node_data


def test_hierarchical_pg_reserves_requested_time_before_starting_an_auction(tmp_path, monkeypatch):
  data = _two_node_data()
  data[None]['incoming_load'] = {(1, 1): 5, (2, 1): 0}
  data[None]['memory_capacity'][1] = 0
  config = _materialized_config(tmp_path, data)
  config['solver_options'] = {'general': {'algorithm_time_limit': 30}, 'madea_pg': {'time_limit': 10}}
  clock = [0.]
  solve = runner.solve_subproblem
  def delayed(*args, **kwargs):
    result = solve(*args, **kwargs)
    result[-1]['tot'] = 20.
    clock[0] = 20.
    return result
  monkeypatch.setattr(runner, 'solve_subproblem', delayed)
  from utils import step_time
  monkeypatch.setattr(step_time, 'perf_counter', lambda: clock[0])
  folder = Path(runner._run(config, 0, True, True,
                          engine_class=IterativeHierarchicalAuctionEngine,
                          result_name='HierarchicalPG', refine_welfare=True))
  pg = pd.read_csv(folder / 'refinement.csv').iloc[0]
  # The local solve used all 20 seconds available to MADEA/hierarchy.
  # With no local service or existing seller replica, its incumbent rejects all five requests.
  assert pg.welfare_before == pytest.approx(-0.1)
  assert pg.welfare_after == pytest.approx(-0.1)
  assert pg.time_budget == 10.
  sol, replicas, details, *_ = load_solution(str(folder), 'LSPc')
  arrays = encode_solution(2, 1, sol, details, replicas, 0)
  validate_centralized_solution(*arrays[:4], data)


@pytest.mark.parametrize('refine,pg_budget,sweeps', [(False, 10, 5), (True, 0, 5),
                                                   (True, 10, 0), (True, 10, 5)])
def test_every_variant_stops_on_the_modelled_budget(tmp_path, monkeypatch, refine, pg_budget, sweeps):
  data = _two_node_data()
  data[None]['incoming_load'] = {(1, 1): 5, (2, 1): 0}
  data[None]['memory_capacity'][1] = 0
  config = _materialized_config(tmp_path, data)
  config['solver_options'] = {'general': {'algorithm_time_limit': 30},
                             'madea_pg': {'time_limit': pg_budget, 'max_sweeps': sweeps}}
  clock = [0.]
  solve = runner.solve_subproblem
  def delayed(*args, **kwargs):
    result = solve(*args, **kwargs)
    clock[0] = 40.
    return result
  monkeypatch.setattr(runner, 'solve_subproblem', delayed)
  from utils import step_time
  monkeypatch.setattr(step_time, 'perf_counter', lambda: clock[0])
  folder = Path(runner._run(config, 0, True, True,
                          engine_class=IterativeHierarchicalAuctionEngine,
                          result_name='HierarchicalPG', refine_welfare=refine))
  sol, replicas, details, *_ = load_solution(str(folder), 'LSPc')
  arrays = encode_solution(2, 1, sol, details, replicas, 0)
  validate_centralized_solution(*arrays[:4], data)
  # A 40 s local solve exceeds the 30 s modelled budget: no variant starts an auction.
  np.testing.assert_array_equal(arrays[1][:, :, 0], [[0, 0], [0, 0]])
  assert pd.read_csv(folder / 'runtime.csv')['algorithm_s'].iloc[0] >= 40.
  if not refine:
    assert not (folder / 'refinement.csv').exists()
  else:
    pg = pd.read_csv(folder / 'refinement.csv').iloc[0]
    assert pg.time_budget == 0. and pg.moves == 0


def test_pg_fractional_auction_budget_keeps_glpk_solver_limit_integer(tmp_path, monkeypatch):
  import run_faasmadea as madea
  from models.sp import LSPr_x
  class FallbackRestricted(LSPr_x):
    """Force the real MILP solver outside the exact-type direct backend."""
  monkeypatch.setattr(madea, 'LSPr_x', FallbackRestricted)
  data = _two_node_data()
  data[None]['incoming_load'] = {(1, 1): 5, (2, 1): 0}
  data[None]['memory_capacity'][1] = 0
  config = _materialized_config(tmp_path, data)
  config['solver_options'] = {'general': {'TimeLimit': 30}, 'madea_pg': {'time_limit': .25}}
  folder = Path(runner._run(config, 0, True, True,
                          engine_class=IterativeHierarchicalAuctionEngine,
                          result_name='HierarchicalPG', refine_welfare=True))
  sol, replicas, details, *_ = load_solution(str(folder), 'LSPc')
  arrays = encode_solution(2, 1, sol, details, replicas, 0)
  validate_centralized_solution(*arrays[:4], data)
  assert arrays[1].sum() == 5
