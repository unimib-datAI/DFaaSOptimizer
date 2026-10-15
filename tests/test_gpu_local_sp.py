"""GPU dispatch preserves per-agent inputs and CPU semantics."""
from copy import deepcopy
import numpy as np
import pytest
from models import local_sp, dp_gpu
from models.sp import LSP, LSPr_x
from plasma.core.sbm import minimize_replica_costs
from run_faasmacro import solve_subproblem, compute_social_welfare, parallel_solver_session
from test_gpu_dp import real_gpu

OPTIONS = {'use_dp': True, 'dp_device': 'cuda'}

def data():
  from test_local_sp import _data
  d = _data()
  v = d[None]
  v['Nn'] = {None: 3}
  for key in ('incoming_load', 'demand', 'alpha', 'delta', 'gamma'):
    v[key] = {(n, f): val for (_, f), val in v[key].items() for n in (1, 2, 3)}
  v['memory_capacity'] = {1: 4, 2: 5, 3: 7}
  return d

@pytest.fixture
def fake_gpu(monkeypatch):
  calls = []
  def batch(cs, ms, bs):
    calls.append(len(cs))
    out = []
    for c, m, b in zip(cs, ms, bs):
      try: out.append(minimize_replica_costs(c, m, b))
      except ValueError: out.append(None)
    return out
  monkeypatch.setattr(dp_gpu, 'minimize_replica_costs_batch', batch)
  return calls

@pytest.mark.parametrize('parallelism', [0, 2, -1])
def test_common_dispatch_bypasses_pool_and_preserves_solutions(monkeypatch, fake_gpu, parallelism):
  import run_faasmacro as runner
  def no_pool(*a, **k): raise AssertionError('GPU must not call pool')
  monkeypatch.setattr(runner, '_solve_agents_parallel', no_pool)
  d = data()
  original = deepcopy(d)
  prices = np.array([[1, 2], [4, 1], [3, 0.]])
  cpu = solve_subproblem(d, [2, 0], LSP(), 'missing_solver', {'use_dp': True}, 0, detailed_pi=prices)
  gpu = solve_subproblem(d, dict.fromkeys([2, 0]).keys(), LSP(), 'missing_solver', OPTIONS, parallelism, detailed_pi=prices)
  for i in range(1, 8): np.testing.assert_array_equal(gpu[i], cpu[i])
  assert gpu[8:10] == cpu[8:10]
  assert fake_gpu == [2]
  assert d == original
  # Both ordinary restricted DP and direct fixed-x welfare calls stay equivalent.
  from models.sp import LSPr
  for model, fixed in ((LSPr(), None), (LSPr_x(), np.zeros((3, 2)))):
    args = (model, d, [2, 0], 'missing_solver')
    tail = (np.zeros((3, 3, 2)), np.zeros((3, 2)))
    expected = compute_social_welfare(*args, {'use_dp': True}, *tail, 0, fixed)
    actual = compute_social_welfare(*args, OPTIONS, *tail, parallelism, fixed)
    for a, b in zip(actual[0], expected[0]): np.testing.assert_array_equal(a, b)
    assert actual[1:3] == expected[1:3]


def test_batch_order_fresh_inputs_and_residual_policy(fake_gpu, monkeypatch):
  d = data()
  original = deepcopy(d)
  opts = dict(OPTIONS)
  prices = np.array([[0, 0], [1, 2], [3, 0.]])
  for change in (False, True):
    if change:
      d[None]['incoming_load'][3, 1] = 8
      d[None]['memory_capacity'][1] = 9
      prices += 1
    result = local_sp.solve_agent_batch(d, [2, 0], LSP(), opts, 'missing_solver', prices)
    assert list(result) == [2, 0]
    for agent in (2, 0):
      single = deepcopy(d)
      single[None].update(whoami={None: agent + 1}, pi={f+1: p for f, p in enumerate(prices[agent])})
      expected = local_sp.solve_agent_problem(LSP(), single, {'use_dp': True}, 'missing_solver')
      assert {k: v for k, v in result[agent].items() if not k.startswith('runtime')} == {k: v for k, v in expected.items() if k != 'runtime'}
  assert opts == OPTIONS
  assert original != d  # only the explicit changes above
  monkeypatch.setattr(dp_gpu, 'gpu_problem_fits', lambda *a: False)
  result = local_sp.solve_agent_batch(d, [2, 0], LSP(), opts, 'missing_solver', prices)
  assert all(r['solution_exists'] for r in result.values())
  assert fake_gpu == [2, 2]


def test_singleton_and_direct_do_not_call_gpu(monkeypatch):
  def fail(*a): raise AssertionError('no eligible batch')
  monkeypatch.setattr(dp_gpu, 'minimize_replica_costs_batch', fail)
  assert local_sp.solve_agent_batch(data(), [0], LSP(), OPTIONS, 'missing_solver')[0]['solution_exists']
  from models.sp import LSP_fixedr
  assert len(local_sp.solve_agent_batch(data(), [0, 2], LSP_fixedr(), OPTIONS, 'missing_solver')) == 2

@pytest.mark.parametrize('error', [RuntimeError('CUDA driver unavailable'), MemoryError('CUDA OOM'), ImportError('numba absent')])
def test_cuda_errors_propagate(monkeypatch, error):
  def fail(*a): raise error
  monkeypatch.setattr(dp_gpu, 'minimize_replica_costs_batch', fail)
  with pytest.raises(type(error), match=str(error)):
    local_sp.solve_agent_batch(data(), [0, 1], LSP(), OPTIONS, 'missing_solver')


def test_infeasible_and_unsupported_keep_dp_only_failure(fake_gpu):
  from models.sp import LSP_v0
  d = data()
  d[None]['delta'] = {(n, f): 0 for n in (1, 2, 3) for f in (1, 2)}
  d[None]['whoami'] = {None: 1}
  # Unsupported subclass retains the configured solver, including missing_solver.
  class Custom(LSP): pass
  with pytest.raises(RuntimeError, match='unavailable'):
    local_sp.solve_agent_batch(d, [0, 1], Custom(), OPTIONS, 'missing_solver')
  # Huge DP retains the solver path rather than silently bypassing the size threshold.
  d[None]['memory_capacity'][1] = 10_000_000
  with pytest.raises(RuntimeError, match='unavailable'):
    local_sp.solve_agent_batch(d, [0, 1], LSP(), OPTIONS, 'missing_solver')

@real_gpu
def test_real_gpu_after_warm_cpu_pool():
  d = data()
  with parallel_solver_session():
    cpu = solve_subproblem(d, [0, 1, 2], LSP(), 'missing_solver', {'use_dp': True}, 2)
    gpu = solve_subproblem(d, [0, 1, 2], LSP(), 'missing_solver', OPTIONS, 2)
  for i in range(1, 8): np.testing.assert_array_equal(gpu[i], cpu[i])
  assert gpu[8:10] == cpu[8:10]


def test_mixed_supported_and_unsupported_nodes(fake_gpu):
  class FallbackLSP:
    pass
  model = LSP()
  d = data()
  d[None]['demand'][2, 1] = 0  # unsupported locally, needs configured Pyomo solver
  expected = {'solution_exists': False, 'runtime': 0., 'termination_condition': 'infeasible'}
  model.solve = lambda *args, **kwargs: dict(expected)
  result = local_sp.solve_agent_batch(d, [2, 1, 0], model, OPTIONS, 'missing_solver')
  assert list(result) == [2, 1, 0]
  assert result[2]['solution_exists'] and result[0]['solution_exists']
  assert result[1]['termination_condition'] == 'infeasible'
  assert fake_gpu == [2]


def test_gpu_infeasible_keeps_configured_solver_policy(fake_gpu):
  from models.sp import LSPr
  d = data()
  d[None].update(y_bar={(m,n,f): (1. if n == 1 else 0.) for m in (1,2,3) for n in (1,2,3) for f in (1,2)},
                 omega_bar={(n,f): 0. for n in (1,2,3) for f in (1,2)})
  d[None]['memory_capacity'][1] = 0
  with pytest.raises(RuntimeError, match='unavailable'):
    local_sp.solve_agent_batch(d, [0, 1, 2], LSPr(), OPTIONS, 'missing_solver')
  assert fake_gpu == [3]


@pytest.mark.parametrize('count', [3, 500])
def test_batch_latency_counts_preparation_chunks_decode_once(monkeypatch, count):
  from models import dp_solver
  from utils import step_time
  import run_faasmacro as macro
  from test_step_time import FakeClock
  now = FakeClock()
  monkeypatch.setattr(step_time, 'perf_counter', now)
  d = data()
  v = d[None]
  v['Nn'] = {None: count}
  for key in ('incoming_load', 'demand', 'alpha', 'delta', 'gamma'):
    v[key] = {(n, f): val for (node, f), val in v[key].items() if node == 1 for n in range(1, count+1)}
  v['memory_capacity'] = {n: 4 for n in range(1, count+1)}
  original_prepare, original_finish = dp_solver.prepare_local_problem, dp_solver.finish_local_solution
  def prepare(*args):
    now.t += .01 / count
    return original_prepare(*args)
  def finish(*args, **kwargs):
    result = original_finish(*args, **kwargs)
    now.t += .01 / count
    return result
  calls = []
  def batch(cs, ms, bs):
    # Initial context/JIT costs count in the first call; two synchronized chunks.
    now.t += .1 if not calls else 0
    calls.append(1)
    now.t += .01
    result = [minimize_replica_costs(c, m, b) for c, m, b in zip(cs, ms, bs)]
    now.t += .01
    return result
  monkeypatch.setattr(dp_solver, 'prepare_local_problem', prepare)
  monkeypatch.setattr(dp_solver, 'finish_local_solution', finish)
  monkeypatch.setattr(dp_gpu, 'minimize_replica_costs_batch', batch)
  for expected in (.14, .04):
    clock = step_time.start_step()
    result = macro.solve_subproblem(d, range(count), LSP(), 'missing_solver', OPTIONS, -1)
    row = step_time.end_step(clock)
    assert row['algorithm_s'] == pytest.approx(expected)
    assert row['workers'] == 0
    assert result[-1]['tot'] == pytest.approx(expected)
  result = local_sp.solve_agent_batch(d, range(count), LSP(), OPTIONS, 'missing_solver')
  assert all(r['runtime_kind'] == 'batch_latency' and r['runtime'] == pytest.approx(.04) for r in result.values())


@real_gpu
@pytest.mark.parametrize('name', ['LSP', 'LSP_detailed', 'LSP_v0', 'LSP_fixedr',
  'LSP_fixedr_v0', 'LSP_capped', 'LSP_capped_fixedr', 'LSP_pg', 'LSP_pg_fixedr',
  'LSPr', 'LSPr_v0', 'LSPr_x', 'LSPr_fixedr'])
def test_real_gpu_all_local_models_and_pyomo_constraints(name):
  import pyomo.environ as pyo
  from models import sp
  d = data()
  v = d[None]
  v.update(neighborhood={(n,m): int(n != m) for n in (1,2,3) for m in (1,2,3)},
           beta={(n,m,f): float(m+f) for n in (1,2,3) for m in (1,2,3) for f in (1,2)},
           y_bar={(m,n,f): 0. for n in (1,2,3) for m in (1,2,3) for f in (1,2)},
           omega_bar={(n,f): 0. for n in (1,2,3) for f in (1,2)},
           x_bar={(n,f): 1 for n in (1,2,3) for f in (1,2)},
           r_bar={(n,f): 1 for n in (1,2,3) for f in (1,2)})
  model = getattr(sp, name)()
  actual = local_sp.solve_agent_batch(d, [2, 0], model, OPTIONS, 'missing_solver')
  for agent in (2, 0):
    single = deepcopy(d)
    single[None]['whoami'] = {None: agent + 1}
    expected = local_sp.solve_agent_problem(model, single, {'use_dp': True}, 'missing_solver')
    assert {k: val for k, val in actual[agent].items() if not k.startswith('runtime')} == {k: val for k, val in expected.items() if k != 'runtime'}
    instance = model.generate_instance(single)
    for key in ('x', 'y', 'r', 'z', 'omega'):
      if key in actual[agent]:
        for index, val in zip(getattr(instance, key), actual[agent][key]):
          assert val in getattr(instance, key)[index].domain
          getattr(instance, key)[index].set_value(val)
    for constraint in instance.component_data_objects(pyo.Constraint, active=True):
      value = pyo.value(constraint.body)
      assert constraint.lower is None or value >= pyo.value(constraint.lower) - 1e-9
      assert constraint.upper is None or value <= pyo.value(constraint.upper) + 1e-9
    assert pyo.value(instance.OBJ) == pytest.approx(actual[agent]['obj'], abs=1e-9)


def test_logging_is_excluded_from_batch_latency(monkeypatch, fake_gpu):
  from utils import step_time
  from test_step_time import FakeClock
  now = FakeClock()
  monkeypatch.setattr(step_time, 'perf_counter', now)
  def slow_log(*args):
    now.t += .25
  monkeypatch.setattr(local_sp.log, 'info', slow_log)
  clock = step_time.start_step()
  with step_time.parallel_phase():
    results = local_sp.solve_agent_batch(data(), [0, 2], LSP(), OPTIONS, 'missing_solver')
  row = step_time.end_step(clock)
  assert row['algorithm_s'] == 0
  assert row['wall_s'] == .25
  assert all(r['runtime'] == 0 for r in results.values())
