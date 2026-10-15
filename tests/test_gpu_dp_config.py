"""Device selection must stay explicit and must not touch CUDA on CPU."""
import builtins
import pytest
from models import local_sp

@pytest.mark.parametrize('options,expected', [({}, 'cpu'), ({'dp_device': 'cpu'}, 'cpu'), ({'use_dp': True, 'dp_device': 'cuda'}, 'cuda')])
def test_device_selection(options, expected):
  assert local_sp.dp_device(options) == expected

@pytest.mark.parametrize('options', [{'dp_device': v, 'use_dp': True} for v in ('auto', None, True, False)] + [{'dp_device': 'cuda'}, {'dp_device': 'cuda', 'use_dp': False}])
def test_invalid_device(options):
  with pytest.raises(ValueError):
    local_sp.dp_device(options)

def test_cpu_does_not_import_cuda(monkeypatch):
  from test_dp_solver import _data
  from models.sp import LSP
  original = builtins.__import__
  def guarded(name, *args, **kwargs):
    if name.startswith(('numba', 'cuda', 'models.dp_gpu')):
      raise AssertionError(name)
    return original(name, *args, **kwargs)
  monkeypatch.setattr(builtins, '__import__', guarded)
  assert local_sp.solve_agent_problem(LSP(), _data(), {'use_dp': True}, 'missing_solver')['solution_exists']

def test_application_option_filtered_with_cached_solver(monkeypatch):
  from models import model
  from types import SimpleNamespace
  class ReachedSolver(Exception): pass
  class Solver:
    options = {'dp_device': 'stale'}
    def solve(self, instance, **kwargs):
      assert self.options == {'some_option': 7}
      raise ReachedSolver
  solver = Solver()
  monkeypatch.setattr(model.pyo, 'SolverFactory', lambda _: solver)
  monkeypatch.setattr(model, '_SOLVER_CACHE', {})
  m = model.BaseAbstractModel()
  for _ in range(2):
    with pytest.raises(ReachedSolver):
      m.solve(None, {'dp_device': 'cuda', 'use_dp': True, 'some_option': 7}, 'fake')
