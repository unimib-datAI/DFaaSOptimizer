"""Device selection must remain driver-free, scoped and main-process only."""
import builtins
import pytest
from utils.compute_backend import compute_device, compute_session, cuda_compute_enabled
from models.local_sp import dp_device


@pytest.mark.parametrize('dp,compute', [('cpu','cpu'),('cuda','cpu'),('cpu','cuda'),('cuda','cuda')])
def test_device_matrix(dp, compute):
  options = dict(dp_device=dp, compute_device=compute, use_dp=dp=='cuda')
  assert dp_device(options) == dp
  assert compute_device(options) == compute
  with compute_session(options):
    assert cuda_compute_enabled() is (compute=='cuda')
  assert not cuda_compute_enabled()


@pytest.mark.parametrize('value', [None, True, 1, '', 'auto', 'CUDA', [], {}])
def test_invalid_device(value):
  with pytest.raises(ValueError, match='compute_device'):
    compute_device({'compute_device': value})


def test_default_nested_and_exception_reset():
  assert compute_device({}) == 'cpu'
  assert not cuda_compute_enabled()
  with pytest.raises(RuntimeError):
    with compute_session({'compute_device':'cuda'}):
      assert cuda_compute_enabled()
      with compute_session({}):
        assert not cuda_compute_enabled()
      assert cuda_compute_enabled()
      raise RuntimeError('fail')
  assert not cuda_compute_enabled()


def test_worker_pid_and_no_cuda_import(monkeypatch):
  import utils.compute_backend as backend
  original = builtins.__import__
  def guarded(name, *args, **kwargs):
    assert not name.startswith(('numba', 'cuda'))
    return original(name, *args, **kwargs)
  monkeypatch.setattr(builtins, '__import__', guarded)
  with compute_session({'compute_device':'cuda'}):
    assert cuda_compute_enabled()
    monkeypatch.setattr(backend.os, 'getpid', lambda: -1)
    assert not cuda_compute_enabled()
  with compute_session({}):
    assert not cuda_compute_enabled()


def test_compute_option_is_application_only():
  from models.model import APPLICATION_OPTIONS
  assert 'compute_device' in APPLICATION_OPTIONS


@pytest.mark.parametrize('solver_name', ['glpk', 'appsi_highs', 'highs', 'gurobi'])
def test_compute_option_filtered_before_external_solver(monkeypatch, solver_name):
  from models import model
  class ReachedSolver(Exception):
    pass
  class Solver:
    options = {'compute_device':'stale'}
    def solve(self, instance, **kwargs):
      assert self.options == {}
      raise ReachedSolver
  monkeypatch.setattr(model, '_SOLVER_CACHE', {})
  monkeypatch.setattr(model.pyo, 'SolverFactory', lambda name: Solver())
  with pytest.raises(ReachedSolver):
    model.BaseAbstractModel().solve(None, {'compute_device':'cuda'}, solver_name)


def test_coverage_is_scoped_and_records_cpu_reasons():
  from utils.compute_backend import record_operation
  with compute_session({'compute_device':'cuda'}) as coverage:
    record_operation('residual_capacity','cpu',30,shape=[30,5],reason='small batch')
    with compute_session({'compute_device':'cuda'}) as inner:
      record_operation('local_numeric','cuda',100,shape=[100,5,20])
      assert len(inner)==1
    assert coverage==[dict(operation='residual_capacity',backend='cpu',count=30,shape=[30,5],reason='small batch')]
  with compute_session({}) as coverage:
    record_operation('ignored','cpu',1)
    assert coverage==[]


@pytest.mark.parametrize('dp,compute',[('cpu','cpu'),('cuda','cpu'),('cpu','cuda'),('cuda','cuda')])
def test_dp_compute_gate_is_scoped_and_driver_free(dp,compute,monkeypatch):
  from utils import compute_backend as backend
  original=builtins.__import__
  def guarded(name,*a,**kw):
    assert not name.startswith(('numba','cuda'))
    return original(name,*a,**kw)
  monkeypatch.setattr(builtins,'__import__',guarded)
  assert not backend.cuda_dp_compute_enabled()
  with pytest.raises(RuntimeError):
    with compute_session(dict(dp_device=dp,compute_device=compute,use_dp=True)):
      assert backend.cuda_dp_compute_enabled() is (dp==compute=='cuda')
      with compute_session({}):assert not backend.cuda_dp_compute_enabled()
      assert backend.cuda_dp_compute_enabled() is (dp==compute=='cuda')
      with monkeypatch.context() as patch:
        patch.setattr(backend.os,'getpid',lambda:-1)
        assert not backend.cuda_dp_compute_enabled()
      raise RuntimeError('reset')
  assert not backend.cuda_dp_compute_enabled()
