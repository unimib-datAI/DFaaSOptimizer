from copy import deepcopy
import os
import numpy as np
import pytest
from models import sp
from models.dp_solver import try_solve_local
from models.local_numeric_gpu import try_solve_local_batch_cuda
from test_gpu_local_sp import data

pytestmark = pytest.mark.skipif(os.environ.get('DFAAS_TEST_CUDA') != '1', reason='real CUDA opt-in required')
NAMES = ['LSP','LSP_detailed','LSP_v0','LSP_fixedr','LSP_fixedr_v0',
         'LSP_capped','LSP_capped_fixedr','LSP_pg','LSP_pg_fixedr',
         'LSPr','LSPr_v0','LSPr_x','LSPr_fixedr']


def fixtures():
  d=data();v=d[None]
  v.update(neighborhood={(n,m):int(n!=m) for n in (1,2,3) for m in (1,2,3)},
           beta={(n,m,f):float(m+f) for n in (1,2,3) for m in (1,2,3) for f in (1,2)},
           y_bar={(m,n,f):0. for n in (1,2,3) for m in (1,2,3) for f in (1,2)},
           omega_bar={(n,f):0. for n in (1,2,3) for f in (1,2)},
           x_bar={(n,f):1 for n in (1,2,3) for f in (1,2)},
           r_bar={(n,f):1 for n in (1,2,3) for f in (1,2)})
  result=[]
  for n in (3,1,2):
    local=deepcopy(d);local[None]['whoami']={None:n}
    result.append(local)
  return result


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('epsilon',[0.,-1e-9,1e-9])
def test_full_whitelist_exact_allocations(name,epsilon):
  model=getattr(sp,name)();batch=fixtures()
  for i,d in enumerate(batch):
    v=d[None];n=v['whoami'][None]
    v['demand'][(n,1)]=.5+epsilon
    v['pi']={1:float(i),2:float(2-i)}
  original=deepcopy(batch)
  expected=[try_solve_local(model,d) for d in batch]
  actual=try_solve_local_batch_cuda(model,batch)
  assert len(actual)==len(expected)
  for a,b in zip(actual,expected):
    if b is None:
      assert a is None
    else:
      assert a is not None
      assert {k:v for k,v in a.items() if k!='runtime'} == {k:v for k,v in b.items() if k!='runtime'}
  assert batch==original


def test_unsupported_mixed_inputs_and_fresh_values():
  model=sp.LSP();batch=fixtures()
  batch[1][None]['memory_capacity'][1]=1.5
  for changed in (False,True):
    if changed:
      batch[0][None]['incoming_load'][(3,1)]=0
      batch[2][None]['alpha'][(2,1)]=0
    actual=try_solve_local_batch_cuda(model,batch)
    assert actual[1] is None
    for i in (0,2):
      expected=try_solve_local(model,batch[i])
      assert {k:v for k,v in actual[i].items() if k!='runtime'}=={k:v for k,v in expected.items() if k!='runtime'}


def test_shared_dispatch_configuration_and_full_phase():
  from models.local_sp import solve_agent_batch
  from utils.compute_backend import compute_session
  from utils import step_time
  from experiments.gpu_dp.benchmark import expanded_data
  model=sp.LSP();d=expanded_data(fixtures()[0],30)
  batch=[]
  for agent in range(30):
    local=deepcopy(d);local[None]['whoami']={None:agent+1};batch.append(local)
  expected=[try_solve_local(model,item) for item in batch]
  options={'use_dp':True,'dp_device':'cuda','compute_device':'cuda'}
  clock=step_time.start_step()
  with compute_session(options), step_time.parallel_phase() as phase:
    phase.uniform(100)
    result=solve_agent_batch(d,range(30),model,options,'missing_solver')
    assert phase.shared_device
  row=step_time.end_step(clock)
  assert row['algorithm_s']>0 and row['workers']==0
  for a,b in zip(result.values(),expected):
    assert {k:v for k,v in a.items() if not k.startswith('runtime')}=={k:v for k,v in b.items() if k!='runtime'}


def test_numeric_cuda_failure_is_visible(monkeypatch):
  import models.local_numeric_gpu as module
  def broken(items):
    raise RuntimeError('CUDA probe failure')
  monkeypatch.setattr(module,'_run_chunk',broken)
  with pytest.raises(RuntimeError,match='CUDA probe failure'):
    try_solve_local_batch_cuda(sp.LSP(),fixtures())


def test_direct_dispatch_performance_gate():
  from models.local_sp import compute_batch_enabled
  from utils.compute_backend import compute_session
  with compute_session({'use_dp':True,'dp_device':'cuda','compute_device':'cuda'}):
    assert not compute_batch_enabled(sp.LSPr_x(),{'use_dp':True,'dp_device':'cuda'})
    assert not compute_batch_enabled(sp.LSP_fixedr(),{'use_dp':True,'dp_device':'cuda'})


def test_small_new_curve_batch_uses_existing_dispatch(monkeypatch):
  from models import local_numeric_gpu
  from models.local_sp import solve_agent_batch
  from utils.compute_backend import compute_session
  def forbidden(*args):
    raise AssertionError('small new numeric batch')
  monkeypatch.setattr(local_numeric_gpu,'try_solve_local_batch_cuda',forbidden)
  options={'use_dp':True,'dp_device':'cuda','compute_device':'cuda'}
  with compute_session(options):
    assert len(solve_agent_batch(fixtures()[0],[2,0,1],sp.LSP(),options,'missing_solver'))==3
