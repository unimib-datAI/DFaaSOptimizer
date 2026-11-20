"""Fixed-x certificate and CUDA branch restrictions (CUDA simulator supported)."""
import numpy as np
import pytest

sc = pytest.importorskip('pyscipopt')
cuda = pytest.importorskip('numba.cuda')
pytestmark = pytest.mark.skipif(not cuda.is_available(), reason='CUDA or its simulator required')

from test_selfish_distributed import reusable_seed, tiny_data
from centralized_vns_scip import coordinator
from centralized_vns_scip.cuda import NodePool
from centralized_vns.core import Options
from utils.warm_start import file_sha256


def test_coordinator_certificate_and_seed(tmp_path,monkeypatch):
  data=tiny_data()
  data[None]['gamma']={(1,1):.8,(2,1):.8}
  seed=tmp_path/'seed.npz'
  reusable_seed(seed,'tiny',data)
  monkeypatch.setattr(coordinator,'load_instance',lambda path,step:(data,{}))
  result=coordinator.solve(tmp_path,10,1,.001,tmp_path/'solver.log',tmp_path/'sol.npz',
                           seed,file_sha256(seed),'tiny')
  assert result['bound_scope']=='coordinator' and result['gap_target_met']
  assert result['bound']==pytest.approx(.45)
  assert result['welfare']==pytest.approx(1.95)


def test_gpu_rejects_out_of_box_moves():
  from centralized_vns.io import from_data
  p=from_data(tiny_data(),[[1],[1]])
  pool=NodePool(p,Options(backend='cuda',pool_size=1,candidates=8))
  zero=np.zeros(len(p.src),dtype=np.int64)
  pool.lo.copy_to_device(zero)
  pool.hi.copy_to_device(zero)
  for it in range(3):
    pool.advance(it,1)
  np.testing.assert_array_equal(pool.result(),zero)
  assert not np.isfinite(pool.gains.copy_to_host()).any()
  pool.hi.copy_to_device((p.load-p.x)[p.src,p.fun])
  pool.advance(3,1)
  assert pool.result().sum()>0


@pytest.mark.parametrize("with_rx", [False, True])
def test_combined_start_reuses_bound_on_fixed_x_objective(tmp_path, monkeypatch, with_rx):
  import json
  data = tiny_data()
  data[None]['gamma'] = {(1,1):.8, (2,1):.8}
  seed = tmp_path/'original.npz'
  reusable_seed(seed,'tiny',data)
  with np.load(seed) as s:
    payload = dict(s)
  idx=np.argwhere(payload['y'])
  record=dict(instance='tiny',source='new_coordinator',objective=.45,upper=.4501,welfare=1.95)
  merged=tmp_path/'combined.npz'
  np.savez_compressed(merged, **payload, scip_indices=idx, scip_values=payload['y'][tuple(idx.T)],
                      scip_record=json.dumps(record))
  monkeypatch.setattr(coordinator,'load_instance',lambda path,step:(data,{}))
  def no_model(*args):
    raise AssertionError('target already certified')
  monkeypatch.setattr(coordinator,'Model',no_model)
  row=coordinator.solve(tmp_path,10,1,.001,tmp_path/'log',tmp_path/'sol.npz',
                        merged,file_sha256(merged),'tiny',refine_local=with_rx)
  if with_rx:
    assert row['variable_x']['attempted']>0
    assert row['welfare']>=row['fixed_x']['welfare']-1e-9
    with np.load(tmp_path/'fixed_x/solution.npz') as fixed,np.load(tmp_path/'sol.npz') as final:
      np.testing.assert_array_equal(fixed['x'],payload['x'])
      assert set(final.files)>= {'x','y','z','r'}
    assert 'bound' not in row['variable_x']
    row=row['fixed_x']
  assert row['termination']=='archived_gap' and row['gap_target_met'] and not row['proven_optimal']
  assert row['initial_coordinator_objective']==pytest.approx(.45)
  assert row['bound']==.4501 and row['nodes_processed']==0
