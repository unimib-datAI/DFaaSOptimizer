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
