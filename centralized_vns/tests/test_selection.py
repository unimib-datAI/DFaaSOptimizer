"""Selection must preserve first-index ties, shaking, and float64 ordering."""
import numpy as np
import pytest


@pytest.mark.parametrize('level,gains,nonempty,want', [
  (0, [2.,2.,1.], [True,True,True], 0),
  (0, [2.,2.+1e-13,1.], [True,True,True], 1),
  (3, [-2.,8.,9.], [True,True,True], 0),
  (0, [np.nan,np.inf,-np.inf,3.,2.], [True,True,True,False,True], 4),
  (3, [np.nan,np.inf,-np.inf,3.,-2.], [True,True,True,False,True], 4),
  (0, [np.nan,-np.inf,3.], [True,True,False], -1),
  (3, [np.nan,-np.inf,3.], [True,True,False], -1),
  (0, [-0.,0.], [True,True], 0),
])
def test_select_candidate(level,gains,nonempty,want):
  from centralized_vns.core import select_candidate
  moves=np.zeros((len(gains),4,2),dtype=np.int64)
  moves[:,0,1]=nonempty
  assert select_candidate(moves,gains,level)==want
