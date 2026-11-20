from dataclasses import replace
from itertools import product

import numpy as np
import pytest
from centralized_vns.core import Problem
from centralized_vns_r.bounds import welfare_bound
from centralized_vns_r.xstage import local_gains


@pytest.mark.parametrize('variable',[False,True])
def test_lp_bounds_every_feasible_integer_solution(variable):
  p=Problem(load=[[2],[2]],x=[[1],[0]],capacity=[[2],[2]],memory=[1],ram=[1,1],
    alpha=[[4],[1]],gamma=[[1],[2]],edges=[(0,1,0),(1,0,0)],beta=[3,6])
  upper=welfare_bound(p,[0,0],variable)['upper']
  values=[]
  for plan in product(range(3),repeat=2):
    x=np.array(plan).reshape(2,1)
    if not variable and not np.array_equal(x,p.x): continue
    if np.any(local_gains(p,x)<local_gains(p,p.x)): continue
    q=replace(p,x=x)
    for y in product(range(3),repeat=2):
      try: values.append(q.check(y)['welfare'])
      except ValueError: continue
  assert max(values)<=upper+1e-9
  assert upper == pytest.approx(7.0 if variable else 5.5)
  integer=welfare_bound(p,[0,0],variable,mip_seconds=10)
  assert integer['mip_upper']==pytest.approx(max(values))
