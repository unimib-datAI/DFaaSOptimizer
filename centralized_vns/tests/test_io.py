import json
import numpy as np
import pytest
from centralized_vns.io import initialize, from_data, load_instance


def data():
  return {None:{'Nn':{None:2},'Nf':{None:1},'incoming_load':{(1,1):3,(2,1):0},
    'demand':{(1,1):1.,(2,1):1.},'max_utilization':{1:1.},
    'memory_requirement':{1:1},'memory_capacity':{1:1,2:2},
    'alpha':{(1,1):4.,(2,1):4.},'gamma':{(1,1):1.,(2,1):1.},
    'beta':{(1,1,1):0.,(1,2,1):2.,(2,1,1):2.,(2,2,1):0.},
    'neighborhood':{(1,2):1,(2,1):1}}}


def test_lsp_initialization_and_global_objective():
  pytest.importorskip('pyomo')
  from utils.faasmacro import compute_centralized_objective
  d=data();x,coverage=initialize(d,'cpu');p=from_data(d,x)
  assert x.tolist()==[[1],[0]]
  y=np.zeros(len(p.src),dtype=np.int64);y[0]=2
  check=p.check(y);dense=np.zeros((2,2,1))
  dense[p.src,p.dst,p.fun]=y
  assert check['welfare']==pytest.approx(compute_centralized_objective(d,x,dense,check['z']))
  assert coverage['cpu_problems']==2


def test_loader_rejects_hash_mismatch(tmp_path):
  (tmp_path/'metadata.json').write_text(json.dumps({'files':{'base_instance_data.json':'bad'}}))
  (tmp_path/'base_instance_data.json').write_text('{}')
  with pytest.raises(ValueError,match='checksum'):
    load_instance(tmp_path,0)


def test_lsp_rounding_boundary_is_accepted():
  pytest.importorskip('pyomo')
  d=data();v=d[None]
  v['incoming_load'][1,1]=7;v['demand'][1,1]=.1;v['max_utilization'][1]=.7
  x,_=initialize(d,'cpu')
  assert x[0,0]==7
  assert from_data(d,x).check(np.zeros(2))['replicas'][0,0]==1


def test_cli_roundtrip(tmp_path):
  pytest.importorskip('pyomo')
  from pathlib import Path
  import subprocess,sys
  root=Path(__file__).resolve().parents[2]
  out=tmp_path/'out'
  args=[sys.executable,'-m','centralized_vns',str(root/'centralized_vns/examples/tiny'),
        '--output',str(out),'--pool-size','2','--candidates','8','--iterations','12']
  subprocess.run(args,cwd=root,check=True,capture_output=True,text=True)
  report=json.loads((out/'result.json').read_text())
  solution=np.load(out/'solution.npz')
  assert report['feasible'] and len(report['input_sha256'])==4
  assert solution['x'].tolist()==[[1],[0]]
  assert report['welfare']==pytest.approx(8/3)
  assert subprocess.run(args,cwd=root,capture_output=True).returncode!=0
