import csv
import gzip
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from centralized_vns.io import load_instance


ROOT=Path(__file__).resolve().parents[2]
INSTANCE=ROOT/'centralized_vns/examples/tiny'


def test_standalone_npz_and_native_csv_preserve_the_same_seed(tmp_path):
  from centralized_vns_r.io import load_solution
  data,_=load_instance(INSTANCE,0)
  n,f=data[None]['Nn'][None],data[None]['Nf'][None]
  x=np.zeros((n,f),dtype=np.int64);y=np.zeros((n,n,f),dtype=np.int64)
  np.savez(tmp_path/'seed.npz',x=x,y=y)
  p,flow,_=load_solution(data,tmp_path/'seed.npz')
  checked=p.check(flow)
  native=tmp_path/'native';native.mkdir()
  rows={
    'LSPc_solution.csv':{**{f'n{i}_f{g}_loc':x[i,g] for i in range(n) for g in range(f)},
      **{f'n{i}_f{g}':checked['z'][i,g] for i in range(n) for g in range(f)}},
    'LSPc_replicas.csv':{f'n{i}_f{g}':0 for i in range(n) for g in range(f)},
    'LSPc_detailed_fwd_solution.csv':{f'n{i}_f{g}_n{j}':0 for i in range(n) for g in range(f) for j in range(n) if i!=j}}
  for name,row in rows.items():
    with (native/name).open('w') as file:
      writer=csv.DictWriter(file,fieldnames=row);writer.writeheader();writer.writerow(row)
  q,other,_=load_solution(data,native)
  np.testing.assert_array_equal(q.x,p.x);np.testing.assert_array_equal(other,flow)
  path=native/'LSPc_detailed_fwd_solution.csv';original=path.read_text()
  lines=original.splitlines();names=lines[0].split(',')
  names[1]=names[0].replace('n0_','n00_',1)
  lines[0]=','.join(names);path.write_text('\n'.join(lines)+'\n')
  with pytest.raises(ValueError,match='duplicate'):
    load_solution(data,native)
  path.write_text(original)
  for path in native.iterdir():
    with gzip.open(str(path)+'.gz','wb') as stream:
      stream.write(path.read_bytes())
    path.unlink()
  q,other,_=load_solution(data,native)
  np.testing.assert_array_equal(other,flow)
  command=[sys.executable,'-m','centralized_vns_r.refine',str(INSTANCE),'--solution',str(native),
    '--output',str(tmp_path/'refined-native'),'--iterations','0']
  subprocess.run(command,cwd=ROOT,check=True,capture_output=True,text=True)
  with pytest.raises(ValueError,match='row'):
    load_solution(data,native,row=1)


def test_import_rejects_nonpermitted_flow_and_inconsistent_rejections(tmp_path):
  from centralized_vns_r.io import load_solution
  data,_=load_instance(INSTANCE,0)
  n,f=data[None]['Nn'][None],data[None]['Nf'][None]
  x=np.zeros((n,f));y=np.zeros((n,n,f));y[0,0,0]=1
  np.savez(tmp_path/'bad.npz',x=x,y=y)
  with pytest.raises(ValueError,match='permitted'):
    load_solution(data,tmp_path/'bad.npz')
  y[:]=0
  np.savez(tmp_path/'bad.npz',x=x,y=y,z=np.full((n,f),-1))
  with pytest.raises(ValueError):
    load_solution(data,tmp_path/'bad.npz')


def test_sparse_edge_mapping_is_explicit_and_validated(tmp_path):
  from centralized_vns_r.io import load_solution
  data,_=load_instance(INSTANCE,0)
  n,f=data[None]['Nn'][None],data[None]['Nf'][None]
  np.savez(tmp_path/'seed.npz',x=np.zeros((n,f)),y=np.zeros(1))
  with pytest.raises(ValueError,match='edges'):
    load_solution(data,tmp_path/'seed.npz')


def test_sparse_import_reorders_edges_and_keeps_nonzero_flows(tmp_path):
  from centralized_vns.io import from_data
  from centralized_vns_r.io import load_solution
  data,_=load_instance(INSTANCE,0)
  n,f=data[None]['Nn'][None],data[None]['Nf'][None]
  p=from_data(data,np.zeros((n,f)))
  y=np.zeros(len(p.edges),dtype=int)
  for e in range(len(y)):
    trial=y.copy();trial[e]=1
    try:p.check(trial)
    except ValueError:continue
    y=trial;break
  assert y.any()
  np.savez(tmp_path/'seed.npz',x=p.x,edges=p.edges[::-1],y=y[::-1])
  _,loaded,_=load_solution(data,tmp_path/'seed.npz')
  np.testing.assert_array_equal(loaded,y)


def test_both_stages_and_standalone_cli_keep_original_artifacts(tmp_path):
  target=tmp_path/'both'
  args=[sys.executable,'-m','centralized_vns_r',str(INSTANCE),'--output',str(target),
    '--pool-size','2','--candidates','16','--iterations','10','--refine-x','--x-iterations','20']
  subprocess.run(args,cwd=ROOT,check=True,capture_output=True,text=True)
  fixed=target/'fixed_x/solution.npz';original=fixed.read_bytes()
  a=json.loads((target/'fixed_x/result.json').read_text())
  b=json.loads((target/'variable_x/result.json').read_text())
  assert a['algorithm']=='VNS-R' and b['algorithm']=='VNS-RX'
  assert b['welfare']>=a['welfare']-1e-12
  assert all(g>=f-1e-12 for g,f in zip(b['local_gain'],b['local_gain_floor']))
  separate=tmp_path/'separate'
  cmd=[sys.executable,'-m','centralized_vns_r.refine',str(INSTANCE),'--solution',str(fixed),
    '--output',str(separate),'--iterations','0']
  subprocess.run(cmd,cwd=ROOT,check=True,capture_output=True,text=True)
  assert fixed.read_bytes()==original
  with np.load(fixed) as x,np.load(separate/'solution.npz') as y:
    np.testing.assert_array_equal(x['x'],y['x']);np.testing.assert_array_equal(x['y'],y['y'])
  assert subprocess.run(cmd,cwd=ROOT,capture_output=True).returncode!=0
