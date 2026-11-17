"""Materialized inputs and the existing LSP_detailed initialization adapter."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
from .core import Problem


def load_instance(folder,step):
  folder=Path(folder)
  if step<0:
    raise ValueError('step must be nonnegative')
  metadata=json.loads((folder/'metadata.json').read_text())
  hashes={}
  required=('base_instance_data.json','load_limits.json','input_requests_traces.json','graph.json')
  for filename in required:
    path=folder/filename
    digest=hashlib.sha256()
    with path.open('rb') as stream:
      for chunk in iter(lambda:stream.read(1024*1024),b''):
        digest.update(chunk)
    hashes[filename]=digest.hexdigest()
    if metadata.get('files',{}).get(filename)!=hashes[filename]:
      raise ValueError(f'checksum mismatch: {filename}')
  raw=json.loads((folder/'base_instance_data.json').read_text())['None']
  data={None:{name:{ast.literal_eval(k):v for k,v in values.items()} for name,values in raw.items()}}
  traces=json.loads((folder/'input_requests_traces.json').read_text())
  n,f=data[None]['Nn'][None],data[None]['Nf'][None]
  try:
    data[None]['incoming_load']={(i+1,g+1):traces[str(g)][str(i)][step]
                                 for i in range(n) for g in range(f)}
  except (KeyError,IndexError) as exc:
    raise ValueError('requested step or node/function trace missing') from exc
  return data,hashes


def initialize(data,device='cpu'):
  from models.dp_solver import try_solve_local
  from models.sp import LSP_detailed
  if device not in ('cpu','cuda'):
    raise ValueError('initialization device must be cpu or cuda')
  n,f=data[None]['Nn'][None],data[None]['Nf'][None]
  locals_=[{None:{**data[None],'whoami':{None:i+1}}} for i in range(n)]
  solutions=[None]*n;coverage={'cpu_problems':0,'cuda_problems':0}
  if device=='cuda':
    from models.local_numeric_gpu import try_solve_local_batch_cuda
    from utils.compute_backend import compute_session
    with compute_session({'use_dp':True,'dp_device':'cuda','compute_device':'cuda'}):
      solutions=try_solve_local_batch_cuda(LSP_detailed(),locals_)
    coverage['cuda_problems']=sum(s is not None for s in solutions)
  for i,s in enumerate(solutions):
    if s is None:
      solutions[i]=try_solve_local(LSP_detailed(),locals_[i])
      coverage['cpu_problems']+=1
    if solutions[i] is None:
      raise ValueError(f'LSP_detailed DP unsupported at node {i+1}; no external-solver fallback')
  return np.array([s['x'] for s in solutions]).reshape(n,f),coverage


def from_data(data,x):
  v=data[None];n,f=v['Nn'][None],v['Nf'][None]
  def matrix(key,default=None):
    return np.array([[v.get(key,{}).get((i+1,g+1),default) for g in range(f)] for i in range(n)],dtype=float)
  demand=matrix('demand')
  if np.any(~np.isfinite(demand)) or np.any(demand<=0):
    raise ValueError('positive finite demand required')
  edges=[(i,j,g) for i in range(n) for g in range(f) for j in range(n)
         if i!=j and v.get('neighborhood',{}).get((i+1,j+1),0)==1]
  if any(a not in (0,1) for a in v.get('neighborhood',{}).values()):
    raise ValueError('neighborhood must be binary')
  beta=[v.get('beta',{}).get((i+1,j+1,g+1),.9) for i,j,g in edges]
  return Problem(load=matrix('incoming_load'),x=x,
    capacity=np.array([v.get('max_utilization',{}).get(g+1,.8) for g in range(f)])/demand,
    memory=[v['memory_requirement'][g+1] for g in range(f)],
    ram=[v['memory_capacity'][i+1] for i in range(n)],
    alpha=matrix('alpha',1.),gamma=matrix('gamma',.1),edges=edges,beta=beta)
