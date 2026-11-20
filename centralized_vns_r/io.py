"""Checked interchange for standalone refinement and separate stage artifacts."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from centralized_vns.io import from_data
from .xstage import local_gains


def digest(path):
  h=hashlib.sha256()
  with Path(path).open('rb') as stream:
    for block in iter(lambda:stream.read(1024*1024),b''):
      h.update(block)
  return h.hexdigest()


def integers(values,name):
  a=np.asarray(values)
  if np.iscomplexobj(a):
    raise ValueError(f'{name}: real integers required')
  a=np.asarray(a,dtype=np.float64)
  if (not np.all(np.isfinite(a)) or np.any(a<0) or np.any(a>2**52) or
      np.any(np.abs(a-np.rint(a))>1e-6)):
    raise ValueError(f'{name}: finite nonnegative integers required (tolerance 1e-6)')
  return np.rint(a).astype(np.int64)


def load_solution(data,source,row=0):
  source=Path(source)
  if type(row) is not int or row<0:
    raise ValueError('solution row must be a nonnegative integer')
  hashes={}
  if source.is_file():
    hashes[source.name]=digest(source)
    with np.load(source,allow_pickle=False) as archive:
      arrays={k:archive[k] for k in ('x','y','z','replicas','r','edges') if k in archive}
    if 'x' not in arrays or 'y' not in arrays:
      raise ValueError('solution NPZ requires x and y')
  else:
    def read(name):
      path=source/name
      if not path.is_file():
        path=path.with_suffix(path.suffix+'.gz')
      hashes[path.name]=digest(path)
      opener=gzip.open if path.suffix=='.gz' else open
      with opener(path,'rt',newline='') as stream:
        reader=csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames)!=len(set(reader.fieldnames)):
          raise ValueError('invalid CSV columns')
        for index,record in enumerate(reader):
          if index==row:
            if None in record or any(v is None for v in record.values()):
              raise ValueError('malformed CSV row')
            return record
      raise ValueError('solution row missing')
    n,f=data[None]['Nn'][None],data[None]['Nf'][None]
    solution=read('LSPc_solution.csv');replicas=read('LSPc_replicas.csv')
    def matrix(record,suffix=''):
      return [[float(record[f'n{i}_f{g}{suffix}']) for g in range(f)] for i in range(n)]
    arrays=dict(x=matrix(solution,'_loc'),z=matrix(solution),replicas=matrix(replicas))
    flows=read('LSPc_detailed_fwd_solution.csv')
    total=any(k.endswith('_tot') for k in flows)
    pattern=re.compile(r'n(\d+)_f(\d+)_n(\d+)'+('_tot' if total else ''))
    y=np.zeros((n,n,f));seen=np.zeros((n,n,f),dtype=bool);count=0
    for name,value in flows.items():
      if total and name.endswith('_accepted'):
        continue
      match=pattern.fullmatch(name)
      if not match:
        raise ValueError(f'unrecognized flow column: {name}')
      i,g,j=map(int,match.groups())
      if i>=n or j>=n or g>=f or i==j:
        raise ValueError(f'invalid flow column: {name}')
      if seen[i,j,g]:
        raise ValueError(f'duplicate flow column: {name}')
      seen[i,j,g]=True;y[i,j,g]=float(value);count+=1
    if count!=n*(n-1)*f:
      raise ValueError('missing flow columns')
    arrays['y']=y
  p=from_data(data,integers(arrays['x'],'x'))
  incoming=integers(arrays['y'],'y');n,f=p.load.shape
  if incoming.shape==(n,n,f):
    permitted=np.zeros_like(incoming,dtype=bool)
    permitted[p.src,p.dst,p.fun]=True
    if np.any(incoming[~permitted]):
      raise ValueError('flow on a non-permitted edge')
    y=incoming[p.src,p.dst,p.fun]
  elif incoming.ndim==1:
    if 'edges' not in arrays:
      raise ValueError('sparse y requires explicit edges')
    edges=integers(arrays['edges'],'edges')
    if edges.shape!=(len(incoming),3) or len(set(map(tuple,edges)))!=len(edges):
      raise ValueError('invalid sparse edges')
    lookup={tuple(e):i for i,e in enumerate(p.edges)};y=np.zeros(len(p.src),dtype=np.int64)
    for edge,value in zip(edges,incoming):
      if tuple(edge) not in lookup:
        raise ValueError('flow on a non-permitted edge')
      y[lookup[tuple(edge)]]=value
  else:
    raise ValueError('y must be a sparse vector or source/destination/function tensor')
  checked=p.check(y)
  if 'z' in arrays and not np.array_equal(integers(arrays['z'],'z'),checked['z']):
    raise ValueError('inconsistent rejections')
  for name in ('replicas','r'):
    if name in arrays:
      replicas=integers(arrays[name],name)
      if (replicas.shape!=p.load.shape or np.any(replicas<checked['replicas']) or
          np.any(replicas.astype(float)@p.memory>p.ram)):
        raise ValueError('inconsistent replica allocation')
  return p,y,dict(solution_source=str(source.resolve()),solution_row=row,
    solution_sha256=hashes,integer_import_tolerance=1e-6,
    floor_reference='input_solution_x',replicas='minimal feasible replicas recomputed')


def save_result(folder,p,result,metadata):
  folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
  x=result['x'];checked=from_data_check(p,x,result['y'])
  floor=np.asarray(result.get('local_gain_floor',local_gains(p,p.x)))
  gain=local_gains(p,x)
  if np.any(gain<floor-1e-12):
    raise ValueError('local gain floor violated')
  report={k:v for k,v in result.items() if not isinstance(v,np.ndarray)}
  report.update(metadata,feasible=True,welfare=checked['welfare'],
    local_gain_floor=floor.tolist(),local_gain=gain.tolist(),local_floor_satisfied=True)
  np.savez_compressed(folder/'solution.npz',x=x,edges=p.edges,y=result['y'],
    z=checked['z'],replicas=checked['replicas'],local_gain_floor=floor,local_gain=gain)
  (folder/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
  return report


def from_data_check(p,x,y):
  from dataclasses import replace
  return replace(p,x=x).check(y)
