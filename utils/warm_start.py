"""Verified decision artifacts shared by selfish baseline experiments."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pyomo.environ as pyo

from utils.centralized import validate_centralized_solution
from utils.faasmacro import compute_centralized_objective

TOLERANCE = 1e-6
DECISIONS = ('x', 'y', 'z', 'r')


def file_sha256(path):
  digest = hashlib.sha256()
  with Path(path).open('rb') as stream:
    for block in iter(lambda: stream.read(1024**2), b''):
      digest.update(block)
  return digest.hexdigest()


def evaluated_data_and_floors(data):
  """Evaluate central defaults and exact detailed references without constructing a global MIP."""
  from models.selfish import SelfishLoadManagementModel
  model = SelfishLoadManagementModel()
  for component in list(model.model.component_objects()):
    if component.ctype in (pyo.Var, pyo.Constraint, pyo.Expression, pyo.Objective):
      model.model.del_component(component)
  instance = model.generate_instance(data)
  model.compute_local_gain_floors(instance, {'use_dp': True}, 'appsi_highs')
  evaluated = {None: {p.name: p.extract_values() for p in instance.component_objects(pyo.Param)}}
  floors = np.array([pyo.value(instance.minimum_local_gain[n]) for n in instance.N])
  return evaluated, floors


def data_fingerprint(data):
  """Stream canonical evaluated parameters; do not serialize a dense N²F dictionary twice."""
  h = hashlib.sha256()
  parameters = dict(data[None])
  parameters.setdefault('pi', {f:0 for f in range(1, int(parameters['Nf'][None])+1)})
  for name in sorted(parameters):
    if name == 'minimum_local_gain':
      continue
    h.update((name+'\n').encode())
    for index, value in sorted(parameters[name].items(), key=lambda x: str(x[0])):
      if isinstance(value, (int, float, np.number)):
        value = float(value)
      h.update((json.dumps([index, value], separators=(',', ':'), allow_nan=False)+'\n').encode())
  return h.hexdigest()


def validate_decisions(arrays, data, floors=None):
  """Check integer and physical constraints, then the selfish local-processing floors."""
  arrays = {k: np.asarray(arrays[k]) for k in DECISIONS}
  for name, values in arrays.items():
    if values.dtype.kind not in 'buif' or not np.isfinite(values).all():
      raise ValueError(f'{name}: non-finite or nonnumeric decisions')
    if np.any(np.abs(values-np.rint(values)) > TOLERANCE):
      raise ValueError(f'{name}: non-integer decisions')
  validate_centralized_solution(**arrays, data=data, tolerance=TOLERANCE)
  rounded = {k: np.rint(v).astype(np.int64) for k, v in arrays.items()}
  validate_centralized_solution(**rounded, data=data, tolerance=TOLERANCE)
  d = data[None]
  if floors is not None:
    gain = np.array([sum(d['alpha'][n+1, f+1]*rounded['x'][n, f] /
                        (d['incoming_load'][n+1, f+1] or 1)
                        for f in range(rounded['x'].shape[1]))
                     for n in range(rounded['x'].shape[0])])
    if np.any(gain < np.asarray(floors)-TOLERANCE):
      raise ValueError('selfish local-gain floor violated')
  welfare = float(compute_centralized_objective(data, rounded['x'], rounded['y'], rounded['z']))
  if not np.isfinite(welfare):
    raise ValueError('non-finite welfare')
  return rounded, welfare


def load_start(path, data, expected_sha256=None, floors=None, instance_name=None):
  """Load a pickling-free NPZ and require its instance, step and coefficient identity."""
  if expected_sha256 is not None and file_sha256(path) != expected_sha256:
    raise ValueError('warm-start checksum mismatch')
  with np.load(path, allow_pickle=False) as archive:
    arrays = {k: archive[k].copy() for k in DECISIONS}
    metadata = json.loads(str(archive['metadata_json']))
  if metadata.get('step') != 0:
    raise ValueError('warm-start physical step must be zero')
  if instance_name is not None and metadata.get('instance') != instance_name:
    raise ValueError('warm-start instance mismatch')
  if metadata.get('data_sha256') != data_fingerprint(data):
    raise ValueError('warm-start input fingerprint mismatch')
  arrays, welfare = validate_decisions(arrays, data, floors)
  if 'welfare' in metadata and not np.isclose(metadata['welfare'], welfare, atol=1e-6, rtol=1e-10):
    raise ValueError('warm-start recorded welfare mismatch')
  return arrays, {**metadata, 'welfare': welfare, 'sha256': file_sha256(path)}


def save_decisions(path, arrays, metadata):
  """Atomically save complete decisions and derived binary roles as compressed NPZ."""
  target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
  tmp = target.with_name(target.name+'.tmp')
  with tmp.open('wb') as stream:
    np.savez_compressed(stream, **{k: arrays[k] for k in DECISIONS},
      i_sends_f=(arrays['y'].sum(axis=1)>0).astype(np.int8),
      i_receives_f=(arrays['y'].sum(axis=0)>0).astype(np.int8),
      metadata_json=json.dumps(metadata, allow_nan=False))
  tmp.replace(target)


def initialize_variables(instance, arrays):
  """Set all primal variables, including role indicators required by HiGHS starts."""
  for name in DECISIONS:
    if not hasattr(instance, name):
      continue
    component = getattr(instance, name)
    for indices in component:
      component[indices].value = int(arrays[name][tuple(i-1 for i in indices)])
  for name, axis in [('i_sends_f', 1), ('i_receives_f', 0)]:
    if hasattr(instance, name):
      roles = arrays['y'].sum(axis=axis)>0
      for n, f in getattr(instance, name):
        getattr(instance, name)[n, f].value = int(roles[n-1, f-1])


def solution_nodes(arrays, data):
  """Compact per-source traffic/utility records for existing offline fairness metrics."""
  import math
  d=data[None];nn,nf=arrays['x'].shape;nodes=[]
  for n in range(nn):
    loads=[d['incoming_load'][n+1,f+1] for f in range(nf)]
    demand=float(sum(loads));local=float(arrays['x'][n].sum())
    forwarded=float(arrays['y'][n].sum());cloud=float(arrays['z'][n].sum())
    utility=math.fsum((d['alpha'][n+1,f+1]*arrays['x'][n,f]
      +math.fsum(d['beta'].get((n+1,j+1,f+1),0)*float(arrays['y'][n,j,f]) for j in range(nn))
      -d['gamma'][n+1,f+1]*arrays['z'][n,f])/(loads[f] or 1) for f in range(nf))
    nodes.append(dict(step=0,node=n,demand=demand,local=local,forwarded=forwarded,cloud=cloud,
      utility=float(utility),service=(local+forwarded)/demand if demand else None))
  return nodes
