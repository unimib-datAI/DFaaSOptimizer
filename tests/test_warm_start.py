"""Real HiGHS starts, including retained feasible solutions after timeout."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/madea_pg_compact'))


def tiny_data():
  return {None: {
    'Nn': {None: 2}, 'Nf': {None: 1},
    'incoming_load': {(1, 1): 2, (2, 1): 1},
    'demand': {(1, 1): 1., (2, 1): 1.},
    'max_utilization': {1: 1.},
    'memory_requirement': {1: 1}, 'memory_capacity': {1: 1, 2: 2},
    'neighborhood': {(1, 2): 1, (2, 1): 1},
  }}


def decisions():
  return dict(x=np.array([[1], [1]]), y=np.array([[[0], [1]], [[0], [0]]]),
              z=np.zeros((2, 1), dtype=int), r=np.array([[1], [2]]))


def make_start(path, data=None, arrays=None, **metadata):
  from utils.warm_start import data_fingerprint, evaluated_data_and_floors
  evaluated, floors = evaluated_data_and_floors(data or tiny_data())
  info = dict(instance='tiny', step=0, data_sha256=data_fingerprint(evaluated),
              algorithm='MADEA-detailed', **metadata)
  np.savez_compressed(path, **(arrays or decisions()), metadata_json=json.dumps(info))
  return evaluated, floors


def fake_instance(monkeypatch, tmp_path):
  import catalog_centralized as catalog
  monkeypatch.setattr(catalog.families, 'load_materialized_instance',
                      lambda path: (tiny_data(), {}, [], None))
  monkeypatch.setattr(catalog, 'get_current_load', lambda traces, agents, t: tiny_data()[None]['incoming_load'])
  (tmp_path/'metadata.json').write_text('{}')
  return catalog


def test_load_start_checks_input_identity_hash_and_floor(tmp_path):
  from utils.warm_start import load_start
  path = tmp_path/'start.npz'
  data, floors = make_start(path)
  start, meta = load_start(path, data, floors=floors, instance_name='tiny',
                           expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
  assert start['y'][0, 1, 0] == 1 and meta['algorithm'] == 'MADEA-detailed'
  with pytest.raises(ValueError, match='checksum'):
    load_start(path, data, expected_sha256='0'*64, floors=floors)
  with pytest.raises(ValueError, match='instance'):
    load_start(path, data, floors=floors, instance_name='different')
  changed = {None: {k: dict(v) for k, v in data[None].items()}}
  changed[None]['incoming_load'][1, 1] = 3
  with pytest.raises(ValueError, match='input'):
    load_start(path, changed, floors=floors)
  bad = decisions(); bad['x'][0, 0] = 0; bad['z'][0, 0] = 1; bad['r'][0, 0] = 0
  make_start(path, arrays=bad)
  with pytest.raises(ValueError, match='floor'):
    load_start(path, data, floors=floors)


def test_central_start_is_sent_to_highs_and_full_solution_saved(monkeypatch, tmp_path):
  from pyomo.contrib.appsi.solvers import Highs
  catalog = fake_instance(monkeypatch, tmp_path)
  start = tmp_path/'start.npz'; make_start(start, instance_unused='x')
  # Identity is the materialized folder name, independent of identical coefficients.
  with np.load(start, allow_pickle=False) as f:
    values = {k: f[k].copy() for k in f.files}
  info = json.loads(str(values['metadata_json'])); info['instance'] = tmp_path.name
  values['metadata_json'] = json.dumps(info); np.savez_compressed(start, **values)
  received = []
  original = Highs._warm_start
  def capture(self):
    values = {str(record[0]):record[0].value for record in self._vars.values()}
    received.append((self.config.warmstart, values['i_sends_f[1,1]'],
                     values['i_receives_f[2,1]']))
    return original(self)
  monkeypatch.setattr(Highs, '_warm_start', capture)
  solution = tmp_path/'solution.npz'
  row = catalog.solve(tmp_path, 5, selfish=True, warm_start=start, solution_file=solution)
  assert row['status'] == 'ok' and row['welfare'] >= 1.95 - 1e-6
  assert row['warm_start']['accepted'] and received == [(True, 1, 1)]
  assert row['solution_source'] == 'solver'
  with np.load(solution, allow_pickle=False) as f:
    assert set(('x','y','z','r','i_sends_f','i_receives_f')) <= set(f.files)
    assert f['y'][0,1,0] == 1 and f['r'][1,0] == 2


def test_exhausted_budget_keeps_validated_seed(monkeypatch, tmp_path):
  catalog = fake_instance(monkeypatch, tmp_path)
  start = tmp_path/'start.npz'
  make_start(start)
  with np.load(start, allow_pickle=False) as f: values = {k:f[k].copy() for k in f.files}
  info=json.loads(str(values['metadata_json']));info['instance']=tmp_path.name
  values['metadata_json']=json.dumps(info);np.savez_compressed(start,**values)
  row = catalog.solve(tmp_path, 0, selfish=True, warm_start=start,
                      solution_file=tmp_path/'solution.npz')
  assert row['status'] == 'ok' and row['feasible']
  assert row['solution_source'] == 'warm_start' and row['welfare'] == pytest.approx(1.95)
  assert row['bound'] is None and row['rel_gap_to_bound'] is None
  assert not row['proven_optimal']


def test_no_start_preserves_no_solution_semantics(monkeypatch, tmp_path):
  catalog = fake_instance(monkeypatch, tmp_path)
  row = catalog.solve(tmp_path, 0, selfish=True)
  assert row['status'] == 'no_solution' and row['welfare'] is None


def test_valid_seed_is_saved_before_global_model_build(monkeypatch, tmp_path):
  from models.selfish import SelfishLoadManagementModel
  catalog = fake_instance(monkeypatch, tmp_path)
  start=tmp_path/'start.npz'; make_start(start)
  with np.load(start,allow_pickle=False) as f: values={k:f[k].copy() for k in f.files}
  info=json.loads(str(values['metadata_json']));info['instance']=tmp_path.name
  values['metadata_json']=json.dumps(info);np.savez_compressed(start,**values)
  solution=tmp_path/'solution.npz'
  original=SelfishLoadManagementModel.generate_instance
  def build(self,data):
    if self.model.component('x') is not None:
      assert solution.is_file(), 'validated seed must survive a killed global build'
      raise RuntimeError('simulated interrupted build')
    return original(self,data)
  monkeypatch.setattr(SelfishLoadManagementModel,'generate_instance',build)
  with pytest.raises(RuntimeError,match='interrupted build'):
    catalog.solve(tmp_path,5,selfish=True,warm_start=start,solution_file=solution)
  with np.load(solution,allow_pickle=False) as saved:
    assert saved['y'][0,1,0] == 1


def test_exported_nodes_preserve_service_and_true_beta_utility():
  from utils.warm_start import solution_nodes,evaluated_data_and_floors
  data,_=evaluated_data_and_floors(tiny_data())
  rows=solution_nodes(decisions(),data)
  assert [r['demand'] for r in rows]==[2,1]
  assert [r['local'] for r in rows]==[1,1]
  assert [r['forwarded'] for r in rows]==[1,0]
  assert [r['service'] for r in rows]==[1,1]
  assert [r['utility'] for r in rows]==pytest.approx([.95,1.])


def test_incomplete_local_reference_is_no_solution_not_uninitialized_error(monkeypatch,tmp_path):
  from models.selfish import SelfishLoadManagementModel
  catalog=fake_instance(monkeypatch,tmp_path)
  def expired(*args,**kwargs):raise TimeoutError('local deadline')
  monkeypatch.setattr(SelfishLoadManagementModel,'compute_local_gain_floors',expired)
  row=catalog.solve(tmp_path,5,selfish=True)
  assert row['status']=='no_solution' and row['welfare'] is None


def test_nonselfish_export_can_be_reused_when_selfish_feasible(monkeypatch,tmp_path):
  from utils.warm_start import evaluated_data_and_floors,load_start
  catalog=fake_instance(monkeypatch,tmp_path)
  target=tmp_path/'unconstrained.npz'
  result=catalog.solve(tmp_path,5,selfish=False,solution_file=target)
  assert result['status']=='ok'
  data,floors=evaluated_data_and_floors(tiny_data())
  arrays,_=load_start(target,data,floors=floors,instance_name=tmp_path.name)
  assert arrays['y'][0,1,0]==1


def test_shared_highs_wrapper_can_change_thread_count_in_same_process(monkeypatch):
  import highspy
  import pyomo.environ as pyo
  from models import model as module
  from models.model import BaseAbstractModel
  monkeypatch.setattr(module,'_SOLVER_CACHE',{})
  highspy.Highs.resetGlobalScheduler(True)
  try:
    for threads in (0,1,2):
      model=pyo.ConcreteModel();model.x=pyo.Var([1,2],domain=pyo.NonNegativeIntegers,bounds=(0,1))
      model.cap=pyo.Constraint(expr=sum(model.x.values())<=1.5)
      model.OBJ=pyo.Objective(expr=sum(model.x.values()),sense=pyo.maximize)
      result=BaseAbstractModel().solve(model,{'threads':threads,'presolve':'off'},'appsi_highs')
      assert result['solution_exists'] and result['obj']==1,(threads,result)
  finally:
    highspy.Highs.resetGlobalScheduler(True)
