"""One coordinator solve with historical or protected detailed local decisions."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/madea_pg_compact'))
from test_warm_start import tiny_data, decisions


def bind_instance(monkeypatch,tmp_path,data):
  import selfish_distributed as sd
  monkeypatch.setattr(sd,'load_materialized_instance',lambda path:(data,{},list(range(data[None]['Nn'][None])),None))
  monkeypatch.setattr(sd,'get_current_load',lambda traces,agents,t:data[None]['incoming_load'])
  (tmp_path/'metadata.json').write_text('{}')
  return sd


def seed_for(path,name,data):
  from utils.warm_start import evaluated_data_and_floors,data_fingerprint,save_decisions
  evaluated,_=evaluated_data_and_floors(data)
  save_decisions(path,decisions(),{'instance':name,'step':0,'data_sha256':data_fingerprint(evaluated),'algorithm':'MADEA-detailed','welfare':1.95})


def test_projection_uses_extra_replicas_and_respects_new_omega_bounds():
  from selfish_distributed import coordinator_start
  from utils.warm_start import evaluated_data_and_floors
  data,_=evaluated_data_and_floors(tiny_data())
  local=dict(x=np.array([[1],[1]]),r=np.array([[1],[1]]),omega=np.array([[1],[0]]))
  projected=coordinator_start(decisions(),data,**local)
  np.testing.assert_array_equal(projected['r'],[[0],[1]])
  assert projected['y'][0,1,0]==1
  local['omega'][:]=0
  repaired=coordinator_start(decisions(),data,**local)
  assert repaired['y'].sum()==0 and repaired['r'].sum()==0


def test_detailed_keeps_local_x_and_labels_coordinator_bound(monkeypatch,tmp_path):
  sd=bind_instance(monkeypatch,tmp_path,tiny_data())
  seed=tmp_path/'seed.npz';seed_for(seed,tmp_path.name,tiny_data())
  target=tmp_path/'solution.npz'
  row=sd.solve(tmp_path,5,threads=1,detailed=True,warm_start=seed,solution_file=target)
  assert row['status']=='ok' and row['welfare']==pytest.approx(1.95)
  assert row['warm_start']['accepted']
  assert row['bound_scope']=='coordinator' and row['proven_optimal_scope']=='coordinator'
  assert row['local_reference']=='LSP_detailed' and row['restricted_model']=='LSPr_x'
  with np.load(target,allow_pickle=False) as result:
    np.testing.assert_array_equal(result['x'],[[1],[1]])
    assert result['r'][1,0]==2


def test_detailed_uses_central_defaults_while_historical_keeps_native_model(monkeypatch,tmp_path):
  data={None:{'Nn':{None:2},'Nf':{None:2},
    'incoming_load':{(1,1):1,(1,2):10,(2,1):0,(2,2):0},
    'demand':{(n,f):1. for n in (1,2) for f in (1,2)},
    'max_utilization':{1:1.,2:1.},'memory_requirement':{1:1,2:1},
    'memory_capacity':{1:1,2:0},'neighborhood':{},
    'alpha':{(1,1):.1,(1,2):2.,(2,1):1.,(2,2):1.}}}
  sd=bind_instance(monkeypatch,tmp_path,data)
  for detailed,expected in [(True,[[1,0],[0,0]]),(False,[[0,1],[0,0]])]:
    target=tmp_path/f'{detailed}.npz'
    row=sd.solve(tmp_path,5,threads=1,detailed=detailed,solution_file=target)
    assert row['status']=='ok',row
    with np.load(target,allow_pickle=False) as result:
      np.testing.assert_array_equal(result['x'],expected)


def test_zero_budget_retains_adapted_feasible_start(monkeypatch,tmp_path):
  sd=bind_instance(monkeypatch,tmp_path,tiny_data())
  seed=tmp_path/'seed.npz';seed_for(seed,tmp_path.name,tiny_data())
  row=sd.solve(tmp_path,0,threads=1,detailed=True,warm_start=seed)
  assert row['status']=='ok' and row['welfare']==pytest.approx(1.95)
  assert row['solution_source']=='warm_start' and not row['proven_optimal']
  assert row['bound'] is None


def test_zero_coordinator_objective_has_zero_optimal_gap(monkeypatch,tmp_path):
  data=tiny_data();data[None]['memory_capacity']={1:3,2:4}
  sd=bind_instance(monkeypatch,tmp_path,data)
  row=sd.solve(tmp_path,5,threads=1,detailed=True)
  assert row['status']=='ok' and row['bound']==0
  assert row['rel_gap_to_bound']==0


@pytest.mark.parametrize('adapter',['centralized','distributed'])
def test_solver_releases_global_highs_thread_scheduler(monkeypatch,tmp_path,adapter):
  import pyomo.environ as pyo
  from models import model as module
  from models.model import BaseAbstractModel
  if adapter=='distributed':
    solve=bind_instance(monkeypatch,tmp_path,tiny_data()).solve
  else:
    from test_warm_start import fake_instance
    solve=fake_instance(monkeypatch,tmp_path).solve
  assert solve(tmp_path,5,threads=2)['status']=='ok'
  monkeypatch.setattr(module,'_SOLVER_CACHE',{})
  m=pyo.ConcreteModel();m.x=pyo.Var(bounds=(0,1),domain=pyo.NonNegativeIntegers)
  m.OBJ=pyo.Objective(expr=m.x,sense=pyo.maximize)
  import highspy
  try:
    row=BaseAbstractModel().solve(m,{'threads':1},'appsi_highs')
    assert row['solution_exists'] and row['obj']==1,row
  finally:
    highspy.Highs.resetGlobalScheduler(True)


def test_spare_coordinator_replicas_are_tightened_before_global_validation(monkeypatch,tmp_path):
  data=tiny_data();data[None]['memory_capacity'][2]=5
  sd=bind_instance(monkeypatch,tmp_path,data)
  output=tmp_path/'solution.npz'
  row=sd.solve(tmp_path,5,threads=1,detailed=True,solution_file=output)
  assert row['welfare']==pytest.approx(1.95) and row['solution_source']=='solver',row
  with np.load(output,allow_pickle=False) as artifact:
    assert artifact['r'][1,0]==2


def test_rejected_solver_incumbent_gap_uses_returned_solution(monkeypatch,tmp_path):
  sd=bind_instance(monkeypatch,tmp_path,tiny_data())
  original=sd._check_coordinator;calls=[]
  def reject_loaded(instance):
    original(instance);calls.append(1)
    if len(calls)>1:raise ValueError('simulated invalid solver incumbent')
  monkeypatch.setattr(sd,'_check_coordinator',reject_loaded)
  row=sd.solve(tmp_path,5,threads=1,detailed=True)
  assert row['solution_source']=='local_only' and row['welfare']==pytest.approx(1.1)
  assert row['rel_gap_to_bound']==pytest.approx((.45-(-.4))/.45),row
  assert row['incumbent_coordinator_objective']==pytest.approx(-.4)
  assert not row['proven_optimal']


def test_detailed_local_prices_match_the_exact_central_reference(monkeypatch,tmp_path):
  data={None:{'Nn':{None:2},'Nf':{None:2},
    'incoming_load':{(1,1):1,(1,2):10,(2,1):0,(2,2):0},
    'demand':{(n,f):1. for n in (1,2) for f in (1,2)},
    'max_utilization':{1:1.,2:1.},'memory_requirement':{1:1,2:1},
    'memory_capacity':{1:1,2:1},'neighborhood':{(1,2):1,(2,1):1},
    'alpha':{(1,1):.5,(1,2):2.,(2,1):1.,(2,2):1.},
    'beta':{(1,2,1):.4,(1,2,2):.9},'pi':{1:1.,2:0.}}}
  sd=bind_instance(monkeypatch,tmp_path,data)
  target=tmp_path/'solution.npz'
  row=sd.solve(tmp_path,0,threads=1,detailed=True,solution_file=target)
  assert row['status']=='ok'
  with np.load(target,allow_pickle=False) as saved:
    np.testing.assert_array_equal(saved['x'][0],[1,0])
