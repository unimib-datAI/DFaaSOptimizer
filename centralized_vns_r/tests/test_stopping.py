"""Stopping policy tests; scripted archive streams isolate stochastic search."""
import json
import subprocess
import sys
import numpy as np
import pytest
from centralized_vns.core import Options,Problem
from centralized_vns_r import core
from centralized_vns_r.tests.test_io import INSTANCE,ROOT


def saturated_problem():
  return Problem(load=[[1]],x=[[1]],capacity=[[1]],memory=[1],ram=[1],
    alpha=[[1]],gamma=[[1]],edges=[],beta=[])


def test_default_plateau_stops_a_saturated_search_and_can_be_disabled():
  p=saturated_problem();o=Options(pool_size=2,candidates=1,iterations=250)
  stopped=core.solve(p,o)
  full=core.solve(p,o,patience=0)
  assert stopped['termination_reason']=='plateau'
  assert stopped['iterations_completed']==200
  assert full['termination_reason']=='iteration_limit'
  assert full['iterations_completed']==250
  assert [r['best_welfare'] for r in stopped['trace']]==[
    r['best_welfare'] for r in full['trace'][:201]]
  np.testing.assert_array_equal(stopped['y'],full['y'])
  assert stopped['welfare']==full['welfare']==1


@pytest.mark.parametrize('patience',[-1,True,1.5,1])
def test_invalid_patience_is_rejected(patience):
  with pytest.raises(ValueError,match='patience'):
    core.solve(saturated_problem(),patience=patience)


@pytest.mark.parametrize('tolerance',[-1,float('nan'),float('inf'),'small',True])
def test_invalid_tolerance_is_rejected(tolerance):
  with pytest.raises(ValueError,match='plateau_tolerance'):
    core.solve(saturated_problem(),plateau_tolerance=tolerance)


def test_iteration_and_time_limits_take_precedence():
  p=saturated_problem()
  for iterations,limit,reason in [(0,300,'iteration_limit'),(4,300,'iteration_limit'),
                                 (10,1e-12,'time_limit')]:
    r=core.solve(p,Options(pool_size=1,candidates=1,iterations=iterations,
      time_limit=limit,shake_after=1),patience=4)
    assert r['termination_reason']==reason
    assert r['iterations_completed']==(0 if limit<1 else iterations)


def test_improving_subordinate_trajectory_prevents_global_plateau(monkeypatch):
  # Global best stays at 100, but trajectory 1 improves through iteration 5.
  def step(self,iteration):
    return np.array([100.,min(iteration+1,5.)])
  monkeypatch.setattr(core.CPU,'step',step)
  r=core.solve(saturated_problem(),Options(pool_size=2,candidates=1,iterations=20,
    shake_after=1),patience=4,plateau_tolerance=0)
  assert r['termination_reason']=='plateau'
  assert r['iterations_completed']==9


def test_shaking_losses_do_not_reset_plateau(monkeypatch):
  from centralized_vns_r.tests.test_moves import relocation_problem
  actual_step=core.CPU.step;values=[]
  def step(self,iteration):
    best=actual_step(self,iteration)
    values.append(self.value.copy())
    return best
  monkeypatch.setattr(core.CPU,'step',step)
  r=core.solve(relocation_problem(),Options(pool_size=2,candidates=16,iterations=80,
    shake_after=1),initial_y=np.array([0,10,10]),patience=8,plateau_tolerance=0)
  assert np.any(np.asarray(values)<r['welfare'])
  assert r['termination_reason']=='plateau'
  assert r['iterations_completed']==8
  np.testing.assert_array_equal(r['y'],[0,10,10])


def test_deadline_reached_in_last_step_is_reported(monkeypatch):
  clock=[0.];actual_step=core.CPU.step
  def step(self,iteration):
    best=actual_step(self,iteration)
    clock[0]=2.
    return best
  monkeypatch.setattr(core.time,'perf_counter',lambda:clock[0])
  monkeypatch.setattr(core.CPU,'step',step)
  r=core.solve(saturated_problem(),Options(pool_size=1,candidates=1,iterations=1,
    time_limit=1.),patience=0)
  assert r['termination_reason']=='time_limit'
  assert r['iterations_completed']==1


def test_cumulative_progress_exceeds_tolerance_despite_small_individual_steps(monkeypatch):
  def step(self,iteration):
    return np.array([1.+.01*min(iteration+1,6)])
  monkeypatch.setattr(core.CPU,'step',step)
  r=core.solve(saturated_problem(),Options(pool_size=1,candidates=1,iterations=20,
    shake_after=1),patience=4,plateau_tolerance=.025)
  # At step 7: (1.06-1.03)/1.03 > .025; at 8: (1.06-1.04)/1.04 < .025.
  assert r['termination_reason']=='plateau'
  assert r['iterations_completed']==8


@pytest.mark.parametrize('base',[-10.,0.])
def test_negative_and_zero_objectives_do_not_cause_premature_stop(monkeypatch,base):
  def step(self,iteration):
    return np.array([base+min(iteration+1,5)*.1])
  monkeypatch.setattr(core.CPU,'step',step)
  p=Problem(load=[[1]],x=[[0]],capacity=[[1]],memory=[1],ram=[0],
    alpha=[[1]],gamma=[[abs(base)]],edges=[],beta=[])
  r=core.solve(p,Options(pool_size=1,candidates=1,iterations=20,
    shake_after=1),patience=4,plateau_tolerance=0)
  assert r['termination_reason']=='plateau'
  assert r['iterations_completed']==9


def test_cli_persists_plateau_settings_and_keeps_both_stages(tmp_path):
  target=tmp_path/'result'
  subprocess.run([sys.executable,'-m','centralized_vns_r',str(INSTANCE),
    '--output',str(target),'--pool-size','2','--candidates','16','--iterations','100',
    '--patience','10','--plateau-tolerance','0','--refine-x','--x-iterations','0'],
    cwd=ROOT,check=True,capture_output=True,text=True)
  fixed=json.loads((target/'fixed_x/result.json').read_text())
  variable=json.loads((target/'variable_x/result.json').read_text())
  assert fixed['termination_reason']=='plateau'
  assert fixed['iterations_completed']<100
  assert fixed['options']['patience']==10
  assert fixed['options']['plateau_tolerance']==0
  assert fixed['plateau']==dict(patience=10,tolerance=0)
  assert fixed['welfare']==variable['welfare']
  with np.load(target/'fixed_x/solution.npz') as a,np.load(target/'variable_x/solution.npz') as b:
    np.testing.assert_array_equal(a['x'],b['x'])
    np.testing.assert_array_equal(a['y'],b['y'])


@pytest.mark.parametrize('flag,value',[('--patience','1'),('--plateau-tolerance','nan')])
def test_invalid_cli_settings_do_not_create_output(tmp_path,flag,value):
  target=tmp_path/'retry'
  result=subprocess.run([sys.executable,'-m','centralized_vns_r',str(INSTANCE),
    '--output',str(target),flag,value],cwd=ROOT,capture_output=True,text=True)
  assert result.returncode!=0
  assert not target.exists()
  assert 'error:' in result.stderr
