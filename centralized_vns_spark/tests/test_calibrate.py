"""Pure orchestration checks; no CUDA kernels or timed workloads run here."""
import json
import numpy as np
import pytest
from centralized_vns.core import Options
from centralized_vns_r.tests.test_moves import relocation_problem
from centralized_vns_spark import calibrate


@pytest.fixture
def hardware(monkeypatch):
  monkeypatch.setattr(calibrate, '_hardware', lambda: dict(name='test GPU', simulator=False))
  monkeypatch.setattr(calibrate, '_free_device_bytes', lambda: 10**9)


def trial(y, candidates=100, search=2., reason='time_limit'):
  return dict(y=np.array(y), welfare=999., candidates_evaluated=candidates,
    elapsed_s=search+50, termination_reason=reason,
    execution=dict(search_elapsed_s=search))


def test_checked_mean_welfare_wins_before_throughput_and_warmup_is_excluded(hardware,monkeypatch):
  p=relocation_problem();calls=[]
  def run(p,o,y,**kwargs):
    calls.append((o,kwargs))
    if o.iterations==1:
      return trial([0,10,10],candidates=10**8)
    return trial([10,0,0] if o.pool_size==1 else [0,10,10],
                 candidates=10**7 if o.pool_size==1 else 10)
  monkeypatch.setattr(calibrate,'solve',run)
  report=calibrate.tune(p,Options(backend='cuda'),pool_sizes=(1,2),candidates=(4,),
    threads=(64,),time_limit=2.,seeds=(3,4))
  assert report['choice']['pool_size']==2
  assert report['choice']['mean_welfare']==p.check([0,10,10])['welfare']
  assert report['choice']['mean_candidates_per_second']==5
  assert len(report['trials'])==2
  assert all(len(t['runs'])==2 for t in report['trials'])
  assert calls[0][0].iterations==1
  assert all(o.time_limit==2. and kw['patience']==0 and kw['time_budget']=='search' for o,kw in calls[1:])
  assert json.loads(json.dumps(report,allow_nan=False))['hardware']['name']=='test GPU'


def test_throughput_breaks_checked_welfare_ties(hardware,monkeypatch):
  monkeypatch.setattr(calibrate,'solve',lambda p,o,y,**kw: trial([10,0,0],o.pool_size*10))
  report=calibrate.tune(relocation_problem(),pool_sizes=(1,2),candidates=(4,),threads=(64,),seeds=(0,))
  assert report['choice']['pool_size']==2


def test_memory_budget_and_free_reserve_guard_before_runner(hardware,monkeypatch):
  monkeypatch.setattr(calibrate,'_free_device_bytes',lambda: 1000)
  monkeypatch.setattr(calibrate,'estimate_device_bytes',lambda p,o,*args: o.pool_size*100)
  calls=[]
  def run(p,o,y,**kw):
    calls.append(o.pool_size)
    return trial([10,0,0])
  monkeypatch.setattr(calibrate,'solve',run)
  report=calibrate.tune(relocation_problem(),Options(backend='cuda',max_device_bytes=850),
    pool_sizes=(8,9,10),candidates=(1,),threads=(64,),seeds=(0,),reserve_fraction=.2)
  assert report['choice']['pool_size']==8
  assert [row['status'] for row in report['trials']]==['ok','skipped_memory','skipped_memory']
  assert calls==[8,8]


def test_failed_validation_cannot_win(hardware,monkeypatch):
  monkeypatch.setattr(calibrate,'solve',lambda p,o,y,**kw: trial([100,0,0]))
  report=calibrate.tune(relocation_problem(),pool_sizes=(1,),candidates=(1,),threads=(64,),seeds=(0,))
  assert report['choice'] is None
  assert report['trials'][0]['status']=='failed'


def test_iteration_capped_trial_is_not_ranked_as_equal_time(hardware,monkeypatch):
  monkeypatch.setattr(calibrate,'solve',lambda p,o,y,**kw: trial([0,10,10],reason='iteration_limit'))
  report=calibrate.tune(relocation_problem(),pool_sizes=(1,),candidates=(1,),threads=(64,),seeds=(0,))
  assert report['choice'] is None


@pytest.mark.parametrize('kwargs',[{'pool_sizes':()},{'pool_sizes':(True,)},
  {'candidates':(0,)},{'threads':(32,)},{'seeds':()},{'seeds':(-1,)},
  {'time_limit':0},{'time_limit':float('nan')},{'time_limit':True},
  {'batch_size':0},{'reserve_fraction':1},{'reserve_fraction':-.1}])
def test_invalid_settings_precede_hardware_initialization(monkeypatch,kwargs):
  monkeypatch.setattr(calibrate,'_hardware',lambda: pytest.fail('hardware initialized'))
  with pytest.raises(ValueError):
    calibrate.tune(relocation_problem(),**kwargs)


def test_simulator_is_never_hardware_evidence(monkeypatch):
  numba=pytest.importorskip('numba')
  monkeypatch.setattr(numba.config,'ENABLE_CUDASIM',1)
  with pytest.raises(RuntimeError,match='simulator'):
    calibrate._hardware()


def test_seed_mean_beats_a_lucky_single_run(hardware,monkeypatch):
  def run(p,o,y,**kw):
    flows=([0,10,10] if o.seed==0 else [0,0,0]) if o.pool_size==1 else [0,5,6]
    return trial(flows)
  monkeypatch.setattr(calibrate,'solve',run)
  report=calibrate.tune(relocation_problem(),pool_sizes=(1,2),candidates=(1,),
    threads=(64,),seeds=(0,1))
  assert report['trials'][0]['mean_welfare']==1.5
  assert report['choice']['pool_size']==2
  assert report['choice']['mean_welfare']==2.


def test_overrun_cannot_win_welfare_by_spending_extra_search_time(hardware,monkeypatch):
  def run(p,o,y,**kw):
    return trial([10,0,0] if o.pool_size==1 else [0,10,10],
      search=2. if o.pool_size==1 else 8.)
  monkeypatch.setattr(calibrate,'solve',run)
  report=calibrate.tune(relocation_problem(),pool_sizes=(1,2),candidates=(1,),
    threads=(64,),seeds=(0,),time_limit=2.)
  assert report['choice']['pool_size']==1
  assert report['trials'][1]['status']=='excluded_time_overrun'
  assert report['trials'][1]['runs'][0]['search_elapsed_s']==8.
  assert report['trials'][1]['exclusion_reason']
  assert report['settings']['max_time_overrun_s']==.2


def test_preflight_reserves_production_history_even_when_trial_plateau_is_disabled(hardware,monkeypatch):
  from dataclasses import replace
  p=relocation_problem();o=Options(backend='cuda',pool_size=1,candidates=1)
  trial_bytes=calibrate.estimate_device_bytes(p,o,0,16)
  monkeypatch.setattr(calibrate,'solve',lambda *a,**kw: pytest.fail('production history did not fit'))
  report=calibrate.tune(p,replace(o,max_device_bytes=trial_bytes),pool_sizes=(1,),
    candidates=(1,),threads=(64,),seeds=(0,),patience=200)
  assert report['choice'] is None
  assert report['trials'][0]['status']=='skipped_memory'
  assert report['trials'][0]['estimated_device_bytes']==trial_bytes+1600
  assert report['settings']['production_patience']==200


@pytest.mark.parametrize('packaged',[True,False])
def test_hardware_versions_include_driver_and_optional_numba_cuda(monkeypatch,packaged):
  from types import SimpleNamespace
  from importlib import metadata
  numba=pytest.importorskip('numba')
  monkeypatch.setattr(numba.config,'ENABLE_CUDASIM',0)
  fake=SimpleNamespace(is_available=lambda:True,
    get_current_device=lambda:SimpleNamespace(name=b'Test',compute_capability=(12,1),
      MULTIPROCESSOR_COUNT=48),
    current_context=lambda:SimpleNamespace(get_memory_info=lambda:(1024,2048)),
    runtime=SimpleNamespace(get_version=lambda:(13,0)),
    cudadrv=SimpleNamespace(driver=SimpleNamespace(get_version=lambda:(13,1))))
  monkeypatch.setattr(numba,'cuda',fake,raising=False)
  def version(name):
    assert name=='numba-cuda'
    if packaged:
      return '0.19.0'
    raise metadata.PackageNotFoundError(name)
  monkeypatch.setattr(metadata,'version',version)
  result=calibrate._hardware()
  assert result['cuda_driver']==[13,1]
  assert result['numba_cuda']==('0.19.0' if packaged else None)
