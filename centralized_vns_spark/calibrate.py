"""Real-CUDA calibration by checked welfare under equal search-time budgets."""
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib import metadata
from itertools import product
import math
import platform
from statistics import mean

import numpy as np
from centralized_vns.core import Options
from .gpu import estimate_device_bytes, solve, validate_execution


def _hardware():
  import numba
  if numba.config.ENABLE_CUDASIM:
    raise RuntimeError('CUDA simulator cannot provide hardware calibration evidence')
  from numba import cuda
  if not cuda.is_available():
    raise RuntimeError('CUDA unavailable; calibration does not fall back to CPU')
  device=cuda.get_current_device()
  free,total=cuda.current_context().get_memory_info()
  name=device.name.decode() if isinstance(device.name,bytes) else str(device.name)
  try:
    numba_cuda=metadata.version('numba-cuda')
  except metadata.PackageNotFoundError:
    numba_cuda=None
  return dict(name=name,compute_capability=list(device.compute_capability),
    multiprocessors=int(device.MULTIPROCESSOR_COUNT),total_memory_bytes=int(total),
    initial_free_memory_bytes=int(free),numba=numba.__version__,numpy=np.__version__,
    cuda_runtime=list(cuda.runtime.get_version()),cuda_driver=list(cuda.cudadrv.driver.get_version()),
    numba_cuda=numba_cuda,python=platform.python_version(),
    platform=platform.platform(),simulator=False)


def _free_device_bytes():
  from numba import cuda
  return int(cuda.current_context().get_memory_info()[0])


def tune(p,options=None,initial_y=None,*,pool_sizes=(64,128,256,512,1024),
         candidates=(256,512,1024,2048),threads=(64,128,256),time_limit=2.,
         seeds=(4850,4851),batch_size=16,reserve_fraction=.1,patience=200):
  """Return JSON-ready trial evidence and a choice; never claim simulated speed.

  Compilation is warmed once per block width before measured runs. Plateau is
  disabled and the iteration ceiling is deliberately unreachable in normal use.
  The deadline is checked between batches; actual search duration is recorded.
  Memory preflight includes the requested production plateau history.
  """
  o=options or Options(backend='cuda')
  o.validate()
  for name,values in (('pool_sizes',pool_sizes),('candidates',candidates),
                      ('threads',threads),('seeds',seeds)):
    if not values or any(type(v) is not int for v in values):
      raise ValueError(f'{name} must contain integers')
  if any(v<1 for v in (*pool_sizes,*candidates)):
    raise ValueError('pool_sizes and candidates must be positive')
  if any(v not in (64,128,256) for v in threads):
    raise ValueError('threads must be 64, 128 or 256')
  if any(not 0<=v<2147483646 for v in seeds):
    raise ValueError('seeds must be in [0, 2147483646)')
  if isinstance(time_limit,bool) or not math.isfinite(time_limit) or time_limit<=0:
    raise ValueError('time_limit must be positive and finite')
  if isinstance(reserve_fraction,bool) or not math.isfinite(reserve_fraction) or not 0<=reserve_fraction<1:
    raise ValueError('reserve_fraction must be finite and in [0, 1)')
  validate_execution(o,patience,0.,'all',batch_size,threads[0])
  max_overrun=max(.05,.1*time_limit)
  y=np.zeros(len(p.src),dtype=np.int64) if initial_y is None else np.asarray(initial_y).copy()
  initial=p.check(y)
  report=dict(created_at=datetime.now(timezone.utc).isoformat(),hardware=_hardware(),
    problem=dict(nodes=p.load.shape[0],functions=p.load.shape[1],edges=len(p.src),
      initial_welfare=initial['welfare']),
    settings=dict(options=asdict(o),pool_sizes=list(pool_sizes),candidates=list(candidates),
      threads=list(threads),seeds=list(seeds),search_time_limit_s=time_limit,
      max_time_overrun_s=max_overrun,
      batch_size=batch_size,reserve_fraction=reserve_fraction,patience=0,
      production_patience=patience,
      ranking='mean independently checked welfare, then mean candidates per search second',
      timing='JIT/setup excluded; deadline checked between batches'),
    trials=[],choice=None)
  warmed=set()
  for pool,candidate_count,width in product(pool_sizes,candidates,threads):
    trial=dict(pool_size=pool,candidates=candidate_count,threads=width,runs=[])
    report['trials'].append(trial)
    config=replace(o,pool_size=pool,candidates=candidate_count,iterations=2**31-1,
      seed=seeds[0],time_limit=time_limit)
    required=estimate_device_bytes(p,config,patience,batch_size)
    trial['estimated_device_bytes']=required
    try:
      for seed in seeds:
        available=min(o.max_device_bytes,int(_free_device_bytes()*(1-reserve_fraction)))
        trial['available_device_bytes']=available
        if required>available:
          trial['status']='skipped_memory'
          break
        if width not in warmed:
          solve(p,replace(config,iterations=1,time_limit=3600.),y,
            patience=0,batch_size=batch_size,threads=width,time_budget='search')
          warmed.add(width)
          # Warm-up allocations and concurrent jobs can change free device memory.
          available=min(o.max_device_bytes,int(_free_device_bytes()*(1-reserve_fraction)))
          trial['available_device_bytes']=available
          if required>available:
            trial['status']='skipped_memory'
            break
        result=solve(p,replace(config,seed=seed),y,patience=0,
          batch_size=batch_size,threads=width,time_budget='search')
        checked=p.check(result['y'])
        elapsed=float(result['execution']['search_elapsed_s'])
        if not math.isfinite(checked['welfare']) or not math.isfinite(elapsed) or elapsed<=0:
          raise ValueError('nonfinite welfare or invalid measured search duration')
        run=dict(seed=seed,welfare=checked['welfare'],search_elapsed_s=elapsed,
          elapsed_s=result['elapsed_s'],candidates_evaluated=result['candidates_evaluated'],
          candidates_per_second=result['candidates_evaluated']/elapsed,
          termination_reason=result['termination_reason'],independently_validated=True)
        trial['runs'].append(run)
        if result['termination_reason']!='time_limit':
          raise ValueError('trial ended before its search-time budget')
        if elapsed>time_limit+max_overrun:
          trial.update(status='excluded_time_overrun',
            exclusion_reason=f'search exceeded budget by {elapsed-time_limit:.6g}s; allowed {max_overrun:.6g}s')
          break
      else:
        trial.update(status='ok',mean_welfare=mean(r['welfare'] for r in trial['runs']),
          mean_candidates_per_second=mean(r['candidates_per_second'] for r in trial['runs']))
    except Exception as exc:
      trial.update(status='failed',error=f'{type(exc).__name__}: {exc}')
  eligible=[row for row in report['trials'] if row['status']=='ok']
  if eligible:
    winner=max(eligible,key=lambda row:(row['mean_welfare'],row['mean_candidates_per_second']))
    report['choice']={key:winner[key] for key in ('pool_size','candidates','threads',
      'mean_welfare','mean_candidates_per_second')}
  return report
