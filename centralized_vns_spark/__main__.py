"""GPU-resident VNS-R, optional hardware calibration and standalone x refinement."""
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import time
import numpy as np
from centralized_vns.core import Options
from centralized_vns.io import load_instance,from_data,initialize as initialize_reference
from centralized_vns_r.io import digest,load_solution,save_result
from .gpu import solve,validate_execution
from .xstage import refine_x,validate_x_options


def initialize(data,device):
  if device=='cpu':
    return initialize_reference(data,'cpu')
  from models.local_numeric_gpu import try_solve_local_batch_cuda
  from models.sp import LSP_detailed
  from utils.compute_backend import compute_session
  n,f=data[None]['Nn'][None],data[None]['Nf'][None]
  local=[{None:{**data[None],'whoami':{None:i+1}}} for i in range(n)]
  with compute_session({'use_dp':True,'dp_device':'cuda','compute_device':'cuda'}):
    solutions=try_solve_local_batch_cuda(LSP_detailed(),local)
  if len(solutions)!=n or any(s is None for s in solutions):
    raise RuntimeError('CUDA DP unsupported for this input; supply --solution or explicitly --init-device cpu')
  return np.array([s['x'] for s in solutions]).reshape(n,f),dict(cpu_problems=0,cuda_problems=n)


def main(argv=None):
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('instance',type=Path)
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--step',type=int,default=0)
  parser.add_argument('--solution',type=Path,help='NPZ or native selfish-distributed result folder')
  parser.add_argument('--solution-row',type=int,default=0)
  parser.add_argument('--init-device',choices=('cuda','cpu'),default='cuda')
  parser.add_argument('--pool-size',type=int,default=256)
  parser.add_argument('--candidates',type=int,default=512)
  parser.add_argument('--iterations',type=int,default=100000)
  parser.add_argument('--time-limit',type=float,default=300.)
  parser.add_argument('--patience',type=int,default=200)
  parser.add_argument('--plateau-tolerance',type=float,default=1e-6)
  parser.add_argument('--plateau-scope',choices=('all','global'),default='all')
  parser.add_argument('--threads',type=int,default=128)
  parser.add_argument('--batch-size',type=int,default=16)
  parser.add_argument('--seed',type=int,default=4850)
  parser.add_argument('--max-device-mib',type=int,default=4096)
  parser.add_argument('--autotune',action='store_true')
  parser.add_argument('--calibration-time',type=float,default=2.,help='search seconds per calibration trial')
  parser.add_argument('--refine-x',action='store_true')
  parser.add_argument('--refine-only',action='store_true',help='apply CUDA x stage to --solution')
  parser.add_argument('--x-iterations',type=int,default=10000)
  parser.add_argument('--x-time-limit',type=float,default=15.)
  args=parser.parse_args(argv)
  o=Options(pool_size=args.pool_size,candidates=args.candidates,iterations=args.iterations,
    time_limit=args.time_limit,seed=args.seed,backend='cuda',max_device_bytes=args.max_device_mib*1024**2)
  try:
    validate_execution(o,args.patience,args.plateau_tolerance,args.plateau_scope,args.batch_size,args.threads)
    validate_x_options(args.x_iterations,args.x_time_limit,args.seed,o.max_device_bytes)
    if not np.isfinite(args.calibration_time) or args.calibration_time<=0:
      raise ValueError('calibration-time must be positive and finite')
    if args.step<0 or args.solution_row<0:
      raise ValueError('step and solution-row must be nonnegative')
    if args.refine_only and (args.solution is None or args.autotune):
      raise ValueError('--refine-only requires --solution and cannot use --autotune')
    if args.output.exists():
      raise ValueError('output already exists; choose a new directory')
  except ValueError as exc:
    parser.error(str(exc))
  data,hashes=load_instance(args.instance,args.step)
  start=time.perf_counter()
  if args.solution is None:
    x,coverage=initialize(data,args.init_device)
    p=from_data(data,x);y=np.zeros(len(p.src),dtype=np.int64)
    provenance=dict(initialization_coverage=coverage,floor_reference='initial_DP_x')
  else:
    sidecar=args.solution.parent/'result.json'
    if args.solution.is_file() and sidecar.is_file():
      saved=json.loads(sidecar.read_text())
      if 'input_sha256' in saved and (saved['input_sha256']!=hashes or saved.get('step')!=args.step):
        raise ValueError('solution provenance does not match instance/step')
    p,y,provenance=load_solution(data,args.solution,args.solution_row)
  root=Path(__file__).resolve().parents[1]
  metadata=dict(instance=str(args.instance.resolve()),step=args.step,input_sha256=hashes,
    initialization_s=time.perf_counter()-start,**provenance,
    source_sha256={str(f.relative_to(root)):digest(f)
      for pkg in ('centralized_vns','centralized_vns_r','centralized_vns_spark')
      for f in (root/pkg).iterdir() if f.suffix in ('.py','.metal')})
  calibration=None
  if args.autotune:
    from .calibrate import tune
    calibration=tune(p,o,y,time_limit=args.calibration_time,batch_size=args.batch_size,
      patience=args.patience)
    choice=calibration['choice']
    if choice is not None:
      o=replace(o,pool_size=choice['pool_size'],candidates=choice['candidates'])
      args.threads=choice['threads']
  args.output.mkdir(parents=True,exist_ok=False)
  if calibration is not None:
    (args.output/'calibration.json').write_text(json.dumps(calibration,indent=2,allow_nan=False)+'\n')
    if calibration['choice'] is None:
      raise RuntimeError('no valid CUDA calibration configuration; see calibration.json')
  summary={}
  if not args.refine_only:
    fixed=solve(p,o,y,patience=args.patience,plateau_tolerance=args.plateau_tolerance,
      plateau_scope=args.plateau_scope,batch_size=args.batch_size,threads=args.threads)
    save_result(args.output/'fixed_x',p,fixed,{**metadata,'options':{**asdict(o),
      'patience':args.patience,'plateau_tolerance':args.plateau_tolerance,
      'plateau_scope':args.plateau_scope,'batch_size':args.batch_size,'threads':args.threads}})
    summary['fixed_x_welfare']=fixed['welfare'];y=fixed['y']
    metadata={**metadata,'parent_solution_sha256':digest(args.output/'fixed_x/solution.npz')}
  if args.refine_x or args.refine_only:
    settings=dict(iterations=args.x_iterations,time_limit=args.x_time_limit,
      seed=args.seed,max_device_bytes=o.max_device_bytes)
    variable=refine_x(p,y,**settings)
    save_result(args.output/'variable_x',p,variable,{**metadata,'options':settings})
    summary['variable_x_welfare']=variable['welfare']
  (args.output/'comparison.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
  print(json.dumps(summary))


if __name__=='__main__':
  main()
