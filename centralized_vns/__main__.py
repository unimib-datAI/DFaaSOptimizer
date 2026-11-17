"""Standalone runner; no registration or changes to existing experiment methods."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
from .core import Options,solve
from .io import load_instance,initialize,from_data


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('instance',type=Path)
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--step',type=int,default=0)
  parser.add_argument('--backend',choices=('cpu','cuda'),default='cpu')
  parser.add_argument('--init-device',choices=('cpu','cuda'),default='cpu')
  parser.add_argument('--pool-size',type=int,default=32)
  parser.add_argument('--candidates',type=int,default=128)
  parser.add_argument('--iterations',type=int,default=200)
  parser.add_argument('--seed',type=int,default=4850)
  parser.add_argument('--time-limit',type=float,default=300.,help='search wall seconds, including CUDA compilation/allocation; initialization timed separately')
  parser.add_argument('--max-device-mib',type=int,default=512)
  args=parser.parse_args()
  options=Options(pool_size=args.pool_size,candidates=args.candidates,iterations=args.iterations,
    seed=args.seed,time_limit=args.time_limit,backend=args.backend,max_device_bytes=args.max_device_mib*1024*1024)
  options.validate()
  # Never overwrite an earlier experiment accidentally.
  if args.output.exists():
    parser.error('output already exists; choose a new directory')
  start=time.perf_counter()
  data,hashes=load_instance(args.instance,args.step)
  init_start=time.perf_counter();x,coverage=initialize(data,args.init_device)
  initialization_s=time.perf_counter()-init_start
  problem=from_data(data,x)
  result=solve(problem,options)
  repo=Path(__file__).resolve().parents[1]
  def git(*cmd):
    r=subprocess.run(['git','-C',str(repo),*cmd],capture_output=True,text=True)
    return r.stdout.strip() if r.returncode==0 else None
  report={k:v for k,v in result.items() if not isinstance(v,np.ndarray)}
  report.update(options=asdict(options),instance=str(args.instance.resolve()),step=args.step,
    input_sha256=hashes,initialization_device=args.init_device,initialization_coverage=coverage,
    initialization_s=initialization_s,total_wall_s=time.perf_counter()-start,
    solver_commit=git('rev-parse','HEAD'),solver_dirty=bool(git('status','--porcelain')),
    python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
    cuda_simulator=os.environ.get('NUMBA_ENABLE_CUDASIM')=='1',feasible=True)
  args.output.mkdir(parents=True,exist_ok=False)
  np.savez_compressed(args.output/'solution.npz',x=problem.x,edges=problem.edges,
    y=result['y'],z=result['z'],replicas=result['replicas'])
  (args.output/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
  print(json.dumps({k:report[k] for k in ('welfare','initial_welfare','elapsed_s','feasible')}))


if __name__=='__main__':
  main()
