"""Run VNS-R and optionally retain a separate VNS-RX refinement."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from centralized_vns.core import Options
from centralized_vns.io import load_instance,initialize,from_data
from .core import solve,validate_plateau
from .io import save_result,digest
from .xstage import refine_x


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('instance',type=Path)
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--step',type=int,default=0)
  parser.add_argument('--backend',choices=('cpu','metal','cuda'),default='cpu')
  parser.add_argument('--init-device',choices=('cpu','cuda'),default='cpu')
  parser.add_argument('--pool-size',type=int,default=32)
  parser.add_argument('--candidates',type=int,default=128)
  parser.add_argument('--iterations',type=int,default=200)
  parser.add_argument('--time-limit',type=float,default=300.)
  parser.add_argument('--patience',type=int,default=200,
    help='plateau window in iterations (default: 200, minimum: 10, 0 disables)')
  parser.add_argument('--plateau-tolerance',type=float,default=1e-6,
    help='maximum relative archive gain per trajectory over the window (default: 1e-6)')
  parser.add_argument('--seed',type=int,default=4850)
  parser.add_argument('--max-device-mib',type=int,default=512)
  parser.add_argument('--refine-x',action='store_true')
  parser.add_argument('--x-iterations',type=int,default=1000)
  parser.add_argument('--x-time-limit',type=float,default=15.)
  args=parser.parse_args()
  options=Options(pool_size=args.pool_size,candidates=args.candidates,iterations=args.iterations,
    time_limit=args.time_limit,seed=args.seed,backend=args.backend,max_device_bytes=args.max_device_mib*1024**2)
  options.validate()
  try:
    validate_plateau(args.patience,args.plateau_tolerance,options.shake_after)
  except ValueError as exc:
    parser.error(str(exc))
  if args.output.exists():
    parser.error('output already exists; choose a new directory')
  data,hashes=load_instance(args.instance,args.step)
  start=time.perf_counter();x,coverage=initialize(data,args.init_device)
  metadata=dict(instance=str(args.instance.resolve()),step=args.step,input_sha256=hashes,
    initialization_s=time.perf_counter()-start,initialization_coverage=coverage,
    options={**asdict(options),'patience':args.patience,'plateau_tolerance':args.plateau_tolerance},
    floor_reference='initial_DP_x',
    source_sha256={f.name:digest(f) for f in Path(__file__).parent.iterdir() if f.suffix in ('.py','.metal')})
  p=from_data(data,x);args.output.mkdir(parents=True,exist_ok=False)
  fixed=solve(p,options,patience=args.patience,plateau_tolerance=args.plateau_tolerance)
  save_result(args.output/'fixed_x',p,fixed,metadata)
  summary=dict(fixed_x_welfare=fixed['welfare'])
  if args.refine_x:
    variable=refine_x(p,fixed['y'],iterations=args.x_iterations,time_limit=args.x_time_limit,seed=args.seed)
    save_result(args.output/'variable_x',p,variable,{**metadata,
      'options':dict(iterations=args.x_iterations,time_limit=args.x_time_limit,seed=args.seed),
      'parent_solution_sha256':digest(args.output/'fixed_x/solution.npz')})
    summary['variable_x_welfare']=variable['welfare']
  (args.output/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
  print(json.dumps(summary))


if __name__=='__main__':
  main()
