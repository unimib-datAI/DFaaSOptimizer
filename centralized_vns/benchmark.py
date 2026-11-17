"""Paired CPU/Metal benchmark with exact outputs and fixed iteration budgets."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time
import numpy as np
from .core import CPU, Options
from .io import load_instance, initialize, from_data


def measure(problem, options):
  from .metal import Metal
  options.validate()
  start=time.perf_counter()
  engine=(CPU if options.backend=='cpu' else Metal)(problem,options)
  setup=time.perf_counter()-start
  start=time.perf_counter()
  trace=[engine.best.copy()]
  for iteration in range(options.iterations):
    trace.append(engine.step(iteration))
  y=engine.result()
  search=time.perf_counter()-start
  checked=problem.check(y)
  return dict(options=asdict(options),setup_s=setup,search_s=search,
              setup_and_search_s=setup+search,welfare=checked['welfare'],
              feasible=True,candidates_evaluated=engine.evaluated), {
                'y':y,'z':checked['z'],'replicas':checked['replicas'],'trace':np.array(trace)}


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('instance',type=Path)
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--step',type=int,default=0)
  parser.add_argument('--pool-sizes',type=int,nargs='+',default=[4,32])
  parser.add_argument('--candidates',type=int,default=64)
  parser.add_argument('--iterations',type=int,default=100)
  parser.add_argument('--seeds',type=int,nargs='+',default=[4850,4851,4852])
  parser.add_argument('--repeats',type=int,default=3)
  parser.add_argument('--max-device-mib',type=int,default=512)
  args=parser.parse_args()
  if args.repeats<1:
    parser.error('repeats must be positive')
  for pool in args.pool_sizes:
    for seed in args.seeds:
      Options(pool_size=pool,candidates=args.candidates,iterations=args.iterations,
              seed=seed,max_device_bytes=args.max_device_mib*1024*1024).validate()
  if args.output.exists():
    parser.error('output already exists; choose a new directory')
  import mlx.core as mx
  start=time.perf_counter()
  data,hashes=load_instance(args.instance,args.step)
  load_s=time.perf_counter()-start
  start=time.perf_counter(); x,coverage=initialize(data,'cpu')
  init_s=time.perf_counter()-start
  p=from_data(data,x)
  root=Path(__file__).resolve().parents[1]
  def git(*cmd):
    return subprocess.run(['git','-C',str(root),*cmd],capture_output=True,text=True,check=True).stdout.strip()
  source_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest()
                 for f in Path(__file__).parent.iterdir() if f.suffix in ('.py','.metal')}
  report=dict(instance=str(args.instance.resolve()),step=args.step,input_sha256=hashes,
    source_sha256=source_hashes,solver_commit=git('rev-parse','HEAD'),
    solver_dirty=bool(git('status','--porcelain')),python=platform.python_version(),
    platform=platform.platform(),numpy=np.__version__,mlx=mx.__version__,device=mx.device_info(),
    load_s=load_s,initialization_s=init_s,initialization_coverage=coverage,
    budget='fixed iterations; no wall-time cutoff',warmup=[],pairs=[],summary=[])
  args.output.mkdir(parents=True,exist_ok=False)
  np.savez_compressed(args.output/'problem.npz',x=p.x,edges=p.edges)
  for pool in args.pool_sizes:
    options=Options(pool_size=pool,candidates=args.candidates,iterations=args.iterations,
                    max_device_bytes=args.max_device_mib*1024*1024)
    # First two steps include lazy kernel compilation; measured runs start fresh.
    for backend in ('cpu','metal'):
      warm,_=measure(p,replace(options,backend=backend,iterations=2))
      report['warmup'].append(warm)
    for seed in args.seeds:
      for repeat in range(args.repeats):
        order=('cpu','metal') if len(report['pairs'])%2==0 else ('metal','cpu')
        metrics={}; solutions={}
        for backend in order:
          metrics[backend],solutions[backend]=measure(p,replace(options,backend=backend,seed=seed))
        for key in ('y','z','replicas','trace'):
          np.testing.assert_array_equal(solutions['cpu'][key],solutions['metal'][key],
            err_msg=f'{key}: pool={pool}, seed={seed}, repeat={repeat}')
        name=f'pool{pool}-seed{seed}-repeat{repeat}'
        for backend in order:
          np.savez_compressed(args.output/f'{name}-{backend}.npz',**solutions[backend])
        pair=dict(pool_size=pool,seed=seed,repeat=repeat,order=order,
                  allocations_equal=True,traces_equal=True,**metrics)
        report['pairs'].append(pair)
        (args.output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(pool=pool,seed=seed,repeat=repeat,
          cpu_s=metrics['cpu']['search_s'],metal_s=metrics['metal']['search_s'],
          welfare=metrics['cpu']['welfare'],equal=True)),flush=True)
    rows=[r for r in report['pairs'] if r['pool_size']==pool]
    cpu=statistics.median(r['cpu']['search_s'] for r in rows)
    metal=statistics.median(r['metal']['search_s'] for r in rows)
    report['summary'].append(dict(pool_size=pool,cpu_median_s=cpu,metal_median_s=metal,
                                   speedup=cpu/metal,pairs=len(rows)))
  (args.output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
  print(json.dumps(report['summary'],indent=2))


if __name__=='__main__':
  main()
