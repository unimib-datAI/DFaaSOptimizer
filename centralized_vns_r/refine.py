"""Standalone guarded x refinement of an NPZ or native LSPc experiment folder."""
import argparse
import json
from pathlib import Path
from centralized_vns.io import load_instance
from .io import load_solution,save_result,digest
from .xstage import refine_x


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('instance',type=Path)
  parser.add_argument('--solution',type=Path,required=True)
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--step',type=int,default=0)
  parser.add_argument('--solution-row',type=int,default=0)
  parser.add_argument('--iterations',type=int,default=1000)
  parser.add_argument('--time-limit',type=float,default=15.)
  parser.add_argument('--seed',type=int,default=4850)
  args=parser.parse_args()
  if args.output.exists():
    parser.error('output already exists; choose a new directory')
  data,hashes=load_instance(args.instance,args.step)
  provenance=args.solution.parent/'result.json'
  if args.solution.is_file() and provenance.is_file():
    saved=json.loads(provenance.read_text())
    if 'input_sha256' in saved and (saved['input_sha256']!=hashes or saved.get('step')!=args.step):
      raise ValueError('solution provenance does not match instance/step')
  p,y,metadata=load_solution(data,args.solution,args.solution_row)
  result=refine_x(p,y,iterations=args.iterations,time_limit=args.time_limit,seed=args.seed)
  report=save_result(args.output,p,result,{**metadata,**dict(instance=str(args.instance.resolve()),
    step=args.step,input_sha256=hashes,options=dict(iterations=args.iterations,time_limit=args.time_limit,seed=args.seed),
    source_sha256={f.name:digest(f) for f in Path(__file__).parent.iterdir() if f.suffix in ('.py','.metal')})})
  print(json.dumps({k:report[k] for k in ('algorithm','initial_welfare','welfare','accepted','local_floor_satisfied')}))


if __name__=='__main__':
  main()
