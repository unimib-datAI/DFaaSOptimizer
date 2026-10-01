from pathlib import Path
import json
import multiprocessing as mp
import sys
import time
import numpy as np
sys.path.insert(0, str(Path.cwd()))
from models.sp import LSP
from remote_experiments.instances import load_materialized_instance
from run_centralized_model import get_current_load, update_data
from run_faasmacro import init_parallel_worker, solve_single_agent

if __name__ == '__main__':
    data,traces,agents,_=load_materialized_instance('solutions/madea-pg-compact-planar-temporal-2026-10-01/instances/n80-f10-s7')
    agents=list(agents)
    data=update_data(data, {'incoming_load':get_current_load(traces,agents,0)})
    data[None]['pi']={f+1:0 for f in range(10)}
    initargs=(data,{},'gurobi',LSP())
    init_parallel_worker(*initargs)
    before=time.perf_counter()
    oracle=[solve_single_agent(i) for i in agents]
    serial=time.perf_counter()-before
    assert all(v['termination_condition']=='optimal' for _,v in oracle)
    rows=[dict(processes=0, phase='sequential',seconds=serial)]
    for n in (1,2,4):
        before=time.perf_counter()
        with mp.Pool(n, initializer=init_parallel_worker,initargs=initargs) as pool:
            for repetition in range(3):
                started=time.perf_counter()
                result=pool.map(solve_single_agent,agents)
                elapsed=time.perf_counter()-started
                total=time.perf_counter()-before
                for (_,a),(_,b) in zip(oracle,result):
                    for key in ('x','r','omega','z'):
                        np.testing.assert_array_equal(a[key],b[key])
                rows.append(dict(processes=n,phase='first_batch_with_startup' if repetition==0 else 'warm_batch',seconds=total if repetition==0 else elapsed))
        print(json.dumps(rows[-3:]),flush=True)
    Path('experiments/madea_pg_compact/results_2026_10_01/pool_microbenchmark.json').write_text(json.dumps(dict(nodes=80,functions=10,seed=7,model='LSP',start_method=mp.get_start_method(),scope='local DP solve only, same input, no auction/PG commits; timings exclude comparison assertions',rows=rows),indent=2))
