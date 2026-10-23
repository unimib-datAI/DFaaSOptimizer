"""One-shot selfish local decisions followed by the historical LRMP coordinator.

The detailed variant uses the central reference/defaults and preserves local x
through LSPr_x. Historical LSP/LSPr remains a separately selectable baseline.
"""
import time
from pathlib import Path

import highspy
import numpy as np
import pyomo.environ as pyo
from pyomo.contrib.appsi.solvers import Highs

from models.rmp import LRMP
from models.sp import LSP, LSP_detailed, LSPr, LSPr_x
from remote_experiments.instances import load_materialized_instance
from run_centralized_model import get_current_load, update_data
from run_faasmacro import (combine_solutions, compute_social_welfare,
  parallel_solver_session, prepare_master_data, solve_subproblem)
from utils.warm_start import (data_fingerprint, evaluated_data_and_floors,
  initialize_variables, load_start, save_decisions, solution_nodes, validate_decisions)


def coordinator_start(seed, data, x, r, omega):
  """Trim saved flows to fixed local decisions; r in the result is ADDITIONAL."""
  d = data[None]; nn, nf = x.shape
  y = np.zeros((nn, nn, nf), dtype=np.int64)
  total = np.rint(r).astype(np.int64).copy()
  memory = np.array([d['memory_requirement'][f+1] for f in range(nf)])
  capacity = np.array([d['memory_capacity'][n+1] for n in range(nn)])
  free = capacity-total@memory
  remaining = np.rint(omega).astype(np.int64).copy()
  inbound = np.zeros((nn, nf), dtype=np.int64)
  if seed is not None:
    flows = np.argwhere(seed['y']>0)
    def score(index):
      i,j,f = map(int,index)
      return (-(d['beta'].get((i+1,j+1,f+1),.9)+d['gamma'].get((i+1,f+1),.8)) /
              (d['incoming_load'][i+1,f+1] or 1),i,j,f)
    for index in sorted(flows,key=score):
      i,j,f = map(int,index)
      if not d['neighborhood'].get((i+1,j+1),0) or i==j:
        continue
      utilization=d['max_utilization'][f+1];demand=d['demand'][j+1,f+1]
      maximum_r=total[j,f]+int(np.floor(max(0.,free[j])/memory[f]+1e-9))
      maximum_in=int(np.floor(maximum_r*utilization/demand-x[j,f]-inbound[j,f]+1e-9))
      q=max(0,min(int(seed['y'][i,j,f]),int(remaining[i,f]),maximum_in))
      if q == 0:
        continue
      needed=int(np.ceil(demand*(x[j,f]+inbound[j,f]+q)/utilization-1e-9))
      added=max(0,needed-total[j,f])
      free[j]-=added*memory[f];total[j,f]+=added
      y[i,j,f]=q;remaining[i,f]-=q;inbound[j,f]+=q
  return dict(y=y,r=total-np.rint(r).astype(np.int64))


def _global_decisions(start, data, x, base_r):
  d=data[None]
  load=np.array([[d['incoming_load'][n+1,f+1] for f in range(x.shape[1])]
                 for n in range(x.shape[0])])
  return dict(x=x,y=start['y'],z=load-x-start['y'].sum(axis=1),r=base_r+start['r'])


def _check_coordinator(instance):
  for constraint in instance.component_data_objects(pyo.Constraint,active=True):
    value=pyo.value(constraint.body)
    if ((constraint.has_lb() and value<pyo.value(constraint.lower)-1e-6) or
        (constraint.has_ub() and value>pyo.value(constraint.upper)+1e-6)):
      raise ValueError(f'coordinator constraint violated: {constraint.name}')


def solve(path,time_limit,threads=1,gap=1e-3,detailed=True,warm_start=None,
          warm_start_sha256=None,solution_file=None,log_file=None,instance_name=None):
  instance_name = instance_name or Path(path).name
  overall=time.perf_counter()
  raw,traces,agents,_=load_materialized_instance(path)
  raw=update_data(raw,{'incoming_load':get_current_load(traces,list(agents),0)})
  data,floors=evaluated_data_and_floors(raw)
  nn,nf=data[None]['Nn'][None],data[None]['Nf'][None]
  seed,seed_info=(None,None)
  if warm_start is not None:
    seed,seed_info=load_start(warm_start,data,warm_start_sha256,
                             floors if detailed else None,instance_name)
  parallelism=threads if threads>1 else 0
  options={'use_dp':True,'threads':1}
  with parallel_solver_session():
    local=solve_subproblem(data if detailed else raw,list(range(nn)),
      LSP_detailed() if detailed else LSP(),'appsi_highs',options,parallelism)
    sp_data,x,sp_y,z,omega,r,rho,_,_,tc,_=local
    if set(str(tc['tot']).split('-')) != {'optimal'}:
      raise ValueError('local reference was not solved optimally')
    local_seconds=time.perf_counter()-overall
    start=coordinator_start(seed,data,x,r,omega)
    chosen,welfare=validate_decisions(_global_decisions(start,data,x,r),data,floors if detailed else None)
    adapted_welfare=welfare
    source='warm_start' if seed is not None else 'local_only'
    metadata=dict(instance=instance_name,step=0,
      algorithm='selfish-distributed-detailed' if detailed else 'selfish-distributed',
      data_sha256=data_fingerprint(data))
    if solution_file:
      save_decisions(solution_file,chosen,dict(metadata,welfare=welfare,solution_source=source))
    started=time.perf_counter();deadline=started+time_limit
    result=None;build=0.;rejected=None
    if time_limit>0:
      master=prepare_master_data(data,(x,sp_y,z,omega,r,rho))
      instance=LRMP().generate_instance(master)
      initialize_variables(instance,start)
      _check_coordinator(instance)
      build=time.perf_counter()-started
      if time.perf_counter()<deadline:
        highspy.Highs.resetGlobalScheduler(True)
        opt=Highs();opt.config.time_limit=max(0.,deadline-time.perf_counter())
        opt.config.load_solution=False;opt.config.warmstart=True
        opt.highs_options.update(threads=threads,mip_rel_gap=gap,random_seed=0)
        if log_file:opt.highs_options['log_file']=str(log_file)
        try:
          result=opt.solve(instance)
        finally:
          highspy.Highs.resetGlobalScheduler(True)
        if result.best_feasible_objective is not None:
          result.solution_loader.load_vars()
          try:
            _check_coordinator(instance)
            candidate=dict(y=np.array([pyo.value(instance.y[i,j,f]) for i,j,f in instance.y]).reshape(nn,nn,nf),
                           r=np.array([pyo.value(instance.r[i,f]) for i,f in instance.r]).reshape(nn,nf))
            arrays,value=validate_decisions(_global_decisions(candidate,data,x,r),data,floors if detailed else None)
            if value>=welfare-1e-6:
              chosen,welfare,source=arrays,value,'solver'
          except ValueError as error:
            rejected=str(error)
    # Reuse the historical restricted solve; the detailed variant fixes local x.
    fixed=chosen['y'];assigned=fixed.sum(axis=1)
    restricted,_,restricted_tc,_=compute_social_welfare(LSPr_x() if detailed else LSPr(),
      data if detailed else sp_data,list(range(nn)),'appsi_highs',options,fixed,assigned,
      parallelism,x if detailed else None)
    if set(str(restricted_tc).split('-')) != {'optimal'}:
      raise ValueError('restricted local solve failed')
    final_x,_,_,_,final_r,final_rho=restricted
    combined=combine_solutions(nn,nf,data,data[None]['incoming_load'],final_x,final_r,final_rho,
      x,fixed,chosen['z'],chosen['r'],fixed.transpose(1,0,2),rho)
    chosen,welfare=validate_decisions(combined['sp'],data,floors if detailed else None)
    if detailed and not np.array_equal(chosen['x'],np.rint(x)):
      raise ValueError('detailed local x changed after coordination')
  bound=result.best_objective_bound if result else None
  if bound is not None and not np.isfinite(bound):bound=None
  objective=result.best_feasible_objective if result else None
  proven=bool(result and source=='solver' and bound is not None and objective is not None and abs(bound-objective)<=1e-6)
  if solution_file:
    save_decisions(solution_file,chosen,dict(metadata,welfare=welfare,solution_source=source))
  return dict(status='ok',feasible=True,welfare=welfare,nodes=nn,functions=nf,
    model='Selfish-DLMM-detailed' if detailed else 'Selfish-DLMM',
    local_reference='LSP_detailed' if detailed else 'LSP',restricted_model='LSPr_x' if detailed else 'LSPr',
    local_reference_seconds=local_seconds,build_seconds=build,wall_seconds=time.perf_counter()-overall,
    mip_rel_gap=gap,solver_threads=threads,solver_log=str(log_file) if log_file else None,
    termination=str(result.termination_condition) if result else 'TerminationCondition.maxTimeLimit',
    bound=bound,bound_scope='coordinator',solver_objective=objective,
    rel_gap_to_bound=((bound-objective)/abs(bound) if bound else (0. if objective==0 else None))
      if objective is not None and bound is not None else None,
    proven_optimal=proven,proven_optimal_scope='coordinator',solution_source=source,
    solution_file=str(solution_file) if solution_file else None,solver_solution_rejected=rejected,
    warm_start=dict(seed_info,accepted=True,adapted_welfare=adapted_welfare) if seed_info else None,
    node_metrics=solution_nodes(chosen,data))
