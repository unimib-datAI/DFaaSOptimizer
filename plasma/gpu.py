"""Opt-in independent PLASMA proposals; host protocols and RNG stay ordered."""
import numpy as np
from utils import step_time
from utils.compute_backend import cuda_dp_compute_enabled,record_operation


def node_problem_fits(node):
  """Driver-free DP preflight, using virtual curves before preparation/JIT."""
  from models.dp_gpu import gpu_problem_fits
  memory=node.params.ram_req;cap=node.params.ram_cap
  if (not np.isfinite(cap) or cap<0 or cap!=int(cap) or not np.isfinite(memory).all()
      or (memory<=0).any() or (memory!=np.rint(memory)).any()):return False
  return gpu_problem_fits([range(int(cap//m)+1) for m in memory],memory,cap)


def exact_sb_batch(nodes,round_):
  """Same barrier snapshots and proposal application order as standalone SB."""
  selected=[i for i,node in enumerate(nodes) if node.alive and node.opts.sbm_method=='exact' and node_problem_fits(node)] if cuda_dp_compute_enabled() else []
  if len(selected)<2:
    record_operation('plasma_exact_sb','cpu',len(nodes),reason='both CUDA options required; fewer than two eligible exact integer-memory problems')
    return [node.sb_pass(round_) for node in step_time.timed(nodes,key=id)]
  step_time.mark_shared_device()
  from plasma.core.sbm import prepare_exact_problem,r_max_per_fn
  from models.dp_gpu import minimize_replica_costs_batch
  prepared=[]
  for i in selected:
    node=nodes[i];ctx=node._hamiltonian_ctx(round_)
    prepared.append((ctx,prepare_exact_problem(ctx,r_max_per_fn(node.params.ram_cap,node.params.ram_req))))
  choices=minimize_replica_costs_batch([p[1][0] for p in prepared],[p[1][1] for p in prepared],[p[1][2] for p in prepared])
  proposals={i:(ctx,r) for i,(ctx,_),r in zip(selected,prepared,choices)}
  result=[]
  for i,node in enumerate(nodes):
    if i not in proposals:result.append(node.sb_pass(round_));continue
    ctx,r=proposals[i]
    # Infeasibility is a per-problem result, not a CUDA failure or a retry.
    result.append(False if r is None else node._apply_sb_proposal(ctx,r.astype(int)))
  record_operation('plasma_exact_sb','cuda',len(selected),shape=[len(nodes),max(node.Nf for node in nodes)])
  if len(selected)<len(nodes):
    record_operation('plasma_exact_sb','cpu',len(nodes)-len(selected),reason='dead, stochastic SB, unsupported integer memory or device bounds')
  return result
