"""Opt-in independent PLASMA proposals; host protocols and RNG stay ordered."""
import numpy as np
from math import gcd
from utils import step_time
from utils.compute_backend import cuda_dp_compute_enabled,record_operation


def node_problem_fits(node):
  """Driver-free DP preflight, using virtual curves before preparation/JIT."""
  from models.dp_gpu import gpu_problem_fits
  memory=node.params.ram_req;cap=node.params.ram_cap
  if (not np.isfinite(cap) or cap<0 or cap!=int(cap) or not np.isfinite(memory).all()
      or (memory<=0).any() or (memory!=np.rint(memory)).any()):return False
  return gpu_problem_fits([range(int(cap//m)+1) for m in memory],memory,cap)


def dp_batch_eligible(nodes):
  """Fixed complete-phase gate, without preparing curves or touching CUDA."""
  supported=[node for node in nodes if node.alive and node_problem_fits(node)]
  if len(supported)<2:return False
  work=0
  for node in supported:
    memory=[int(m) for m in node.params.ram_req];scale=gcd(*memory)
    budget=int(node.params.ram_cap)//scale
    for m in memory:
      m=m//scale;q,remainder=divmod(budget,m)
      work+=m*q*(q+1)//2+(q+1)*(remainder+1)
  return work>=5000 and work>=200*len(supported)


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


def welfare_plan_batch(jobs):
  """Independent prepared plans; reservations/acceptance belong to the caller."""
  selected=[i for i,(node,_) in enumerate(jobs) if node.alive and node_problem_fits(node)] if cuda_dp_compute_enabled() else []
  if len(selected)<2:
    record_operation('plasma_welfare_plan','cpu',len(jobs),reason='both CUDA options and at least two supported independent jobs required')
    return [node.plan(reservations) for node,reservations in step_time.timed(jobs,key=lambda job:job[0].params.node_id)]
  step_time.mark_shared_device()
  from models.dp_gpu import minimize_replica_costs_batch
  prepared=[jobs[i][0]._prepare_plan(jobs[i][1]) for i in selected]
  choices=minimize_replica_costs_batch([p.costs for p in prepared],[p.memory for p in prepared],[p.ram_cap for p in prepared])
  proposals={i:(p,r) for i,p,r in zip(selected,prepared,choices)}
  result=[]
  for i,(node,reservations) in enumerate(jobs):
    if i not in proposals:result.append(node.plan(reservations));continue
    p,r=proposals[i]
    if r is None:raise ValueError('No feasible local replica allocation')
    result.append(node._finish_plan(p,r.astype(int)))
  record_operation('plasma_welfare_plan','cuda',len(selected),shape=[len(jobs),max(node.Nf for node,_ in jobs)])
  if len(selected)<len(jobs):
    record_operation('plasma_welfare_plan','cpu',len(jobs)-len(selected),reason='dead or unsupported integer memory/device bounds')
  return result


def _window_compatible(nodes):
  if not nodes:return False
  nf=nodes[0].Nf
  for node in nodes:
    if node.Nf!=nf or node.opts.kappa not in (1.,2.):return False
    arrays=(node.D,node._phi,node._rewards,node.lam_hat,node._arrivals,node._admitted,node.params.u_max)
    if any(a.dtype!=np.float64 or not np.isfinite(a).all() for a in arrays):return False
    if not all(np.isfinite(v) for v in (node.opts.W,node.opts.mu,node.opts.ewma,node.opts.D_min,node.opts.D_max)):return False
    capacity=node.r.astype(float)*node.params.u_max*node.opts.W
    if not np.isfinite(capacity).all() or (np.abs(capacity)>=float(2**63)).any():return False
  return True


def window_buffer_bytes(nodes):
  if not nodes:return 0
  nf=nodes[0].Nf;width=max(node.deg+2 for node in nodes)
  return 32*len(nodes)*nf*width+96*len(nodes)*nf+56*len(nodes)


def _numeric_kernels():
  global cuda,_numeric_cached
  if '_numeric_cached' in globals():return _numeric_cached
  from numba import cuda
  from numba.cuda import libdevice as ld

  @cuda.jit
  def window(D,phi,rewards,lam,arrivals,admitted,replicas,umax,options,out,newlam,spare):
    k=cuda.grid(1);width=D.shape[2];nf=D.shape[1];i,f,c=k//(nf*width),(k//width)%nf,k%width
    if i<D.shape[0]:
      mu,kappa,low,high,ew,W=options[i,0],options[i,1],options[i,2],options[i,3],options[i,4],options[i,5]
      powered=phi[i,f,c] if kappa==1. else ld.dmul_rn(phi[i,f,c],phi[i,f,c])
      value=ld.dadd_rn(ld.dmul_rn(ld.dadd_rn(1.,-mu),D[i,f,c]),ld.dmul_rn(ld.dmul_rn(mu,powered),rewards[i,f,c]))
      out[i,f,c]=min(high,max(low,value))
      if c==0:
        newlam[i,f]=ld.dadd_rn(ld.dmul_rn(ld.dadd_rn(1.,-ew),lam[i,f]),ld.dmul_rn(ew,arrivals[i,f]))
        cap=int(ld.dadd_rn(ld.dmul_rn(ld.dmul_rn(replicas[i,f],umax[i,f]),W),1e-9))
        spare[i,f]=max(0.,ld.dadd_rn(float(cap),-admitted[i,f]))

  @cuda.jit
  def weights(D,local,spare,eps,out):
    k=cuda.grid(1);width=D.shape[2];nf=D.shape[1];i,f,c=k//(nf*width),(k//width)%nf,k%width
    if i<D.shape[0]:
      value=D[i,f,c]
      if c==0 and not local[i,f]:value=0.
      elif c>=2:value=ld.dmul_rn(value,1. if spare[i,f,c-2]>0. else eps)
      out[i,f,c]=value
  _numeric_cached=cuda,window,weights
  return _numeric_cached


def end_window_batch_cuda(nodes):
  """Full independent window arithmetic plus original ordered host reset."""
  if not _window_compatible(nodes) or window_buffer_bytes(nodes)>64*1024*1024:
    raise ValueError('unsupported window numeric domain or device buffer bound')
  step_time.mark_shared_device()
  from plasma.core.node import WindowCounts
  cuda,kernel,_=_numeric_kernels()
  count,nf=len(nodes),nodes[0].Nf;width=max(node.deg+2 for node in nodes)
  D=np.zeros((count,nf,width));phi=np.zeros_like(D);rewards=np.zeros_like(D)
  for i,node in enumerate(nodes):
    D[i,:,:node.deg+2]=node.D;phi[i,:,:node.deg+2]=node._phi;rewards[i,:,:node.deg+2]=node._rewards
  arrays=[D,phi,rewards]+[np.array([getattr(node,key) for node in nodes]) for key in ('lam_hat','_arrivals','_admitted','r')]
  arrays[6]=arrays[6].astype(float)
  arrays.extend([np.array([node.params.u_max for node in nodes]),np.array([[node.opts.mu,node.opts.kappa,node.opts.D_min,node.opts.D_max,node.opts.ewma,node.opts.W] for node in nodes])])
  inputs=[cuda.to_device(np.array(a,order='C',copy=True)) for a in arrays]
  out=cuda.device_array(D.shape);lam=cuda.device_array((count,nf));spare=cuda.device_array((count,nf))
  kernel[(D.size+127)//128,128](*inputs,out,lam,spare)
  out,lam,spare=out.copy_to_host(),lam.copy_to_host(),spare.copy_to_host();cuda.synchronize()
  counts=[]
  for i,node in enumerate(nodes):
    counts.append(WindowCounts(node._x,node._z,node._y,node._xi))
    node.D=out[i,:,:node.deg+2].copy();node.lam_hat=lam[i].copy();node._spare_last=spare[i].copy()
    node._pull_last=node._pull;node._recv_from_last=node._recv_from
    node.begin_window()
  record_operation('plasma_end_window','cuda',count,shape=[count,nf,width],device_bytes=window_buffer_bytes(nodes))
  return counts


def target_weights_batch_cuda(D,local,spare,eps):
  """Pure routing gate prototype; routing/admission and all RNG stay host."""
  step_time.mark_shared_device()
  cuda,_,kernel=_numeric_kernels()
  inputs=[cuda.to_device(np.array(a,order='C',copy=True)) for a in (D,local,spare)]
  out=cuda.device_array(D.shape,dtype=np.float64)
  kernel[(D.size+127)//128,128](*inputs,eps,out)
  result=out.copy_to_host();cuda.synchronize()
  return result


def route_all_cuda(engine,round_,arrivals):
  """Immutable weight preparation followed by the unchanged admission order."""
  active=[i for i,node in enumerate(engine.nodes) if node.alive]
  if not active:return engine._route_all_host(round_,arrivals)
  with step_time.parallel_phase():
    step_time.mark_shared_device()
    nf=engine.nodes[active[0]].Nf;width=max(engine.nodes[i].deg+2 for i in active)
    if 24*len(active)*nf*width>64*1024*1024:raise ValueError('routing snapshot exceeds device bound')
    D=np.zeros((len(active),nf,width));spare=np.zeros((len(active),nf,width-2))
    for row,i in enumerate(active):
      node=engine.nodes[i]
      D[row,:,:node.deg+2]=node.D;spare[row,:,:node.deg]=node._nbr_spare(round_).T
    weights=target_weights_batch_cuda(D,np.zeros((len(active),nf),bool),spare,engine.opts.eps_explore)
    prepared={i:weights[row,:,:engine.nodes[i].deg+2].copy() for row,i in enumerate(active)}
    record_operation('plasma_route_weights','cuda',len(active),shape=[len(active),nf,width])
  return engine._route_all_host(round_,arrivals,prepared)
