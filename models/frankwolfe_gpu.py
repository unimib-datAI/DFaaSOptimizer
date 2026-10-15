"""Resident float64 Frank–Wolfe arithmetic, lazy CUDA and ordered reductions."""
import numpy as np
from utils import step_time


def _repair_kernel():
  global cuda,_repair_cached
  if '_repair_cached' in globals():return _repair_cached
  from numba import cuda
  from numba.cuda import libdevice as ld
  from models.numeric_gpu import _source_reducer
  pairwise=_source_reducer()
  @cuda.jit
  def kernel(y,base,out):
    k=cuda.grid(1);nf=base.shape[1];j,f=k//nf,k%nf
    if j<base.shape[0]:
      total=0.
      if nf==1:total=pairwise(y,j,f)
      else:
        for i in range(y.shape[0]):total=ld.dadd_rn(total,y[i,j,f])
      out[j,f]=ld.dadd_rn(base[j,f],total)
  _repair_cached=kernel
  return kernel


def repair_memory_cuda(y,params,gains,base_x,replicas,rho,integer=True,tol=1e-9,protected=None):
  """Independent receiver initial loads; original ordered greedy host repairs."""
  step_time.mark_shared_device()
  kernel=_repair_kernel()
  d_y=cuda.to_device(np.array(y,order='C',copy=True));d_base=cuda.to_device(np.array(base_x,dtype=np.float64,order='C',copy=True))
  loads=cuda.device_array(base_x.shape,dtype=np.float64)
  kernel[(loads.size+127)//128,128](d_y,d_base,loads)
  loads=loads.copy_to_host();cuda.synchronize()
  from decentralized_frankwolfe import _repair_memory_host
  return _repair_memory_host(y,params,gains,base_x,replicas,rho,integer,tol,protected,loads)


def buffer_bytes(n,nf):
  # q, gains, mask, gap, reduction tree, input and receiver workspaces.
  # Python integers cannot overflow; bound includes all uploaded temporaries.
  return 48*int(n)*int(n)*int(nf)+256*int(n)*int(nf)+64*int(nf)+16*int(n)


def fw_gains_cuda(params,omega,neighborhood,forbidden,latency,fairness,options):
  """Reuse the shared edge score formula, then reconstruct the existing mask."""
  step_time.mark_shared_device()
  from models.bids_gpu import _kernels
  from utils.compute_backend import record_operation
  cuda,kernel,_=_kernels()
  shape=params['beta'].shape
  inputs=[cuda.to_device(np.array(a,order='C',copy=True)) for a in
          (params['beta'],params['gamma'],latency,fairness,np.asarray(neighborhood)>0)]
  scores=cuda.device_array(shape,dtype=np.float64);mask=cuda.device_array(shape,dtype=np.bool_)
  kernel[(scores.size+127)//128,128](*inputs,options['latency_weight'],options['fairness_weight'],scores,mask)
  scores=scores.copy_to_host();mask=mask.copy_to_host();cuda.synchronize()
  mask &= (~forbidden)[None,:,:] & (np.asarray(omega)[:,None,:]>0)
  mask &= ~np.eye(params['Nn'],dtype=bool)[:,:,None]
  result=np.where(mask,scores,0.),mask
  record_operation('fw_gains','cuda',params['Nn'],shape=list(shape))
  return result


def eligible(params,omega,base,replicas,rho,gains,mask,options):
  n,nf=params['Nn'],params['Nf']
  arrays=(omega,base,replicas,rho,gains,params['demand'],params['max_utilization'],params['memory_requirement'])
  shapes=((n,nf),(n,nf),(n,nf),(n,),(n,n,nf),(n,nf),(nf,),(nf,))
  valid=(n>0 and nf>0 and mask.shape==(n,n,nf) and mask.dtype==np.bool_
         and all(isinstance(a,np.ndarray) and a.shape==s and a.dtype==np.float64 and np.isfinite(a).all()
                 for a,s in zip(arrays,shapes))
         and np.all(params['demand']>0) and np.all(params['max_utilization']>0)
         and buffer_bytes(n,nf)<=64*1024*1024)
  return valid,'finite canonical float64 arrays, positive demand/utilization and <=64 MiB required'


def _tree(size):
  """NumPy PW_BLOCKSIZE=128 split tree, grouped by dependency height."""
  leaves=[];levels={};count=1;roots=[]
  def visit(start,size):
    nonlocal count
    index=count;count+=1
    if size<=128:
      leaves.append((index,start,size));return index,0
    half=size//2;half-=half%8
    left,lh=visit(start,half);right,rh=visit(start+half,size-half)
    height=max(lh,rh)+1;levels.setdefault(height,[]).append((index,left,right))
    return index,height
  # NumPy's contiguous global reduce accumulates pairwise blocks in order.
  # Follow its current buffer setting instead of assuming a single large tree.
  block=np.getbufsize()
  for begin in range(0,size,block):roots.append(visit(begin,min(block,size-begin))[0])
  return np.asarray(leaves,dtype=np.int64),[np.asarray(levels[k],dtype=np.int64) for k in sorted(levels)],count,np.asarray(roots,dtype=np.int64)


def _kernels():
  global cuda,_cached
  if '_cached' in globals():return _cached
  from numba import cuda
  from numba.cuda import libdevice as ld
  from models.numeric_gpu import _source_reducer
  pairwise=_source_reducer()

  @cuda.jit
  def setup(replicas,demand,umax,memory,mref,cap,unit):
    k=cuda.grid(1);nf=cap.shape[1];j,f=k//nf,k%nf
    if j<cap.shape[0]:
      cap[j,f]=ld.ddiv_rn(ld.dmul_rn(replicas[j,f],umax[f]),demand[j,f])
      unit[j,f]=ld.ddiv_rn(ld.ddiv_rn(ld.dmul_rn(memory[f],demand[j,f]),umax[f]),mref)

  @cuda.jit
  def loads(q,base,cap,unit,delta,factors,slope):
    k=cuda.grid(1);nf=base.shape[1];j,f=k//nf,k%nf
    if j<base.shape[0]:
      total=0.
      # NumPy axis=0 for a C-contiguous 3D array walks sources in order.
      for i in range(q.shape[0]):total=ld.dadd_rn(total,q[i,j,f])
      u=ld.dadd_rn(ld.dadd_rn(base[j,f],total),-cap[j,f])
      if delta<=0:
        excess=max(0.,u);slope[j,f]=1. if u>1e-12 else 0.
      else:
        v=ld.dadd_rn(u,delta)
        excess=0. if u<=-delta else ld.ddiv_rn(ld.dmul_rn(v,v),ld.dmul_rn(4.,delta)) if u<delta else u
        slope[j,f]=min(1.,max(0.,ld.ddiv_rn(v,ld.dmul_rn(2.,delta))))
      factors[f,j,0]=ld.dmul_rn(unit[j,f],excess)

  @cuda.jit
  def prices(factors,slope,unit,rho,mref,kappa,price):
    j=cuda.grid(1)
    if j<price.shape[0]:
      violation=max(0.,ld.dadd_rn(pairwise(factors,j,0),-ld.ddiv_rn(rho[j],mref)))
      for f in range(price.shape[1]):
        price[j,f]=ld.dmul_rn(ld.dmul_rn(ld.dmul_rn(kappa,violation),unit[j,f]),slope[j,f])

  @cuda.jit
  def best_dest(gains,mask,price,omega,best,positive):
    k=cuda.grid(1);nf=omega.shape[1];i,f=k//nf,k%nf
    if i<omega.shape[0]:
      index=0;value=-np.inf
      for j in range(omega.shape[0]):
        adjusted=ld.dadd_rn(gains[i,j,f],-price[j,f]) if mask[i,j,f] else -np.inf
        if adjusted>value:value=adjusted;index=j
      best[i,f]=index;positive[i,f]=value>0. and omega[i,f]>0.

  @cuda.jit
  def gap_values(q,gains,mask,price,omega,best,positive,values):
    k=cuda.grid(1);nf=q.shape[2];n=q.shape[0];i,j,f=k//(n*nf),(k//nf)%n,k%nf
    if i<n:
      s=omega[i,f] if positive[i,f] and best[i,f]==j else 0.
      values[k,0,0]=ld.dmul_rn(ld.dadd_rn(gains[i,j,f],-price[j,f]),ld.dadd_rn(s,-q[i,j,f])) if mask[i,j,f] else 0.

  @cuda.jit
  def leaf_sums(values,leaves,sums):
    k=cuda.grid(1)
    if k<leaves.shape[0]:
      begin,size=leaves[k,1],leaves[k,2]
      sums[leaves[k,0]]=pairwise(values[begin:begin+size],0,0)

  @cuda.jit
  def combine(nodes,sums):
    k=cuda.grid(1)
    if k<nodes.shape[0]:sums[nodes[k,0]]=ld.dadd_rn(sums[nodes[k,1]],sums[nodes[k,2]])

  @cuda.jit
  def chunk_sum(roots,sums):
    if cuda.grid(1)==0:
      total=0.
      for i in range(roots.size):total=ld.dadd_rn(total,sums[roots[i]])
      sums[0]=total

  @cuda.jit
  def update(q,omega,best,positive,step):
    k=cuda.grid(1);nf=q.shape[2];n=q.shape[0];i,j,f=k//(n*nf),(k//nf)%n,k%nf
    if i<n:
      s=omega[i,f] if positive[i,f] and best[i,f]==j else 0.
      q[i,j,f]=ld.dadd_rn(ld.dmul_rn(ld.dadd_rn(1.,-step),q[i,j,f]),ld.dmul_rn(step,s))
  _cached=setup,loads,prices,best_dest,gap_values,leaf_sums,combine,update,chunk_sum
  return _cached


def relaxed_flows_cuda(params,omega,base_load,replicas,rho,gains,mask,options):
  """Full synchronous phases; CPU final statistic reconstruction is charged too."""
  valid,reason=eligible(params,omega,base_load,replicas,rho,gains,mask,options)
  if not valid:raise ValueError(reason)
  n,nf=params['Nn'],params['Nf'];size=n*n*nf
  with step_time.parallel_phase():
    step_time.mark_shared_device()
    setup,loads,prices,best_dest,gap_values,leaf_sums,combine,update,chunk_sum=_kernels()
    mref=float(np.mean(params['memory_requirement'])) or 1.
    scale=max(1.,float((gains.max(axis=1,initial=0.)*omega).sum()))
    d_omega,d_base,d_r,d_rho,d_gains,d_mask,d_demand,d_umax,d_memory=[cuda.to_device(np.array(a,order='C',copy=True)) for a in
      (omega,base_load,replicas,rho,gains,mask,params['demand'],params['max_utilization'],params['memory_requirement'])]
    q=cuda.to_device(np.zeros((n,n,nf)));cap=cuda.device_array((n,nf));unit=cuda.device_array((n,nf))
    factors=cuda.device_array((nf,n,1));slope=cuda.device_array((n,nf));price=cuda.to_device(np.zeros((n,nf)))
    best=cuda.device_array((n,nf),dtype=np.int64);positive=cuda.device_array((n,nf),dtype=np.bool_)
    values=cuda.device_array((size,1,1));leaves,levels,count,roots=_tree(size)
    d_roots=cuda.to_device(roots)
    d_leaves=cuda.to_device(leaves);d_levels=[cuda.to_device(a) for a in levels];sums=cuda.device_array(count)
    setup[(n*nf+127)//128,128](d_r,d_demand,d_umax,d_memory,mref,cap,unit)
    cuda.synchronize()
  gap=np.inf;n_iter=0
  for t in range(int(options['iterations'])):
    with step_time.parallel_phase():
      step_time.mark_shared_device()
      kappa=options['kappa']*options['kappa_growth']**t
      loads[(n*nf+127)//128,128](q,d_base,cap,unit,options['smoothing'],factors,slope)
      prices[(n+127)//128,128](factors,slope,unit,d_rho,mref,kappa,price)
      best_dest[(n*nf+127)//128,128](d_gains,d_mask,price,d_omega,best,positive)
      gap_values[(size+127)//128,128](q,d_gains,d_mask,price,d_omega,best,positive,values)
      leaf_sums[(len(leaves)+127)//128,128](values,d_leaves,sums)
      for nodes in d_levels:combine[(nodes.shape[0]+127)//128,128](nodes,sums)
      chunk_sum[1,1](d_roots,sums)
      gap=float(sums[:1].copy_to_host()[0]);n_iter=t+1
      converged=gap<=options['fw_tolerance']*scale and t>0
      if not converged:update[(size+127)//128,128](q,d_omega,best,positive,2./(t+2.))
      cuda.synchronize()
    if converged:break
  with step_time.parallel_phase():
    step_time.mark_shared_device()
    result=q.copy_to_host();last_price=price.copy_to_host();cuda.synchronize()
    # Reconstruct the existing final unsmoothed statistics from final q.
    # Last advertised price is from the last iteration, before its update.
    capacity=replicas*params['max_utilization'][None,:]/params['demand']
    unit_rep=params['memory_requirement'][None,:]*params['demand']/params['max_utilization'][None,:]/mref
    ell=base_load+result.sum(axis=0)
    violation=np.maximum(0.,(unit_rep*np.maximum(0.,ell-capacity)).sum(axis=1)-rho/mref)
    stats=dict(iterations=n_iter,gap=gap,relative_gap=gap/scale,violation=float(violation.sum()*mref),
      objective=float((gains*result).sum()-.5*options['kappa']*options['kappa_growth']**max(n_iter-1,0)*(violation**2).sum()),
      price=last_price)
  from utils.compute_backend import record_operation
  record_operation('relaxed_flows','cuda',n,shape=[n,nf],device_bytes=buffer_bytes(n,nf))
  return result,stats
