"""Independent buyer arithmetic; host ordering and ordered final reductions."""
from functools import lru_cache
import numpy as np
import pandas as pd
from utils import step_time
from utils.compute_backend import record_operation


@lru_cache(maxsize=1)
def _kernels():
  from numba import cuda
  from numba.cuda import libdevice as ld
  globals()['cuda'] = cuda

  @cuda.jit
  def scores(beta,gamma,latency,fairness,neighbors,lw,fw,out,eligible):
    k = cuda.grid(1)
    nf,nn = gamma.shape[1],gamma.shape[0]
    if k < nn*nn*nf:
      i,j,f = k//(nn*nf),(k//nf)%nn,k%nf
      score = ld.dadd_rn(ld.dadd_rn(ld.dadd_rn(beta[i,j,f],-ld.dmul_rn(lw,latency[i,j])),
                                   -ld.dmul_rn(fw,fairness[i,f])),gamma[i,f])
      ok = neighbors[i,j] != 0 and score > 0.
      eligible[i,j,f] = ok
      out[i,j,f] = score if ok else -float('inf')

  @cuda.jit
  def waterfill(omega,capacity,ranked,offsets,scores,quantities,best,terms):
    k = cuda.grid(1)
    nf = omega.shape[1]
    if k < omega.size:
      i,f = k//nf,k%nf
      remaining = omega[i,f]
      best[k],terms[k] = -1,0.
      if remaining > 0. and offsets[k] < offsets[k+1]:
        j = ranked[offsets[k]]
        best[k] = j
        terms[k] = ld.dmul_rn(remaining,scores[offsets[k]])
        for index in range(offsets[k],offsets[k+1]):
          quantity = min(remaining,capacity[ranked[index],f])
          if quantity > 0.:
            quantities[index] = quantity
            remaining = ld.dadd_rn(remaining,-quantity)
          if remaining <= 0.:
            break
  return cuda,scores,waterfill


def pair_scores_cuda(data,neighborhood,latency,fairness,dual_options):
  step_time.mark_shared_device()
  values=data[None];nn,nf=values['Nn'][None],values['Nf'][None]
  # Read precisely the same advertised neighboring entries as the CPU oracle.
  beta=np.zeros((nn,nn,nf));gamma=np.zeros((nn,nf))
  for i in range(nn):
    for j in np.flatnonzero(neighborhood[i]):
      for f in range(nf):
        beta[i,j,f]=values['beta'][i+1,int(j)+1,f+1]
        gamma[i,f]=values['gamma'][i+1,f+1]
  cuda,kernel,_=_kernels()
  if hasattr(latency,'toarray'):latency=latency.toarray()
  inputs=[cuda.to_device(np.array(a,order='C',copy=True)) for a in (beta,gamma,latency,fairness,neighborhood)]
  out=cuda.device_array(beta.shape,dtype=np.float64)
  eligible=cuda.device_array(beta.shape,dtype=np.bool_)
  kernel[(beta.size+127)//128,128](*inputs,dual_options['latency_weight'],dual_options['fairness_weight'],out,eligible)
  result=out.copy_to_host(),eligible.copy_to_host();cuda.synchronize()
  record_operation('pair_scores','cuda',nn,shape=list(beta.shape))
  return result


def buyer_price_response_cuda(omega,capacity,lam,s,elig):
  step_time.mark_shared_device()
  nn,nf=omega.shape
  # Keep CPU stable ordering by (-adjusted advantage, seller id).
  ranked=[];adjusted=[];offsets=[0]
  for i in range(nn):
    for f in range(nf):
      order=sorted(((int(j),s[i,j,f]-lam[j,f]) for j in np.flatnonzero(elig[i,:,f])
                    if s[i,j,f]-lam[j,f]>0.),key=lambda item:(-item[1],item[0])) if omega[i,f]>0 else []
      ranked.extend(j for j,_ in order);adjusted.extend(value for _,value in order)
      offsets.append(len(ranked))
  cuda,_,kernel=_kernels()
  arrays=[np.array(omega,dtype=np.float64,order='C',copy=True),np.array(capacity,dtype=np.float64,order='C',copy=True),
          np.array(ranked,dtype=np.int64),np.array(offsets,dtype=np.int64),np.array(adjusted,dtype=np.float64)]
  inputs=[cuda.to_device(a) for a in arrays]
  quantities=cuda.to_device(np.zeros(len(ranked)))
  best=cuda.device_array(nn*nf,dtype=np.int64);terms=cuda.device_array(nn*nf,dtype=np.float64)
  kernel[(nn*nf+127)//128,128](*inputs,quantities,best,terms)
  quantities,best,terms=quantities.copy_to_host(),best.copy_to_host(),terms.copy_to_host();cuda.synchronize()
  rows=[];demand=np.zeros((nn,nf));dual_term=0.
  # The CPU's row-major += sequence defines the reduction, including rounding.
  for i,f in zip(*np.nonzero(omega>0)):
    k=i*nf+f
    if best[k]<0:continue
    demand[best[k],f]+=omega[i,f];dual_term+=terms[k]
    for index in range(offsets[k],offsets[k+1]):
      if quantities[index]>0:
        j=ranked[index]
        rows.append(dict(i=i,j=j,f=f,d=quantities[index],utility=s[i,j,f]))
  record_operation('buyer_price_response','cuda',nn,shape=[nn,nf,len(ranked)])
  return pd.DataFrame(rows,columns=['i','j','f','d','utility']),demand,dual_term


@lru_cache(maxsize=1)
def _offer_kernels():
  from numba import cuda
  from numba.cuda import libdevice as ld
  import math
  globals()['cuda']=cuda

  @cuda.jit
  def utilities(parameters,internal,output):
    k=cuda.grid(1)
    if k<parameters.shape[0]:
      beta,gamma,load,price,latency,fairness,lw,fw=parameters[k]
      value=ld.ddiv_rn(ld.dadd_rn(beta,gamma),load if load else 1.) if internal else beta
      utility=ld.dadd_rn(ld.dadd_rn(ld.dadd_rn(value,-price),-ld.dmul_rn(lw,latency)),-ld.dmul_rn(fw,fairness))
      output[k,0]=value;output[k,1]=utility

  @cuda.jit
  def quantities(omega,offsets,capacities,unit,out,assigned):
    k=cuda.grid(1)
    if k<omega.size:
      total=0.
      for index in range(offsets[k],offsets[k+1]):
        if total>=omega[k]:break
        if unit:
          quantity=min(int(min(capacities[index],omega[k])),int(math.ceil(ld.dadd_rn(omega[k],-total))))
        else:
          quantity=int(min(capacities[index],ld.dadd_rn(omega[k],-total)))
        out[index]=quantity
        total=ld.dadd_rn(total,quantity)
      assigned[k]=total
  return cuda,utilities,quantities


def _offer_values(edges,data,prices,options,latency,fairness):
  """Ordered sparse edges; values use exactly the original arithmetic grouping."""
  cuda,kernel,_=_offer_kernels()
  v=data[None];parameters=np.empty((len(edges),8))
  internal=options.get('use_internal_value',False)
  for k,(i,j,f) in enumerate(edges):
    parameters[k]=(v['beta'][i+1,j+1,f+1],v['gamma'][i+1,f+1],
                   v['incoming_load'][i+1,f+1] if internal else 1.,prices[j,f],
                   latency[i,j],fairness[i,f],options['latency_weight'],options['fairness_weight'])
  if not edges:return np.empty((0,2))
  output=cuda.device_array((len(edges),2),dtype=np.float64)
  kernel[(len(edges)+127)//128,128](cuda.to_device(parameters),internal,output)
  result=output.copy_to_host();cuda.synchronize()
  return result


def _ordered_quantities(loads,orders,blackboard,functions,unit):
  cuda,_,kernel=_offer_kernels()
  offsets=np.array([0]+list(np.cumsum([len(order) for order in orders])),dtype=np.int64)
  capacities=np.array([blackboard[j,f] for order,f in zip(orders,functions) for j in order],dtype=np.float64)
  out=cuda.to_device(np.full(len(capacities),-1,dtype=np.int64))
  assigned=cuda.device_array(len(loads),dtype=np.float64)
  if len(loads):
    kernel[(len(loads)+127)//128,128](cuda.to_device(np.asarray(loads,dtype=np.float64)),cuda.to_device(offsets),cuda.to_device(capacities),unit,out,assigned)
  result=out.copy_to_host(),assigned.copy_to_host(),offsets;cuda.synchronize()
  return result


def define_bids_cuda(omega,blackboard,p,data,neighborhood,rho,auction_options,latency,fairness,
                     force_memory_bids,memory_bid_value=None):
  from run_faasmadea import VAR_TYPE,define_bids
  if VAR_TYPE is not int:
    record_operation('define_bids','cpu',len(omega),reason='fractional request domain')
    return define_bids(omega,blackboard,p,data,neighborhood,rho,auction_options,latency,fairness,
                       force_memory_bids,memory_bid_value)
  value_mode=getattr(memory_bid_value,'_madea2_gpu_mode',None)
  if memory_bid_value is not None and value_mode not in ('beta','bid'):
    # Custom Python callbacks can mutate state or consume RNG; preserve their path.
    from run_faasmadea import define_bids
    record_operation('define_bids','cpu',len(omega),reason='Python memory callback requires ordered host execution')
    return define_bids(omega,blackboard,p,data,neighborhood,rho,auction_options,latency,fairness,
                       force_memory_bids,memory_bid_value)
  step_time.mark_shared_device()
  pairs=list(zip(*np.nonzero(omega)));groups=[];edges=[]
  for i,f in pairs:
    sellers=set(np.nonzero(neighborhood[i,:])[0])
    capacity=sellers.intersection(set(np.where(blackboard[:,f]>=1)[0]))
    memory=sellers.intersection(set(np.nonzero(rho)[0]))
    start=len(edges);edges.extend((i,j,f) for j in capacity)
    groups.append((list(capacity),memory,capacity,start))
  values=_offer_values(edges,data,p,auction_options,latency,fairness)
  orders=[];utilities=[];seller_values=[]
  for (i,f),(sellers,_,_,start) in zip(pairs,groups):
    threshold=0. if auction_options.get('use_internal_value',False) else -data[None]['gamma'][i+1,f+1]
    candidates=[k for k in range(len(sellers)) if values[start+k,1]>threshold]
    utility=np.array([values[start+k,1] for k in candidates])
    # Use NumPy's same default argsort, including its non-stable tie order.
    order=np.argsort(utility)[::-1]
    orders.append([sellers[candidates[k]] for k in order])
    utilities.append(utility[order]);seller_values.append([values[start+candidates[k],0] for k in order])
  quantities,assigned,offsets=_ordered_quantities([omega[i,f] for i,f in pairs],orders,blackboard,
                                                [f for _,f in pairs],auction_options['unit_bids'])
  bids={k:[] for k in ('i','j','f','d','b','utility')};memory_bids={k:[] for k in ('i','j','f')}
  memory_values={}
  if value_mode is not None:
    memory_bids.update(d=[],v=[])
    needs_memory=[((assigned[k]<omega[i,f] and any(j in memory for j in orders[k]))
                   or ((assigned[k]<omega[i,f] or force_memory_bids) and bool(memory-capacity)))
                  for k,((i,f),(_,memory,capacity,_)) in enumerate(zip(pairs,groups))]
    memory_edges=[(i,j,f) for needed,(i,f),(_,memory,_,_) in zip(needs_memory,pairs,groups)
                  if needed for j in sorted(memory)]
    private=_private_values(memory_edges,data,auction_options,latency,fairness)
    for edge,value in zip(memory_edges,private):memory_values[edge]=value
    if value_mode=='bid':
      for needed,(i,f),(_,memory,_,_) in zip(needs_memory,pairs,groups):
        if not needed:continue
        utility={j:memory_values[i,j,f]-p[j,f] for j in sorted(memory)}
        order=sorted(sorted(memory),key=lambda j:-utility[j])
        for rank,j in enumerate(order):
          memory_values[i,j,f]=p[j,f]+auction_options['epsilon']+(utility[j]-utility[order[rank+1]] if rank+1<len(order) else 0.)
  def memory_row(i,j,f):
    memory_bids['i'].append(i);memory_bids['j'].append(j);memory_bids['f'].append(f)
    if value_mode is not None:
      memory_bids['d'].append(float(max(omega[i,f]-assigned[k],0.)))
      memory_bids['v'].append(memory_values[i,j,f])
  for k,((i,f),(_,memory,capacity,_),order,utility,value) in enumerate(zip(pairs,groups,orders,utilities,seller_values)):
    for rank,j in enumerate(order):
      q=int(quantities[offsets[k]+rank])
      if q<0:break
      delta=utility[rank]-utility[rank+1] if rank+1<len(order) else 0.
      price=p[j,f]+auction_options['epsilon']+delta
      if auction_options.get('use_internal_value',False):price=min(price,value[rank])
      unit=auction_options['unit_bids']
      for _ in range(q if unit else 1):
        for name,item in zip(bids,(i,j,f,1 if unit else q,price,utility[rank])):bids[name].append(item)
    if assigned[k]<omega[i,f]:
      for j in order:
        if j in memory:memory_row(i,j,f)
    if assigned[k]<omega[i,f] or force_memory_bids:
      for j in memory-capacity:memory_row(i,j,f)
  record_operation('define_bids','cuda',len(pairs),shape=[len(pairs),len(edges)])
  return pd.DataFrame(bids),pd.DataFrame(memory_bids),len(pairs)


@lru_cache(maxsize=1)
def _private_kernel():
  from numba import cuda
  from numba.cuda import libdevice as ld
  globals()['cuda']=cuda
  @cuda.jit
  def kernel(parameters,output):
    k=cuda.grid(1)
    if k<output.size:
      beta,gamma,latency,fairness,lw,fw=parameters[k]
      output[k]=ld.dadd_rn(ld.dadd_rn(ld.dadd_rn(beta,gamma),-ld.dmul_rn(lw,latency)),-ld.dmul_rn(fw,fairness))
  return cuda,kernel


def _private_values(edges,data,options,latency,fairness):
  if not edges:return np.empty(0)
  cuda,kernel=_private_kernel();v=data[None]
  parameters=np.array([(v['beta'][i+1,j+1,f+1],v['gamma'][i+1,f+1],latency[i,j],fairness[i,f],
                        options['latency_weight'],options['fairness_weight']) for i,j,f in edges],dtype=np.float64)
  output=cuda.device_array(len(edges),dtype=np.float64)
  kernel[(len(edges)+127)//128,128](cuda.to_device(parameters),output)
  result=output.copy_to_host();cuda.synchronize()
  return result


def define_assignments_cuda(omega,blackboard,data,neighborhood,rho,diffusion_options,latency,fairness,
                            force_memory_bids,current_y=None,tolerance=1e-6):
  from decentralized_diffusion import VAR_TYPE,define_assignments
  if VAR_TYPE is not int:
    record_operation('define_assignments','cpu',len(omega),reason='fractional request domain')
    return define_assignments(omega,blackboard,data,neighborhood,rho,diffusion_options,latency,fairness,
                              force_memory_bids,current_y,tolerance)
  from utils.centralized import ping_pong_forbidden_hosts
  step_time.mark_shared_device()
  if current_y is None:current_y=np.zeros((len(omega),len(omega),omega.shape[1]))
  forbidden=ping_pong_forbidden_hosts(omega,current_y,tolerance)
  pairs=list(zip(*np.nonzero(omega)));groups=[];edges=[]
  for i,f in pairs:
    sellers=set(np.nonzero(neighborhood[i,:])[0])
    sellers-={int(j) for j in np.nonzero(forbidden[:,f])[0]}
    capacity=sellers.intersection(set(np.where(blackboard[:,f]>=1)[0]))
    memory={int(j) for j in sellers if rho[int(j)]>=data[None]['memory_requirement'][f+1]}
    start=len(edges);edges.extend((i,j,f) for j in capacity)
    groups.append((list(capacity),memory,capacity,start))
  values=_offer_values(edges,data,np.zeros_like(omega),dict(diffusion_options,use_internal_value=False),latency,fairness)
  orders=[];utilities=[]
  for (i,f),(sellers,_,_,start) in zip(pairs,groups):
    candidates=[k for k in range(len(sellers)) if values[start+k,1]>-data[None]['gamma'][i+1,f+1]]
    order=sorted(candidates,key=lambda k:(-values[start+k,1],sellers[k]))
    orders.append([sellers[k] for k in order]);utilities.append([values[start+k,1] for k in order])
  quantities,assigned,offsets=_ordered_quantities([omega[i,f] for i,f in pairs],orders,blackboard,
                                                [f for _,f in pairs],diffusion_options.get('unit_bids',False))
  bids={k:[] for k in ('i','j','f','d','utility')};memory_bids={k:[] for k in ('i','j','f')}
  for k,((i,f),(_,memory,capacity,_),order,utility) in enumerate(zip(pairs,groups,orders,utilities)):
    for rank,j in enumerate(order):
      q=int(quantities[offsets[k]+rank])
      if q<0:break
      unit=diffusion_options.get('unit_bids',False)
      for _ in range(q if unit else 1):
        for name,item in zip(bids,(i,j,f,1 if unit else q,utility[rank])):bids[name].append(item)
    if assigned[k]<omega[i,f] or force_memory_bids:
      for j in order+[j for j in memory-capacity]:
        if j in memory:
          memory_bids['i'].append(i);memory_bids['j'].append(j);memory_bids['f'].append(f)
  record_operation('define_assignments','cuda',len(pairs),shape=[len(pairs),len(edges)])
  return pd.DataFrame(bids),pd.DataFrame(memory_bids),len(pairs)


def sample_assignments_cuda(omega,blackboard,data,neighborhood,rho,powerd_options,latency,fairness,
                            force_memory_bids,rng,current_y=None,tolerance=1e-6):
  from utils.centralized import ping_pong_forbidden_hosts
  from decentralized_powerd import _sample_assignments
  # Validate before any device work, including invalid sampling configurations.
  try:d=int(powerd_options['d'])
  except (TypeError,ValueError) as exc:raise ValueError('powerd d must be an integer >= 1') from exc
  if d<1:raise ValueError('powerd d must be >= 1')
  if powerd_options.get('criterion','score') not in ('score','capacity'):
    raise ValueError('powerd criterion must be one of: score, capacity')
  step_time.mark_shared_device()
  if current_y is None:current_y=np.zeros((len(omega),len(omega),omega.shape[1]))
  forbidden=ping_pong_forbidden_hosts(omega,current_y,tolerance)
  edges=[]
  for i,f in zip(*np.nonzero(omega)):
    sellers=set(np.nonzero(neighborhood[i,:])[0])
    sellers-={int(j) for j in np.nonzero(forbidden[:,f])[0]}
    sellers=sellers.intersection(set(np.where(blackboard[:,f]>=1)[0]))
    edges.extend((i,j,f) for j in sellers)
  values=_offer_values(edges,data,np.zeros_like(omega),dict(powerd_options,use_internal_value=False),latency,fairness)
  lookup={edge:value[1] for edge,value in zip(edges,values)}
  result=_sample_assignments(omega,blackboard,data,neighborhood,rho,powerd_options,latency,fairness,
                             force_memory_bids,rng,current_y,tolerance,score_lookup=lookup)
  record_operation('sample_assignments','cuda',len(omega),shape=[len(edges)],reason='numeric scores device; dependent sampling and remaining-capacity decisions host')
  return result


@lru_cache(maxsize=1)
def _hierarchy_kernel():
  from numba import cuda
  from numba.cuda import libdevice as ld
  globals()['cuda']=cuda
  @cuda.jit
  def kernel(parameters,out):
    k=cuda.grid(1)
    if k<out.size:
      structure_price,node_price,latency,fairness,epsilon,lw,fw=parameters[k]
      bid=max(ld.dadd_rn(structure_price,epsilon),node_price)
      out[k]=ld.dadd_rn(ld.dadd_rn(ld.dadd_rn(bid,-node_price),-ld.dmul_rn(lw,latency)),-ld.dmul_rn(fw,fairness))
  return cuda,kernel


def generate_level_requests_cuda(engine,buyer_structures,all_structures,current_y,node_prices,
                                  token_manager,latency,fairness,omega,eta,epsilon,
                                  latency_weight,fairness_weight,level):
  from hierarchical_auction import engine as module
  from hierarchical_auction.token_manager import CapacityTokenManager
  args=(buyer_structures,all_structures,current_y,node_prices,token_manager,latency,fairness,
        omega,eta,epsilon,latency_weight,fairness_weight,level)
  functions=(module.compute_structure_price,module.generate_structure_bid,module.compute_effective_bid,
             getattr(engine._collect_seller_nodes,'__func__',None))
  if functions!=module._GPU_REQUEST_FUNCTIONS or type(token_manager) is not CapacityTokenManager:
    record_operation('hierarchy_bid_scores','cpu',len(buyer_structures),reason='custom preparation callbacks or token manager require original ordered host execution')
    return engine._generate_level_requests_host(*args)
  sending=current_y.sum(axis=1)>1e-10;receiving=current_y.sum(axis=0)>1e-10
  prices={};sellers={};pairs=[];estimated=0
  for root,buyer in buyer_structures.items():
    prices[root]=module.compute_structure_price(buyer,node_prices,token_manager.tokens,eta=eta)
    for f in range(engine._num_functions):
      if not buyer.is_buyer(f):continue
      nodes=engine._collect_seller_nodes(buyer,all_structures,token_manager,f);sellers[root,f]=nodes
      buyers=[i for i in sorted(buyer.member_nodes) if omega[i,f]>1e-10]
      estimated+=len(nodes)*len(buyers)
      if estimated*64>64*1024*1024:
        record_operation('hierarchy_bid_scores','cpu',len(buyer_structures),reason='candidate preparation exceeds 64 MiB bound')
        return engine._generate_level_requests_host(*args)
      for i in buyers:
        for j in nodes:
          if j!=i and engine._neighborhood[i,j]>0 and not receiving[i,f] and not sending[j,f] and token_manager.available_tokens(j,f)>0:
            pairs.append((root,i,j,f))
  if not pairs:
    record_operation('hierarchy_bid_scores','cpu',len(buyer_structures),reason='no eligible candidate edges')
    return engine._generate_level_requests_host(*args)
  # The marker applies to the whole enclosing phase, including eligibility prep.
  step_time.mark_shared_device()
  if hasattr(latency,'toarray'):latency=latency.toarray()
  parameters=np.array([(prices[root][f],node_prices[j,f],latency[i,j],fairness[i,f],epsilon,latency_weight,fairness_weight)
                        for root,i,j,f in pairs],dtype=np.float64)
  cuda,kernel=_hierarchy_kernel();out=cuda.device_array(len(pairs),dtype=np.float64)
  kernel[(len(pairs)+127)//128,128](cuda.to_device(parameters),out)
  scores=out.copy_to_host();cuda.synchronize()
  record_operation('hierarchy_bid_scores','cuda',len(pairs),shape=list(parameters.shape),device_bytes=parameters.nbytes+scores.nbytes)
  prepared=dict(prices=prices,sellers=sellers,scores=dict(zip(pairs,scores)),sending=sending,receiving=receiving)
  return engine._generate_level_requests_host(*args,prepared=prepared)
