"""Numerical LP and branch-and-cut upper bounds on VNS welfare."""
import argparse
import json

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import bmat, coo_matrix, csr_matrix, diags, eye, hstack, vstack

from centralized_vns.io import load_instance
from .io import load_solution
from .xstage import local_gains


def welfare_bound(problem, y, variable_x=False, *, mip_seconds=None, log=False):
  p=problem
  incumbent=p.check(y)['welfare']
  nodes,functions=p.load.shape
  cells=p.load.size
  denom=np.maximum(1,p.load)
  cell=np.arange(cells)
  node=cell//functions
  edges=np.arange(len(y))
  def sparse(values, rows, columns, shape):
    return coo_matrix((values,(rows,columns)),shape=shape).tocsr()
  def incidence(endpoint):
    return sparse(np.ones(len(edges)),endpoint*functions+p.fun,edges,(cells,len(edges)))
  identity=eye(cells,format='csr')
  ram=sparse(np.tile(p.memory,nodes),node,cell,(nodes,cells))
  local=sparse(-(p.alpha/denom).ravel(),node,cell,(nodes,cells))
  # Variables in column order: local service, forwarded flows, receiver replicas.
  rows=[[identity,incidence(p.src),None],
        [diags(1/p.capacity.ravel()),diags(1/p.capacity.ravel())@incidence(p.dst),-identity],
        [None,None,ram]]
  rhs=[p.load.ravel(),np.full(cells,1e-9),p.ram]
  if variable_x:
    rows.append([local,None,None])
    rhs.append(-local_gains(p,p.x)+1e-12)
  matrix=bmat(rows,format='csr')
  rhs=np.concatenate(rhs)
  rewards=np.r_[((p.alpha+p.gamma)/denom).ravel(),p.reward,np.zeros(cells)]
  minimum=np.r_[np.zeros(cells) if variable_x else p.x.ravel(), np.zeros(len(edges)),
                np.zeros(cells) if variable_x else p.check(np.zeros(len(edges)))['replicas'].ravel()]
  maximum=np.r_[p.load.ravel() if variable_x else p.x.ravel(),p.load[p.src,p.fun],
                (p.ram[:,None]//p.memory).ravel()]
  result=linprog(-rewards,A_ub=matrix,b_ub=rhs,bounds=np.column_stack((minimum,maximum)),method='highs')
  if not result.success:
    raise RuntimeError(result.message)
  # Any nonnegative multiplier gives a dual upper bound after maximizing the
  # remaining linear objective over the finite variable bounds.
  multipliers=np.maximum(0.,-result.ineqlin.marginals)
  residual=rewards-matrix.T@multipliers
  upper=float(-np.sum(p.gamma*p.load/denom)+rhs@multipliers+
              np.maximum(residual*minimum,residual*maximum).sum())
  if upper < incumbent-1e-7*max(1.,abs(incumbent)):
    raise ArithmeticError('upper bound below feasible incumbent')
  details={}
  if mip_seconds is not None:
    if not np.isfinite(mip_seconds) or mip_seconds<=0:
      raise ValueError('mip_seconds must be positive and finite')
    # A binary sender flag forbids receiving the same function while forwarding.
    available=(p.load if variable_x else p.load-p.x).ravel()
    outgoing=incidence(p.src)
    incoming=incidence(p.dst)
    edge_limit=available[p.src*functions+p.fun]
    receive_limit=np.asarray(incoming@edge_limit).ravel()
    zeros=csr_matrix((cells,cells))
    role_matrix=vstack([
      hstack([matrix,csr_matrix((matrix.shape[0],cells))]),
      hstack([zeros,outgoing,zeros,-diags(available)]),
      hstack([zeros,incoming,zeros,diags(receive_limit)])],format='csr')
    bounds=Bounds(np.r_[minimum,np.zeros(cells)],np.r_[maximum,np.ones(cells)])
    costs=np.r_[-rewards,np.zeros(cells)]
    constraints=LinearConstraint(role_matrix,-np.inf,np.r_[rhs,np.zeros(cells),receive_limit])
    options=dict(time_limit=mip_seconds,mip_rel_gap=0.,disp=log)
    mip=milp(costs,integrality=np.ones(len(costs)),bounds=bounds,
             constraints=constraints,options=options)
    retry=mip.status==2
    if retry:
      # The checked input is feasible. Verify its model encoding before retrying.
      flow=np.asarray(y,dtype=np.int64)
      witness=np.r_[p.x.ravel(),flow,p.check(flow)['replicas'].ravel(),(outgoing@flow)>0]
      if np.max(role_matrix@witness-constraints.ub)>1e-7:
        raise ArithmeticError('MIP constraints exclude the checked incumbent')
      mip=milp(costs,integrality=np.ones(len(costs)),bounds=bounds,
               constraints=constraints,options={**options,'presolve':False})
    if mip.status not in (0,1):
      raise RuntimeError(mip.message)
    dual=getattr(mip,'mip_dual_bound',None)
    mip_upper=None if dual is None or not np.isfinite(dual) else float(-np.sum(p.gamma*p.load/denom)-dual)
    details=dict(retried_without_presolve=retry,lp_upper=upper,mip_upper=mip_upper,mip_status=mip.message,
                 nodes=getattr(mip,'mip_node_count',None))
    if mip_upper is not None:
      if mip_upper<incumbent-1e-7*max(1.,abs(incumbent)):
        raise ArithmeticError('MIP dual bound below feasible incumbent')
      upper=min(upper,mip_upper)
  return dict(lower=incumbent,upper=upper,gap_percent=100*(upper-incumbent)/max(1.,abs(incumbent)),**details)


if __name__=='__main__':
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('instance')
  parser.add_argument('solution',help='fixed-stage solution NPZ, defining initial x and floors')
  parser.add_argument('--step',type=int,default=0)
  parser.add_argument('--variable-x',action='store_true')
  parser.add_argument('--mip-seconds',type=float)
  parser.add_argument('--log',action='store_true')
  args=parser.parse_args()
  data,_=load_instance(args.instance,args.step)
  problem,y,_=load_solution(data,args.solution)
  print(json.dumps(welfare_bound(problem,y,args.variable_x,mip_seconds=args.mip_seconds,log=args.log),indent=2))
