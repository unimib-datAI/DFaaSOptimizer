from pyomo.util.infeasible import log_infeasible_constraints
from pyomo.opt.results import SolverStatus
from pyomo.opt import TerminationCondition
import pyomo.environ as pyo
from abc import ABC, abstractmethod
from datetime import datetime
import logging

PYO_VAR_TYPE = pyo.NonNegativeReals
PYO_PARAM_TYPE = pyo.NonNegativeReals


class BaseAbstractModel():
  def __init__(self):
    self.model = pyo.AbstractModel()
    self.name = "BaseAbstractModel"
  
  # @abstractmethod
  def _provide_initial_solution(self, instance, initial_solution):
    pass
  
  def generate_instance(self, data: dict):
    return self.model.create_instance(data)
  
  def solve(
      self, 
      instance, 
      solver_options: dict, 
      solver_name: str = "glpk",
      initial_solution: dict = None
    ):
    # initialize solver and set options
    solver = pyo.SolverFactory(solver_name)
    for k, v in solver_options.items():
      solver.options[k] = v
    # provide initial solution (if any)
    warmstart = False
    if initial_solution is not None:
      instance = self._provide_initial_solution(instance, initial_solution)
      warmstart = True
    # solve
    s = datetime.now()
    results = solver.solve(instance, warmstart = warmstart)
    e = datetime.now()
    # check solver status
    get_solution_ok = results.solver.status == SolverStatus.ok
    if not get_solution_ok:
      if (results.solver.status == SolverStatus.aborted
        ) and (
          len(results.solution) > 0
        ):
        get_solution_ok = True
    solution = {
      "solver_status": str(results.solver.status),
      "solution_exists": get_solution_ok,
      "termination_condition": str(results.solver.termination_condition),
      "runtime": (e - s).total_seconds()
    }
    # get solution
    if get_solution_ok:
      # get solution
      for v in instance.component_objects(pyo.Var, active=True):
        solution[v.name] = []
        for idx in v:
          solution[v.name].append(pyo.value(v[idx]))
      # get objective function value
      solution["obj"] = pyo.value(instance.OBJ)
    else:
      instance.pprint()
    return solution


class BaseLoadManagementModel(BaseAbstractModel):
  def __init__(self):
    super().__init__()
    self.name = "BaseLoadManagementModel"
    ###########################################################################
    # Problem parameters
    ###########################################################################
    # number and set of nodes
    self.model.Nn = pyo.Param(within = pyo.NonNegativeIntegers)
    self.model.N = pyo.RangeSet(1, self.model.Nn)
    # number and set of functions
    self.model.Nf = pyo.Param(within = pyo.NonNegativeIntegers)
    self.model.F = pyo.RangeSet(1, self.model.Nf)
    # incoming load
    self.model.incoming_load = pyo.Param(
      self.model.N, self.model.F, within = PYO_PARAM_TYPE
    )
    # service demand
    self.model.demand = pyo.Param(
      self.model.N, self.model.F, within = PYO_PARAM_TYPE
    )
    # utilization threshold
    self.model.max_utilization = pyo.Param(
      self.model.F, within = pyo.NonNegativeReals, default = 0.8
    )
    # memory requirement
    self.model.memory_requirement = pyo.Param(
      self.model.F, within = pyo.NonNegativeIntegers
    )
    # memory capacity
    self.model.memory_capacity = pyo.Param(
      self.model.N, within = pyo.NonNegativeIntegers
    )


class BaseCentralizedModel(BaseLoadManagementModel):
  def __init__(self):
    super().__init__()
    self.name = "BaseCentralizedModel"
    ###########################################################################
    # Problem parameters
    ###########################################################################
    # neighborhood (n_{ij}=1 if neighbors)
    self.model.neighborhood = pyo.Param(
      self.model.N, self.model.N, within = pyo.Binary,
      default = 0
    )
    # objective function weights
    self.model.alpha = pyo.Param(
      self.model.N, self.model.F, within = pyo.NonNegativeReals, default = 1
    )
    self.model.beta = pyo.Param(
      self.model.N, self.model.N, self.model.F, 
      within = pyo.NonNegativeReals, default = 0.9
    )
    self.model.gamma = pyo.Param(
      self.model.N, self.model.F, 
      within = pyo.NonNegativeReals, default = 0.8
    )
    ###########################################################################
    # Problem variables
    ###########################################################################
    # number of enqueued requests
    self.model.x = pyo.Var(
      self.model.N, self.model.F, 
      domain = PYO_VAR_TYPE
    )
    # number of forwarded requests
    self.model.y = pyo.Var(
      self.model.N, self.model.N, self.model.F, 
      domain = PYO_VAR_TYPE
    )
    # number of rejected requests
    self.model.z = pyo.Var(
      self.model.N, self.model.F, 
      domain = PYO_VAR_TYPE
    )
    # number of reserved instances
    self.model.r = pyo.Var(
      self.model.N, self.model.F, 
      domain = pyo.NonNegativeIntegers
    )
    ###########################################################################
    # Constraints
    ###########################################################################
    self.model.no_traffic_loss = pyo.Constraint(
      self.model.N, self.model.F, 
      rule = self.no_traffic_loss
    )
    self.model.offload_only_to_neighbors = pyo.Constraint(
      self.model.N, self.model.N, self.model.F, rule = 
      self.offload_only_to_neighbors
    )
  
  @staticmethod
  def no_traffic_loss(model, n, f):
    return (
      model.x[n,f] + model.z[n,f] + sum(
        model.y[n,m,f] for m in model.N
      ) == model.incoming_load[n,f]
    )
  
  @staticmethod
  def offload_only_to_neighbors(model, n, m, f):
    return model.y[n,m,f] <= model.incoming_load[n,f] * model.neighborhood[n,m]


class LoadManagementModel(BaseCentralizedModel):
  """
  max sum(alpha * x + sum(beta * y) - gamma * z)
  """
  def __init__(self):
    super().__init__()
    self.name = "LoadManagementModel"
    ###########################################################################
    # Problem variables
    ###########################################################################
    # 1 if node i is fowarding any request of function f
    self.model.i_sends_f = pyo.Var(
      self.model.N, self.model.F, domain = pyo.Binary
    )
    # 1 if node i receives any request of function f
    self.model.i_receives_f = pyo.Var(
      self.model.N, self.model.F, domain = pyo.Binary
    )
    ###########################################################################
    # Constraints
    ###########################################################################
    self.model.utilization_equilibrium = pyo.Constraint(
      self.model.N, self.model.F, rule = self.utilization_equilibrium
    )
    self.model.utilization_equilibrium2 = pyo.Constraint(
      self.model.N, self.model.F, rule = self.utilization_equilibrium2
    )
    self.model.residual_capacity = pyo.Constraint(
      self.model.N, rule = self.residual_capacity
    )
    self.model.no_ping_pong1 = pyo.Constraint(
      self.model.N, self.model.F, rule = self.no_ping_pong1
    )
    self.model.no_ping_pong2 = pyo.Constraint(
      self.model.N, self.model.F, rule = self.no_ping_pong2
    )
    self.model.no_ping_pong3 = pyo.Constraint(
      self.model.N, self.model.F, rule = self.no_ping_pong3
    )
    ###########################################################################
    # Objective function
    ###########################################################################
    self.model.OBJ = pyo.Objective(
      rule = self.maximize_processing, sense = pyo.maximize
    )
  
  @staticmethod
  def utilization_equilibrium(model, n, f):
    return model.demand[n,f] * (
      model.x[n,f] + sum(
        model.y[m,n,f] for m in model.N
      )
    ) <= model.r[n,f] * model.max_utilization[f]
  
  @staticmethod
  def utilization_equilibrium2(model, n, f):
    return model.demand[n,f] * (
      model.x[n,f] + sum(
        model.y[m,n,f] for m in model.N
      )
    ) >= model.r[n,f] * model.max_utilization[f]
  
  @staticmethod
  def residual_capacity(model, n):
    return sum(
      model.r[n,f] * model.memory_requirement[f] for f in model.F
    ) <= model.memory_capacity[n]
  
  @staticmethod
  def no_ping_pong1(model, n, f):
    return sum(
      model.y[n,m,f] for m in model.N
    ) <= model.incoming_load[n,f] * model.i_sends_f[n,f]
  
  @staticmethod
  def no_ping_pong2(model, n, f):
    return sum(
      model.y[m,n,f] for m in model.N
    ) <= sum(
      model.incoming_load[m,f] for m in model.N
    ) * model.i_receives_f[n,f]
  
  @staticmethod
  def no_ping_pong3(model, n, f):
    return model.i_sends_f[n,f] + model.i_receives_f[n,f] <= 1
  
  @staticmethod
  def maximize_processing(model):
    return sum(
      sum(
        model.alpha[n,f] * model.x[n,f] / model.incoming_load[n,f] + sum(
          model.beta[n,m,f] * model.y[n,m,f] for m in model.N
        ) / model.incoming_load[n,f] - (
          model.gamma[n,f] * model.z[n,f]
        ) / model.incoming_load[n,f] for f in model.F
      ) for n in model.N
    )
