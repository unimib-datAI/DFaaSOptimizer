from models.rmp import RMPAbstractModel
from models.sp import SPAbstractModel
from models.model import PYO_VAR_TYPE

import pyomo.environ as pyo


class BuyerNodeModel(SPAbstractModel):
  def __init__(self):
    super().__init__()
    self.name = "BuyerNodeModel"
    ###########################################################################
    # Problem parameters
    ###########################################################################
    # neighborhood (n_{ij}=1 if neighbors)
    self.model.neighborhood = pyo.Param(
      self.model.N, self.model.N, within = pyo.Binary,
      default = 0
    )
    # objective function weights
    self.model.beta = pyo.Param(
      self.model.N, self.model.N, self.model.F, 
      within = pyo.Reals, default = 0.9
    )
    self.model.gamma = pyo.Param(
      self.model.N, self.model.F, 
      within = pyo.NonNegativeReals, default = 0.8
    )
    ###########################################################################
    # Problem variables
    ###########################################################################
    # number of forwarded requests
    self.model.d = pyo.Var(
      self.model.N, self.model.F, 
      domain = PYO_VAR_TYPE
    )
    # number of rejected requests
    self.model.z = pyo.Var(
      self.model.F, 
      domain = PYO_VAR_TYPE
    )
    ###########################################################################
    # Constraints
    ###########################################################################
    self.model.offload_only_to_neighbors = pyo.Constraint(
      self.model.N, self.model.F, 
      rule = self.offload_only_to_neighbors
    )
    self.model.no_traffic_loss = pyo.Constraint(
      self.model.F, rule = self.no_traffic_loss
    )
    self.model.utilization_equilibrium = pyo.Constraint(
      self.model.F, rule = self.utilization_equilibrium
    )
    self.model.utilization_equilibrium2 = pyo.Constraint(
      self.model.F, rule = self.utilization_equilibrium2
    )
    self.model.residual_capacity = pyo.Constraint(
      rule = self.residual_capacity
    )
    ###########################################################################
    # Objective function
    ###########################################################################
    self.model.OBJ = pyo.Objective(
      rule = self.maximize_utility, sense = pyo.maximize
    )
  
  @staticmethod
  def no_traffic_loss(model, f):
    return model.x[f] + sum(
      model.d[n,f] for n in model.N
    ) + model.z[f] == model.incoming_load[model.whoami,f]
  
  @staticmethod
  def offload_only_to_neighbors(model, m, f):
    return model.d[m,f] <= (
      model.incoming_load[model.whoami,f] * model.neighborhood[model.whoami,m]
    )
  
  @staticmethod
  def utilization_equilibrium(model, f):
    return (
      model.demand[model.whoami,f] * (
        model.x[f]
      ) <= model.r[f] * model.max_utilization[f]
    )
  
  @staticmethod
  def utilization_equilibrium2(model, f):
    return (
      model.demand[model.whoami,f] * (
        model.x[f]
      ) >= (model.r[f] - 1) * model.max_utilization[f]
    )
  
  @staticmethod
  def residual_capacity(model):
    return sum(
      model.r[f] * model.memory_requirement[f] for f in model.F
    ) <= model.memory_capacity[model.whoami]
  
  @staticmethod
  def maximize_utility(model):
    return sum(
      (
        model.alpha[model.whoami,f] * model.x[f] + sum(
          model.beta[model.whoami,n,f] * model.d[n,f] for n in model.N
        ) - model.gamma[model.whoami,f] * model.z[f]
      # ) / model.incoming_load[model.whoami,f] for f in model.F
      ) for f in model.F
    )


class SellerNodeModel(RMPAbstractModel):
  def __init__(self):
    super().__init__()
    self.name = "SellerNodeModel"
    ###########################################################################
    # Problem parameters
    ###########################################################################
    # index of the current node
    self.model.whoami = pyo.Param(
      within = pyo.NonNegativeIntegers, default = 1
    )
    # "desiderata" of forwarded requests
    self.model.d_bar = pyo.Param(
      self.model.N, self.model.N, self.model.F, 
      within = PYO_VAR_TYPE, default = 0
    )
    ###########################################################################
    # Problem variables
    ###########################################################################
    # additional number of function replicas
    self.model.r = pyo.Var(
      self.model.F, 
      domain = pyo.NonNegativeIntegers
    )
    # number of forwarded requests
    self.model.d = pyo.Var(
      self.model.N, self.model.F, 
      domain = PYO_VAR_TYPE
    )
    ###########################################################################
    # Constraints
    ###########################################################################
    self.model.no_traffic_loss = pyo.Constraint(
      self.model.N, self.model.F, 
      rule = self.no_traffic_loss
    )
    self.model.utilization_equilibrium = pyo.Constraint(
      self.model.F, rule = self.utilization_equilibrium
    )
    self.model.utilization_equilibrium2 = pyo.Constraint(
      self.model.F, rule = self.utilization_equilibrium2
    )
    self.model.residual_capacity = pyo.Constraint(
      rule = self.residual_capacity
    )
    ###########################################################################
    # Objective function
    ###########################################################################
    self.model.OBJ = pyo.Objective(
      rule = self.maximize_processing, sense = pyo.maximize
    )
  
  @staticmethod
  def no_traffic_loss(model, n, f):
    return model.d[n,f] <= model.d_bar[n,model.whoami,f]
  
  @staticmethod
  def utilization_equilibrium(model, f):
    return sum(
      model.d[n,f] for n in model.N
    ) <= model.max_utilization[f] * (
      model.r_bar[model.whoami,f] + model.r[f]  
    ) / model.demand[model.whoami,f] - model.x_bar[model.whoami,f]
  
  @staticmethod
  def utilization_equilibrium2(model, f):
    return sum(
      model.d[n,f] for n in model.N
    ) >= model.max_utilization[f] * (
      model.r_bar[model.whoami,f] + model.r[f] - 1
    ) / model.demand[model.whoami,f] - model.x_bar[model.whoami,f]
  
  @staticmethod
  def residual_capacity(model):
    return sum(
      model.r[f] * model.memory_requirement[f] for f in model.F
    ) <= model.memory_capacity[model.whoami] - sum(
      model.r_bar[model.whoami,f] * model.memory_requirement[f] for f in model.F
    )
  
  @staticmethod
  def maximize_processing(model):
    return sum(
      sum(
        model.beta[n,model.whoami,f] * model.d[n,f] for n in model.N
      ) for f in model.F
    )
