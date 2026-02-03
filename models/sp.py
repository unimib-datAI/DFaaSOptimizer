from models.model import BaseLoadManagementModel, PYO_VAR_TYPE

import pyomo.environ as pyo


class SPAbstractModel(BaseLoadManagementModel):
  def __init__(self):
    super().__init__()
    self.name = "SPAbstractModel"
    ###########################################################################
    # Problem parameters
    ###########################################################################
    # index of the current node
    self.model.whoami = pyo.Param(
      within = pyo.NonNegativeIntegers, default = 1
    )
    # objective function weights
    self.model.alpha = pyo.Param(
      self.model.N, self.model.F, within = pyo.NonNegativeReals, default = 1
    )
    self.model.delta = pyo.Param(
      self.model.N, self.model.F, within = pyo.NonNegativeReals, default = 0.9
    )
    ###########################################################################
    # Problem variables
    ###########################################################################
    # number of enqueued requests
    self.model.x = pyo.Var(
      self.model.F, 
      domain = PYO_VAR_TYPE
    )
    # number of reserved instances
    self.model.r = pyo.Var(
      self.model.F, 
      domain = pyo.NonNegativeIntegers
    )


class LSP_v0(SPAbstractModel):
  def __init__(self):
    super().__init__()
    self.name = "LSP_v0"
    ###########################################################################
    # Problem parameters
    ###########################################################################
    # objective function weights
    self.model.pi = pyo.Param(
      self.model.F, 
      within = pyo.NonNegativeReals, default = 0.0
    )
    ###########################################################################
    # Problem variables
    ###########################################################################
    # number of forwarded requests
    self.model.omega = pyo.Var(
      self.model.F, 
      domain = PYO_VAR_TYPE
    )
    ###########################################################################
    # Constraints
    ###########################################################################
    self.model.no_traffic_loss_v0 = pyo.Constraint(
      self.model.F, rule = self.no_traffic_loss_v0
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
      rule = self.minimize_processing_cost_v0
    )
  
  @staticmethod
  def no_traffic_loss_v0(model, f):
    return (
      model.x[f] + model.omega[f]
    ) == model.incoming_load[model.whoami,f]
  
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
  def minimize_processing_cost_v0(model):
    return - (
      sum(
        (
          model.alpha[model.whoami,f] * model.x[f] + 
          model.delta[model.whoami,f] * model.omega[f]
        ) / model.incoming_load[model.whoami,f] for f in model.F
      )
    ) + sum(
      model.pi[f] * model.omega[f] / model.incoming_load[model.whoami,f] for f in model.F
    )

class LSP(LSP_v0):
  def __init__(self):
    super().__init__()
    self.name = "LSP"
    ###########################################################################
    # Problem parameters
    ###########################################################################
    # objective function weights
    self.model.gamma = pyo.Param(
      self.model.N, self.model.F, within = pyo.NonNegativeReals, default = 0.1
    )
    ###########################################################################
    # Problem variables
    ###########################################################################
    # number of rejected requests
    self.model.z = pyo.Var(
      self.model.F, 
      domain = PYO_VAR_TYPE
    )
    ###########################################################################
    # Constraints
    ###########################################################################
    self.model.no_traffic_loss = pyo.Constraint(
      self.model.F, rule = self.no_traffic_loss
    )
    ###########################################################################
    # Objective function
    ###########################################################################
    self.model.OBJ = pyo.Objective(
      rule = self.minimize_processing_cost
    )
  
  @staticmethod
  def no_traffic_loss(model, f):
    return (
      model.x[f] + model.omega[f] + model.z[f]
    ) == model.incoming_load[model.whoami,f]
  
  @staticmethod
  def minimize_processing_cost(model):
    return - (
      sum(
        (
          model.alpha[model.whoami,f] * model.x[f] + 
          model.delta[model.whoami,f] * model.omega[f] -
          model.gamma[model.whoami,f] * model.z[f]
        ) / model.incoming_load[model.whoami,f] for f in model.F
      )
    ) + sum(
      model.pi[f] * model.omega[f] / model.incoming_load[model.whoami,f] for f in model.F
    )


class LSPr_v0(SPAbstractModel):
  def __init__(self):
    super().__init__()
    self.name = "LSPr_v0"
    ###########################################################################
    # Problem parameters
    ###########################################################################
    # assigned offloading
    self.model.omega_bar = pyo.Param(
      self.model.N, self.model.F, 
      within = PYO_VAR_TYPE
    )
    self.model.y_bar = pyo.Param(
      self.model.N, self.model.N, self.model.F, 
      within = PYO_VAR_TYPE
    )
    ###########################################################################
    # Constraints
    ###########################################################################
    self.model.no_traffic_loss_v0 = pyo.Constraint(
      self.model.F, rule = self.no_traffic_loss_v0
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
      rule = self.minimize_processing_cost_v0
    )
  
  @staticmethod
  def no_traffic_loss_v0(model, f):
    return (
      model.x[f] + model.omega_bar[model.whoami,f]
    ) <= model.incoming_load[model.whoami,f]
  
  @staticmethod
  def utilization_equilibrium(model, f):
    return (
      model.demand[model.whoami,f] * (
        model.x[f] + sum(model.y_bar[m,model.whoami,f] for m in model.N)
      ) <= model.r[f] * model.max_utilization[f]
    )
  
  @staticmethod
  def utilization_equilibrium2(model, f):
    return (
      model.demand[model.whoami,f] * (
        model.x[f] + sum(model.y_bar[m,model.whoami,f] for m in model.N)
      ) >= (model.r[f] - 1) * model.max_utilization[f]
    )
  
  @staticmethod
  def residual_capacity(model):
    return sum(
      model.r[f] * model.memory_requirement[f] for f in model.F
    ) <= model.memory_capacity[model.whoami]
  
  @staticmethod
  def minimize_processing_cost_v0(model):
    return - (
      sum(
        (
          model.alpha[model.whoami,f] * model.x[f] + 
          model.delta[model.whoami,f] * model.omega_bar[model.whoami,f]
        ) / model.incoming_load[model.whoami,f] for f in model.F
      )
    )

class LSPr(LSPr_v0):
  def __init__(self):
    super().__init__()
    self.name = "LSPr"
    ###########################################################################
    # Problem parameters
    ###########################################################################
    # objective function weights
    self.model.gamma = pyo.Param(
      self.model.N, self.model.F, within = pyo.NonNegativeReals, default = 0.1
    )
    ###########################################################################
    # Problem variables
    ###########################################################################
    # number of rejected requests
    self.model.z = pyo.Var(
      self.model.F, 
      domain = PYO_VAR_TYPE
    )
    ###########################################################################
    # Constraints
    ###########################################################################
    self.model.no_traffic_loss = pyo.Constraint(
      self.model.F, rule = self.no_traffic_loss
    )
    ###########################################################################
    # Objective function
    ###########################################################################
    self.model.OBJ = pyo.Objective(
      rule = self.minimize_processing_cost
    )
  
  @staticmethod
  def no_traffic_loss(model, f):
    return (
      model.x[f] + model.omega_bar[model.whoami,f] + model.z[f]
    ) <= model.incoming_load[model.whoami,f]
  
  @staticmethod
  def minimize_processing_cost(model):
    return - (
      sum(
        (
          model.alpha[model.whoami,f] * model.x[f] + 
          model.delta[model.whoami,f] * model.omega_bar[model.whoami,f] -
          model.gamma[model.whoami,f] * model.z[f]
        ) / model.incoming_load[model.whoami,f] for f in model.F
      )
    )


##############################################################################
# TEMPORARILY FIX r
##############################################################################

class LSP_fixedr_v0(LSP_v0):
  def __init__(self):
    super().__init__()
    self.name = "LSP_fixedr_v0"
    ###########################################################################
    # Problem parameters
    ###########################################################################
    # number of assigned replicas
    self.model.r_bar = pyo.Param(
      self.model.N, self.model.F, 
      within = pyo.NonNegativeIntegers, default = 0
    )
    ###########################################################################
    # Constraints
    ###########################################################################
    self.model.fix_r = pyo.Constraint(
      self.model.F, rule = self.fix_r
    )
  
  @staticmethod
  def fix_r(model, f):
    return model.r[f] == model.r_bar[model.whoami,f]

class LSP_fixedr(LSP):
  def __init__(self):
    super().__init__()
    self.name = "LSP_fixedr"
    ###########################################################################
    # Problem parameters
    ###########################################################################
    # number of assigned replicas
    self.model.r_bar = pyo.Param(
      self.model.N, self.model.F, 
      within = pyo.NonNegativeIntegers, default = 0
    )
    ###########################################################################
    # Constraints
    ###########################################################################
    self.model.fix_r = pyo.Constraint(
      self.model.F, rule = self.fix_r
    )
  
  @staticmethod
  def fix_r(model, f):
    return model.r[f] == model.r_bar[model.whoami,f]


class LSPr_fixedr(LSPr):
  def __init__(self):
    super().__init__()
    self.name = "LSPr_fixedr"
    ###########################################################################
    # Problem parameters
    ###########################################################################
    # assigned offloading
    self.model.r_bar = pyo.Param(
      self.model.N, self.model.F, 
      within = pyo.NonNegativeIntegers
    )
    ###########################################################################
    # Constraints
    ###########################################################################
    self.model.fix_r = pyo.Constraint(
      self.model.F, rule = self.fix_r
    )
  
  @staticmethod
  def fix_r(model, f):
    return model.r[f] == model.r_bar[model.whoami,f]
