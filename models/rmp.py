from models.model import BaseLoadManagementModel, PYO_VAR_TYPE

import pyomo.environ as pyo


class RMPAbstractModel(BaseLoadManagementModel):
  def __init__(self):
    super().__init__()
    self.name = "RMPAbstractModel"
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
    # number of enqueued requests
    self.model.x_bar = pyo.Param(
      self.model.N, self.model.F, 
      domain = PYO_VAR_TYPE, default = 0
    )
    # number of reserved instances
    self.model.r_bar = pyo.Param(
      self.model.N, self.model.F, 
      domain = pyo.NonNegativeIntegers, default = 0
    )


class RMPCentralized(RMPAbstractModel):
  def __init__(self):
    super().__init__()
    self.name = "RMPCentralized"
    ###########################################################################
    # Problem variables
    ###########################################################################
    # number of forwarded requests
    self.model.y = pyo.Var(
      self.model.N, self.model.N, self.model.F, 
      domain = PYO_VAR_TYPE
    )
    ###########################################################################
    # Constraints
    ###########################################################################
    self.model.offload_only_to_neighbors = pyo.Constraint(
      self.model.N, self.model.N, self.model.F, 
      rule = self.offload_only_to_neighbors
    )
  
  @staticmethod
  def offload_only_to_neighbors(model, n, m, f):
    return model.y[n,m,f] <= model.incoming_load[n,f] * model.neighborhood[n,m]


class LRMP(RMPCentralized):
  def __init__(self):
    super().__init__()
    self.name = "LRMP"
    ###########################################################################
    # Problem parameters
    ###########################################################################
    # "desiderata" of forwarded requests
    self.model.omega_bar = pyo.Param(
      self.model.N, self.model.F, 
      within = PYO_VAR_TYPE, default = 0
    )
    self.model.gamma = pyo.Param(
      self.model.N, self.model.F, 
      within = pyo.NonNegativeReals, default = 0.8
    )
    ###########################################################################
    # Problem variables
    ###########################################################################
    # additional number of function replicas
    self.model.r = pyo.Var(
      self.model.N, self.model.F, 
      domain = pyo.NonNegativeIntegers
    )
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
    self.model.no_traffic_loss = pyo.Constraint(
      self.model.N, self.model.F, 
      rule = self.no_traffic_loss
    )
    self.model.capacity_constraints = pyo.Constraint(
      self.model.N, self.model.F, 
      rule = self.capacity_constraints
    )
    self.model.residual_capacity = pyo.Constraint(
      self.model.N,
      rule = self.residual_capacity
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
  
  def _provide_initial_solution(self, instance, initial_solution):
    Nn = instance.Nn.value
    Nf = instance.Nf.value
    # y
    y = initial_solution["y"].reshape((Nn,Nn,Nf))
    for n1 in range(1, Nn + 1):
      for n2 in range(1, Nn + 1):
        for f in range(1, Nf + 1):
          instance.y[(n1,n2,f)] = y[n1 - 1, n2 - 1, f - 1]
    # r
    r = initial_solution["r"].reshape((Nn,Nf))
    for n in range(1, Nn + 1):
      for f in range(1, Nf + 1):
        instance.r[(n,f)] = r[n - 1, f - 1]
    return instance
  
  @staticmethod
  def no_traffic_loss(model, n, f):
    return sum(model.y[n,m,f] for m in model.N) <= model.omega_bar[n,f]
  
  @staticmethod
  def capacity_constraints(model, n, f):
    return sum(
      model.y[m,n,f] for m in model.N
    ) <= model.max_utilization[f] * (
      model.r_bar[n,f] + model.r[n,f]
    ) / model.demand[n,f] - model.x_bar[n,f]
  
  @staticmethod
  def residual_capacity(model, n):
    return sum(
      model.r[n,f] * model.memory_requirement[f] for f in model.F
    ) <= model.memory_capacity[n] - sum(
      model.r_bar[n,f] * model.memory_requirement[f] for f in model.F
    )
  
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
        model.beta[n,m,f] * model.y[n,m,f] for m in model.N
      ) / model.incoming_load[n,f] for n in model.N for f in model.F
    ) - sum(
      model.gamma[n,f] * (
        model.omega_bar[n,f] - sum(model.y[n,m,f] for m in model.N)
      ) / model.incoming_load[n,f] for n in model.N for f in model.F
    )


class LRMP_freeMemory(RMPAbstractModel):
  def __init__(self):
    super().__init__()
    self.name = "LRMP_freeMemory"
    ###########################################################################
    # Problem parameters
    ###########################################################################
    # "desiderata" of forwarded requests
    self.model.omega_bar = pyo.Param(
      self.model.N, self.model.F, 
      within = PYO_VAR_TYPE, default = 0
    )
    ###########################################################################
    # Problem variables
    ###########################################################################
    # additional number of function replicas
    self.model.r = pyo.Var(
      self.model.N, self.model.F, 
      domain = pyo.NonNegativeIntegers
    )
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
    self.model.no_traffic_loss = pyo.Constraint(
      self.model.N, self.model.F, 
      rule = self.no_traffic_loss
    )
    self.model.capacity_constraints = pyo.Constraint(
      self.model.N, self.model.F, 
      rule = self.capacity_constraints
    )
    self.model.capacity_constraints2 = pyo.Constraint(
      self.model.N, self.model.F, 
      rule = self.capacity_constraints2
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
  def no_traffic_loss(model, n, f):
    return sum(model.y[n,m,f] for m in model.N) <= model.omega_bar[n,f]
  
  @staticmethod
  def capacity_constraints(model, n, f):
    return sum(
      model.y[m,n,f] for m in model.N
    ) <= model.max_utilization[f] * (
      model.r_bar[n,f] + model.r[n,f]
    ) / model.demand[n,f] - model.x_bar[n,f]
  
  @staticmethod
  def capacity_constraints2(model, n, f):
    return sum(
      model.y[m,n,f] for m in model.N
    ) >= model.max_utilization[f] * (
      model.r_bar[n,f] + model.r[n,f] - 1
    ) / model.demand[n,f] - model.x_bar[n,f]
  
  @staticmethod
  def no_ping_pong1(model, n, f):
    return sum(
      model.y[n,m,f] for m in model.N
    ) <= model.incoming_load[n,f] * model.i_sends_f[n,f]
  
  @staticmethod
  def no_ping_pong2(model, n, f):
    return sum(
      model.y[m,n,f] for m in model.N
    ) <= model.incoming_load[n,f] * model.i_receives_f[n,f]
  
  @staticmethod
  def no_ping_pong3(model, n, f):
    return model.i_sends_f[n,f] + model.i_receives_f[n,f] <= 1
  
  @staticmethod
  def maximize_processing(model):
    return sum(
      sum(
        model.beta[n,m,f] * model.y[n,m,f] for m in model.N
      ) / model.incoming_load[n,f] for n in model.N for f in model.F
    )
