import pyomo.environ as pyo
from models.model import (
  BaseAbstractModel,
  BaseLoadManagementModel,
  BaseCentralizedModel,
  LoadManagementModel,
  PYO_VAR_TYPE,
  _solver_option_name,
)
from models.sp import (
  SPAbstractModel,
  LSP_v0,
  LSP,
  LSPr_v0,
  LSPr,
  LSP_fixedr_v0,
  LSP_fixedr,
  LSPr_fixedr,
)
from models.rmp import (
  RMPAbstractModel,
  RMPCentralized,
  LRMP,
  LRMP_freeMemory,
)
from models.auction_models import (
  BuyerNodeModel,
  SellerNodeModel,
  BuyerNodeModel_fixedr,
)
import models.model as model_module


class FakeSolveResults:
  def __init__(self):
    self.solver = type(
      "FakeSolverStatus",
      (),
      {
        "status": model_module.SolverStatus.ok,
        "termination_condition": model_module.TerminationCondition.optimal,
      },
    )()
    self.solution = []


class FakeInstance:
  OBJ = 0

  def component_objects(self, *_args, **_kwargs):
    return []


class FakeSolver:
  def __init__(self, fail_on_warmstart=False):
    self.options = {}
    self.fail_on_warmstart = fail_on_warmstart
    self.solve_kwargs = []

  def solve(self, _instance, **kwargs):
    self.solve_kwargs.append(kwargs)
    if self.fail_on_warmstart and "warmstart" in kwargs:
      raise ValueError("key 'warmstart' not defined")
    return FakeSolveResults()


class WarmstartModel(BaseAbstractModel):
  def _provide_initial_solution(self, instance, initial_solution):
    return instance


def test_base_abstract_model_has_name_and_model():
  m = BaseAbstractModel()
  assert m.name == "BaseAbstractModel"
  assert isinstance(m.model, pyo.AbstractModel)


def test_base_load_management_model_has_params_and_sets():
  m = BaseLoadManagementModel()
  assert m.name == "BaseLoadManagementModel"
  model = m.model
  assert model.Nn.domain == pyo.NonNegativeIntegers
  assert model.Nf.domain == pyo.NonNegativeIntegers
  assert model.N.ctype is not None
  assert model.F.ctype is not None
  assert model.incoming_load is not None
  assert model.demand is not None
  assert model.max_utilization is not None
  assert model.memory_requirement is not None
  assert model.memory_capacity is not None


def test_base_centralized_model_has_routing_vars_and_constraints():
  m = BaseCentralizedModel()
  assert m.name == "BaseCentralizedModel"
  model = m.model
  assert model.x is not None
  assert model.y is not None
  assert model.z is not None
  assert model.r is not None
  assert model.alpha is not None
  assert model.beta is not None
  assert model.gamma is not None
  assert model.neighborhood is not None


def test_base_centralized_model_constraint_rules():
  m = BaseCentralizedModel()
  assert m.no_traffic_loss is not None
  assert m.offload_only_to_neighbors is not None


def test_load_management_model_adds_ping_pong_and_objective():
  m = LoadManagementModel()
  assert m.name == "LoadManagementModel"
  model = m.model
  assert model.i_sends_f is not None
  assert model.i_receives_f is not None
  assert model.utilization_equilibrium is not None
  assert model.utilization_equilibrium2 is not None
  assert model.residual_capacity is not None
  assert model.no_ping_pong1 is not None
  assert model.no_ping_pong2 is not None
  assert model.no_ping_pong3 is not None
  assert model.OBJ is not None


def test_generate_instance_creates_concrete_model():
  m = LoadManagementModel()
  data = {
    None: {
      "Nn": {None: 2},
      "Nf": {None: 1},
      "incoming_load": {(1, 1): 10.0, (2, 1): 5.0},
      "demand": {(1, 1): 1.0, (2, 1): 2.0},
      "max_utilization": {1: 0.8},
      "memory_requirement": {1: 2},
      "memory_capacity": {1: 10, 2: 20},
      "alpha": {(1, 1): 1.0, (2, 1): 1.0},
      "beta": {
        (1, 1, 1): 0.0, (1, 2, 1): 0.9, (2, 1, 1): 0.9, (2, 2, 1): 0.0,
      },
      "gamma": {(1, 1): 0.1, (2, 1): 0.1},
      "neighborhood": {(1, 1): 0, (1, 2): 1, (2, 1): 1, (2, 2): 0},
    }
  }
  instance = m.generate_instance(data)
  assert instance.Nn.value == 2
  assert instance.Nf.value == 1


def test_base_abstract_model_generate_instance():
  m = BaseAbstractModel()
  data = {None: {}}
  instance = m.generate_instance(data)
  assert instance is not None


def test_base_abstract_model_solve_method_exists():
  m = BaseAbstractModel()
  assert hasattr(m, "solve")


def test_solve_does_not_pass_disabled_warmstart(monkeypatch):
  solver = FakeSolver()
  monkeypatch.setattr(model_module.pyo, "SolverFactory", lambda _name: solver)

  solution = BaseAbstractModel().solve(FakeInstance(), {}, solver_name="glpk")

  assert solution["solution_exists"] is True
  assert solver.solve_kwargs == [{}]


def test_solve_maps_glpk_time_limit_option(monkeypatch):
  solver = FakeSolver()
  monkeypatch.setattr(model_module.pyo, "SolverFactory", lambda _name: solver)

  solution = BaseAbstractModel().solve(FakeInstance(), {"TimeLimit": 7}, solver_name="glpk")

  assert solution["solution_exists"] is True
  assert solver.options == {"tmlim": 7}


def test_solver_option_name_keeps_time_limit_for_non_glpk():
  assert _solver_option_name("gurobi", "TimeLimit") == "TimeLimit"


def test_solve_passes_enabled_warmstart_for_initial_solution(monkeypatch):
  solver = FakeSolver()
  monkeypatch.setattr(model_module.pyo, "SolverFactory", lambda _name: solver)

  solution = WarmstartModel().solve(FakeInstance(), {}, initial_solution={}, solver_name="gurobi")

  assert solution["solution_exists"] is True
  assert solver.solve_kwargs == [{"warmstart": True}]


def test_solve_retries_without_warmstart_when_solver_rejects_it(monkeypatch):
  solver = FakeSolver(fail_on_warmstart=True)
  monkeypatch.setattr(model_module.pyo, "SolverFactory", lambda _name: solver)

  solution = WarmstartModel().solve(FakeInstance(), {}, initial_solution={}, solver_name="glpk")

  assert solution["solution_exists"] is True
  assert solver.solve_kwargs == [{"warmstart": True}, {}]


def test_base_load_management_model_inherits_from_base_abstract():
  m = BaseLoadManagementModel()
  assert isinstance(m, BaseAbstractModel)


def test_base_centralized_model_inherits_from_base_load_management():
  m = BaseCentralizedModel()
  assert isinstance(m, BaseLoadManagementModel)


def test_load_management_model_inheritance_chain():
  m = LoadManagementModel()
  assert isinstance(m, BaseCentralizedModel)


# ---- SP models ----


def test_sp_abstract_model_has_whoami_and_vars():
  m = SPAbstractModel()
  assert m.name == "SPAbstractModel"
  model = m.model
  assert model.whoami is not None
  assert model.x is not None
  assert model.r is not None
  assert model.alpha is not None
  assert model.delta is not None


def test_lsp_v0_model_has_omega_and_constraints():
  m = LSP_v0()
  assert m.name == "LSP_v0"
  model = m.model
  assert model.omega is not None
  assert model.pi is not None
  assert model.no_traffic_loss_v0 is not None
  assert model.utilization_equilibrium is not None
  assert model.utilization_equilibrium2 is not None
  assert model.residual_capacity is not None
  assert model.OBJ is not None


def test_lsp_model_adds_rejections():
  m = LSP()
  assert m.name == "LSP"
  model = m.model
  assert model.z is not None
  assert model.gamma is not None
  assert model.no_traffic_loss is not None
  assert model.OBJ is not None


def test_lspr_v0_model_has_assigned_offloading():
  m = LSPr_v0()
  assert m.name == "LSPr_v0"
  model = m.model
  assert model.omega_bar is not None
  assert model.y_bar is not None
  assert model.no_traffic_loss_v0 is not None


def test_lspr_model_adds_rejections_to_lspr_v0():
  m = LSPr()
  assert m.name == "LSPr"
  model = m.model
  assert model.z is not None
  assert model.gamma is not None
  assert model.no_traffic_loss is not None


def test_lsp_fixedr_v0_fixes_replicas():
  m = LSP_fixedr_v0()
  assert m.name == "LSP_fixedr_v0"
  model = m.model
  assert model.r_bar is not None
  assert model.fix_r is not None


def test_lsp_fixedr_fixes_replicas_with_rejections():
  m = LSP_fixedr()
  assert m.name == "LSP_fixedr"
  model = m.model
  assert model.r_bar is not None
  assert model.fix_r is not None
  assert model.z is not None


def test_lspr_fixedr_fixes_replicas_for_restricted():
  m = LSPr_fixedr()
  assert m.name == "LSPr_fixedr"
  model = m.model
  assert model.r_bar is not None
  assert model.fix_r is not None


# ---- RMP models ----


def test_rmp_abstract_model_has_xbar_and_rbar():
  m = RMPAbstractModel()
  assert m.name == "RMPAbstractModel"
  model = m.model
  assert model.neighborhood is not None
  assert model.beta is not None
  assert model.x_bar is not None
  assert model.r_bar is not None


def test_rmp_centralized_has_y_and_offload_constraint():
  m = RMPCentralized()
  assert m.name == "RMPCentralized"
  model = m.model
  assert model.y is not None
  assert model.offload_only_to_neighbors is not None


def test_lrmp_model_has_full_formulation():
  m = LRMP()
  assert m.name == "LRMP"
  model = m.model
  assert model.omega_bar is not None
  assert model.gamma is not None
  assert model.r is not None
  assert model.i_sends_f is not None
  assert model.i_receives_f is not None
  assert model.no_traffic_loss is not None
  assert model.capacity_constraints is not None
  assert model.residual_capacity is not None
  assert model.no_ping_pong1 is not None
  assert model.no_ping_pong2 is not None
  assert model.no_ping_pong3 is not None
  assert model.OBJ is not None


def test_lrmp_free_memory_model_has_capacity_constraints2():
  m = LRMP_freeMemory()
  assert m.name == "LRMP_freeMemory"
  model = m.model
  assert model.omega_bar is not None
  assert model.r is not None
  assert model.capacity_constraints is not None
  assert model.capacity_constraints2 is not None
  assert model.no_traffic_loss is not None


# ---- Auction models ----


def test_buyer_node_model_has_demand_and_bids():
  m = BuyerNodeModel()
  assert m.name == "BuyerNodeModel"
  model = m.model
  assert model.neighborhood is not None
  assert model.beta is not None
  assert model.gamma is not None
  assert model.c is not None
  assert model.d is not None
  assert model.z is not None
  assert model.offload_only_to_neighbors is not None
  assert model.no_traffic_loss is not None
  assert model.OBJ is not None


def test_seller_node_model_has_d_bar_and_whoami():
  m = SellerNodeModel()
  assert m.name == "SellerNodeModel"
  model = m.model
  assert model.whoami is not None
  assert model.d_bar is not None
  assert model.r is not None
  assert model.d is not None
  assert model.no_traffic_loss is not None
  assert model.utilization_equilibrium is not None
  assert model.utilization_equilibrium2 is not None
  assert model.residual_capacity is not None
  assert model.OBJ is not None


def test_buyer_node_model_fixedr_fixes_replicas():
  m = BuyerNodeModel_fixedr()
  assert m.name == "BuyerNodeModel_fixedr"
  model = m.model
  assert model.r_bar is not None
  assert model.fix_r is not None


def test_pyo_var_type_is_non_negative_reals():
  assert PYO_VAR_TYPE == pyo.NonNegativeReals
