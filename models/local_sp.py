"""Select the opt-in local DP backend, with the configured Pyomo solver as fallback."""
from models.dp_solver import try_solve_local


def dp_enabled(solver_options):
  """Require a JSON boolean: strings such as "false" must not enable DP."""
  enabled = solver_options.get('use_dp', False)
  if not isinstance(enabled, bool):
    raise ValueError('use_dp must be a boolean')
  return enabled


def dp_device(solver_options):
  """Validate application device selection without importing CUDA."""
  enabled = dp_enabled(solver_options)
  device = solver_options.get('dp_device', 'cpu')
  if device not in ('cpu', 'cuda') or (device == 'cuda' and not enabled):
    raise ValueError('dp_device must be cpu or cuda; cuda requires use_dp=True')
  return device


def solve_agent_problem(model, data, solver_options, solver_name):
  """Use the same opt-in backend in sequential and multiprocessing SP calls."""
  dp_device(solver_options)
  if dp_enabled(solver_options):
    solution = try_solve_local(model, data)
    if solution is not None:
      return solution
  return model.solve(model.generate_instance(data), solver_options, solver_name)
