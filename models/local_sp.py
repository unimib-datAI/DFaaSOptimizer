"""Select the opt-in local DP backend, with the configured Pyomo solver as fallback."""
from models.dp_solver import try_solve_local


def dp_enabled(solver_options):
  """Require a JSON boolean: strings such as "false" must not enable DP."""
  enabled = solver_options.get('use_dp', False)
  if not isinstance(enabled, bool):
    raise ValueError('use_dp must be a boolean')
  return enabled


def solve_agent_problem(model, data, solver_options, solver_name):
  """Use the same opt-in backend in sequential and multiprocessing SP calls."""
  if dp_enabled(solver_options):
    solution = try_solve_local(model, data)
    if solution is not None:
      return solution
  return model.solve(model.generate_instance(data), solver_options, solver_name)
