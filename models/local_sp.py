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


def solve_agent_batch(data, agents, model, solver_options, solver_name, prices=None):
  """Solve an explicitly requested CUDA group in input order, in this process."""
  if dp_device(solver_options) != 'cuda':
    raise ValueError('solve_agent_batch requires dp_device=cuda')
  from models.dp_solver import prepare_local_problem, finish_local_solution
  from models.dp_gpu import gpu_problem_fits, minimize_replica_costs_batch
  from plasma.core.sbm import minimize_replica_costs
  prepared, local_data, eligible, results = {}, {}, [], {}
  agents = list(agents)
  for agent in agents:
    local = data.copy()
    local[None] = data[None].copy()
    local[None]['whoami'] = {None: agent + 1}
    if prices is not None:
      local[None]['pi'] = {f + 1: price for f, price in enumerate(prices[agent])}
    local_data[agent] = local
    problem = prepare_local_problem(model, local)
    prepared[agent] = problem
    if (problem is not None and not problem.direct
        and gpu_problem_fits(problem.costs, problem.memory, problem.ram_cap)):
      eligible.append(agent)
  if len(eligible) >= 2:
    problems = [prepared[a] for a in eligible]
    choices = minimize_replica_costs_batch([p.costs for p in problems],
                                         [p.memory for p in problems],
                                         [p.ram_cap for p in problems])
    for agent, choice in zip(eligible, choices):
      if choice is not None:
        results[agent] = finish_local_solution(prepared[agent], choice, runtime=0.)
  for agent in agents:
    if agent not in results:
      results[agent] = solve_agent_problem(model, local_data[agent], solver_options, solver_name)
  return {a: results[a] for a in agents}
