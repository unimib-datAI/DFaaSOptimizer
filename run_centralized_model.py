from utilities import delete_tuples, NpEncoder, load_configuration
from utilities import float_to_int
from generate_data import generate_data, update_data
from load_generator import LoadGenerator
from model import BaseLoadManagementModel, LoadManagementModel, PYO_VAR_TYPE
from postprocessing import plot_history

import matplotlib.pyplot as plt
from datetime import datetime
import pyomo.environ as pyo
from pprint import pprint
from typing import Tuple
import pandas as pd
import numpy as np
import argparse
import json
import sys
import os


VAR_TYPE = int if PYO_VAR_TYPE == pyo.NonNegativeIntegers else float


def parse_arguments() -> argparse.Namespace:
   """
   Parse input arguments
   """
   parser: argparse.ArgumentParser = argparse.ArgumentParser(
     description = "run", 
     formatter_class=argparse.ArgumentDefaultsHelpFormatter
   )
   parser.add_argument(
     "-c", "--config",
     help = "Configuration file",
     type = str,
     default = "manual_config.json"
   )
   parser.add_argument(
     "--disable_plotting",
     default = False,
     action = "store_true"
   )
   # Parse the arguments
   args: argparse.Namespace = parser.parse_known_args()[0]
   return args


def compute_residual_capacity(data: dict, r: np.array) -> np.array:
  Nn = data[None]["Nn"][None]
  rho = np.zeros((Nn,))
  indices = data.get("indices", range(Nn))
  for n in indices:
    memory_capacity = data[None]["memory_capacity"][n+1]
    rho_idx = n if len(indices) > 1 else 0
    rho[rho_idx] = memory_capacity
    for f, memory_requirement in data[None]["memory_requirement"].items():
      rho[rho_idx] -= (memory_requirement * r[rho_idx,f-1])
  return rho


def compute_utilization(data: dict, solution: dict) -> np.array:
  x, y, _, r, xi, _, _, _ = extract_solution(data, solution)
  demand = data[None]["demand"]
  utilization = np.zeros(shape = x.shape)
  for (n,f), d in demand.items():
    if r[n-1, f-1] > 0:
      if xi is None or ((xi == 0).all() and (y != 0).any()):
        utilization[n-1, f-1] = d * (
          x[n-1, f-1] + y[:, n-1, f-1].sum()
        ) / r[n-1, f-1]
      else:
        utilization[n-1, f-1] = d * (
          x[n-1, f-1] + xi[n-1, :, f-1].sum()
        ) / r[n-1, f-1]
  return utilization


def count_offloaded_processing(
    detailed_offloading: pd.DataFrame, Nn: int, Nf: int
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
  offloaded_processing = pd.DataFrame()
  detailed_offloaded_processing = pd.DataFrame()
  for n in range(Nn):
    for f in range(Nf):
      offloaded_processing[f"n{n}_f{f}_accepted"] = detailed_offloading.loc[
        :,detailed_offloading.columns.str.endswith(f"_f{f}_n{n}")
      ].sum(axis = "columns")
  return offloaded_processing, detailed_offloaded_processing


def decode_solution(
    x: np.array, 
    y: np.array, 
    z: np.array, 
    r: np.array, 
    xi: np.array, 
    rho: np.array,
    U: np.array,
    complete_solution: dict,
  ) -> dict:
  Nn, Nf = x.shape
  # local processing
  complete_solution["local_processing"] = update_2d_variables(
    x, complete_solution["local_processing"]
  )
  # offloading
  (
    complete_solution["offloading"], complete_solution["detailed_offloading"]
  ) = update_3d_variables(
    y, complete_solution["offloading"], complete_solution["detailed_offloading"]
  )
  # rejections
  complete_solution["rejections"] = update_2d_variables(
    z, complete_solution["rejections"]
  )
  # number of reserved instances
  complete_solution["replicas"] = update_2d_variables(
    r, complete_solution["replicas"]
  )
  # utilization
  complete_solution["utilization"] = update_2d_variables(
    U, complete_solution["utilization"]
  )
  # received offloading
  if xi is not None:
    (
      complete_solution["offloaded_processing"], 
      complete_solution["detailed_offloaded_processing"]
    ) = update_3d_variables(
      xi, 
      complete_solution["offloaded_processing"], 
      complete_solution["detailed_offloaded_processing"]
    )
  else:
    (
      complete_solution["offloaded_processing"], 
      complete_solution["detailed_offloaded_processing"]
    ) = count_offloaded_processing(
      complete_solution["detailed_offloading"], Nn, Nf
    )
  # residual capacity
  complete_solution["residual_capacity"] = update_1d_variables(
    rho, complete_solution["residual_capacity"]
  )
  return complete_solution


def encode_solution(
    Nn: int, Nf: int,
    solution: pd.DataFrame, 
    detailed_fwd_solution: pd.DataFrame, 
    replicas: pd.DataFrame,
    t: int
  ) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
  x = np.zeros((Nn,Nf))
  y = np.zeros((Nn,Nn,Nf))
  z = np.zeros((Nn,Nf))
  r = np.zeros((Nn,Nf))
  # check whether xi and zeta are part of the solution
  xi_exist = detailed_fwd_solution.columns.str.endswith("tot").any()
  xi = np.zeros((Nn,Nn,Nf)) if xi_exist else None
  for n in range(Nn):
    for f in range(Nf):
      basename = f"n{n}_f{f}"
      x[n,f] = solution.loc[t,f"{basename}_loc"]
      z[n,f] = solution.loc[t,basename]
      r[n,f] = replicas.loc[t,basename]
      for m in range(Nn):
        endname = "_tot" if xi_exist else ""
        if m != n:
          y[n,m,f] = detailed_fwd_solution.loc[t,f"{basename}_n{m}{endname}"]
          if xi_exist:
            xi[n,m,f] = detailed_fwd_solution.loc[t,f"{basename}_n{m}_accepted"]
  return x, y, z, r, xi


def extract_solution(
    data: dict, solution: dict
  ) -> Tuple[
    np.array, np.array, np.array, np.array, np.array, np.array, float
  ]:
  Nn = data[None]["Nn"][None]
  Nf = data[None]["Nf"][None]
  # extract values from solution dictionary
  x = np.zeros((Nn,Nf))
  omega = np.zeros((Nn,Nf))
  r = np.zeros((Nn,Nf))
  y = np.zeros((Nn,Nn,Nf))
  z = np.zeros((Nn,Nf))
  # -- local processing
  if "x" in solution:
    x = np.array(solution["x"], dtype = VAR_TYPE).reshape((Nn, Nf))
  elif "x_bar" in data[None]:
    x = np.array(
      list(data[None]["x_bar"].values()), dtype = VAR_TYPE
    ).reshape((Nn, Nf))
  # -- (detailed) offloading
  if "y" in solution:
    y = np.array(solution["y"], dtype = VAR_TYPE).reshape((Nn, -1, Nf))
  elif "y_bar" in data[None]:
    y = np.array(
      list(data[None]["y_bar"].values()), dtype = VAR_TYPE
    ).reshape((Nn, -1, Nf))
  # -- rejections
  if "z" in solution:
    z = np.array(solution["z"], dtype = VAR_TYPE).reshape((Nn, Nf))
  elif "z_bar" in data[None]:
    z = np.array(
      list(data[None]["z_bar"].values()), dtype = VAR_TYPE
    ).reshape((Nn, Nf))
  # -- number of function replicas
  if "r" in solution:
    r = np.array(
      [float_to_int(rval) for rval in np.array(solution["r"]).flatten()], 
      dtype = int
    ).reshape((Nn, Nf))
    if "r_bar" in data[None]:
      r_bar = np.array(
        [float_to_int(rval) for rval in data[None]["r_bar"].values()], 
        dtype = int
      ).reshape((Nn, Nf))
      r += r_bar
  elif "r_bar" in data[None]:
    r = np.array(
      list(data[None]["r_bar"].values()), dtype = int
    ).reshape((Nn, Nf))
  # -- offloading
  if "omega" in solution:
    omega = np.array(solution["omega"], dtype = VAR_TYPE).reshape((Nn, -1))
  else:
    if "y" in solution:
      for n in range(Nn):
        for f in range(Nf):
          omega[n,f] = y[n,:,f].sum()
    elif "omega_bar" in data[None]:
      omega = np.array(
        list(data[None]["omega_bar"].values()), dtype = VAR_TYPE
      ).reshape((Nn, -1))
  # compute xi
  xi = np.zeros((Nn,Nn,Nf))
  for n1 in range(Nn):
    for n2 in range(Nn):
      for f in range(Nf):
        xi[n2,n1,f] = y[n1,n2,f]
  # compute residual capacity
  rho = compute_residual_capacity(data, r)
  return x, y, z, r, xi, omega, rho, solution["obj"]


def generate_load_traces(
    limits: dict, 
    max_steps: int = 100, 
    seed: int = 4850, 
    trace_type: str = "clipped",
    solution_folder: str = None
  ) -> dict:
  LG = LoadGenerator(average_requests = 100, amplitude_requests = 50)
  rng = np.random.default_rng(seed = seed)
  # generate trace for all request classes
  input_requests_traces = {}
  for function, function_limits in limits.items():
    input_requests_traces[function] = LG.generate_traces(
      max_steps = max_steps, 
      limits = function_limits,
      rng = rng,
      trace_type = trace_type #f"manual{function}"#
    )
    # plot trace (if required)
    if len(limits) <= 10 and len(function_limits) <= 10:
      plot_filename = None
      if solution_folder is not None:
        plot_filename = os.path.join(solution_folder, "load")
        os.makedirs(plot_filename, exist_ok = True)
        plot_filename = os.path.join(plot_filename, f"f{function}.png")
      LG.plot_input_load(
        input_requests_traces[function], plot_filename = plot_filename
      )
  # save traces (if required)
  if solution_folder is not None:
    with open(
      os.path.join(solution_folder, "input_requests_traces.json"), "w"
    ) as istream:
      istream.write(
        json.dumps(input_requests_traces, indent = 2, cls = NpEncoder)
      )
  return input_requests_traces


def get_current_load(
    input_requests_traces: dict, agents: list, t: int
  ) -> dict:
  incoming_load = {
    (a+1, f+1): input_requests_traces[f][a][t] \
      for a in agents for f in input_requests_traces
  }
  return incoming_load


def init_complete_solution():
  return {
    "local_processing": pd.DataFrame(),
    "offloading": pd.DataFrame(),
    "detailed_offloading": pd.DataFrame(),
    "rejections": pd.DataFrame(),
    "replicas": pd.DataFrame(),
    "utilization": pd.DataFrame(),
    "offloaded_processing": pd.DataFrame(),
    "detailed_offloaded_processing": pd.DataFrame(),
    "residual_capacity": pd.DataFrame()
  }


def init_empty_solution(Nn: int, Nf: int) -> Tuple:
  x = np.zeros((Nn,Nf))
  y = np.zeros((Nn,Nn,Nf))
  z = np.zeros((Nn,Nf))
  r = np.zeros((Nn,Nf))
  xi = np.zeros((Nn,Nn,Nf))
  omega = np.zeros((Nn,Nf))
  rho = np.zeros((Nn,))
  obj = None
  U = np.zeros((Nn,Nf))
  return x, y, z, r, xi, omega, rho, obj, U


def init_problem(
    limits: dict, 
    trace_type: str, 
    max_steps: int, 
    seed: int, 
    solution_folder: str
  ) -> Tuple[dict, dict, list]:
  # generate base instance data
  rng = np.random.default_rng(seed = seed)
  base_instance_data, load_limits = generate_data(
    "random", rng = rng, limits = limits
  )
  with open(
    os.path.join(solution_folder, "base_instance_data.json"), "w"
  ) as istream:
    istream.write(
      json.dumps(
        delete_tuples(base_instance_data), indent = 2, cls = NpEncoder
      )
    )
  with open(
    os.path.join(solution_folder, "load_limits.json"), "w"
  ) as istream:
    istream.write(json.dumps(load_limits, indent = 2, cls = NpEncoder))
  # generate input load traces
  input_requests_traces = generate_load_traces(
    load_limits, max_steps, seed, trace_type, solution_folder
  )
  return base_instance_data, input_requests_traces, load_limits[0].keys()


def join_complete_solution(
    complete_solution: dict
  ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  solution = complete_solution["local_processing"].join(
    complete_solution["offloading"], lsuffix = "_loc", rsuffix = "_fwd"
  ).join(complete_solution["rejections"])
  offloaded = complete_solution["offloaded_processing"]
  detailed_fwd_solution = complete_solution["detailed_offloading"].join(
    complete_solution["detailed_offloaded_processing"], 
    lsuffix = "_tot", 
    rsuffix = "_accepted"
  )
  return solution, offloaded, detailed_fwd_solution


def solve_instance(
    M: BaseLoadManagementModel, 
    data: dict, 
    solver_name: str, 
    solver_options: dict
  ) -> Tuple[
    np.array, 
    np.array, 
    np.array, 
    np.array, 
    np.array, 
    np.array, 
    np.array, 
    float, 
    float,
    str
  ]:
  instance = M.generate_instance(data)
  # instance.pprint()
  #
  solution = M.solve(instance, solver_options, solver_name)
  tc = solution["termination_condition"]
  x, y, z, r, xi, omega, rho, obj, U = init_empty_solution(
    data[None]["Nn"][None], data[None]["Nf"][None]
  )
  if solution["solution_exists"]:
    x, y, z, r, xi, omega, rho, obj = extract_solution(data, solution)
    U = compute_utilization(data, solution)
  return x, y, z, r, xi, omega, rho, U, obj, solution["runtime"], tc


def save_checkpoint(complete_solution: dict, solution_folder: str, t: int):
  checkpoint_folder = os.path.join(solution_folder, str(t))
  os.makedirs(checkpoint_folder, exist_ok = True)
  for key, val in complete_solution.items():
    checkpoint_path = os.path.join(checkpoint_folder, f"{key}.csv")
    val.to_csv(checkpoint_path, index = False)


def save_solution(
    solution: pd.DataFrame,
    offloaded: pd.DataFrame,
    complete_solution: dict,
    detailed_fwd_solution: pd.DataFrame,
    model_name: str,
    solution_folder: str
  ):
  solution.to_csv(
    os.path.join(solution_folder, f"{model_name}_solution.csv"), 
    index = False
  )
  offloaded.to_csv(
    os.path.join(solution_folder, f"{model_name}_offloaded.csv"), 
    index = False
  )
  complete_solution["utilization"].to_csv(
    os.path.join(solution_folder, f"{model_name}_utilization.csv"), 
    index = False
  )
  complete_solution["replicas"].to_csv(
    os.path.join(solution_folder, f"{model_name}_replicas.csv"), 
    index = False
  )
  detailed_fwd_solution.to_csv(
    os.path.join(solution_folder, f"{model_name}_detailed_fwd_solution.csv"), 
    index = False
  )
  complete_solution["residual_capacity"].to_csv(
    os.path.join(solution_folder, f"{model_name}_residual_capacity.csv"), 
    index = False
  )


def update_1d_variables(
    var: np.array, res: pd.DataFrame
  ) -> pd.DataFrame:
  Nn = var.shape[0]
  df = {f"n{n}": [var[n]] for n in range(Nn)}
  res = pd.concat(
    [res, pd.DataFrame(df)], ignore_index = True
  )
  return res


def update_2d_variables(
    var: np.array, res: pd.DataFrame
  ) -> pd.DataFrame:
  Nn, Nf = var.shape
  df = var.reshape(1,-1).tolist()
  cols = [f"n{n}_f{f}" for n in range(Nn) for f in range(Nf)]
  res = pd.concat(
    [res, pd.DataFrame(df, columns = cols)], ignore_index = True
  )
  return res


def update_3d_variables(
    y: np.array, offloading: pd.DataFrame, detailed_offloading: pd.DataFrame
  ) -> pd.DataFrame:
  Nn, _, Nf = y.shape
  df = {f"n{n}_f{f}": [] for n in range(Nn) for f in range(Nf)}
  detailed_df = {
    f"n{n1}_f{f}_n{n2}": [] \
      for n1 in range(Nn) for f in range(Nf) for n2 in range(Nn) if n2 != n1
  }
  for f in range(Nf):
    for n1 in range(Nn):
      df[f"n{n1}_f{f}"].append(y[n1,:,f].sum())
      for n2 in range(Nn):
        if n1 != n2:
          detailed_df[f"n{n1}_f{f}_n{n2}"].append(y[n1,n2,f])
  offloading = pd.concat(
    [offloading, pd.DataFrame(df)], ignore_index = True
  )
  detailed_offloading = pd.concat(
    [detailed_offloading, pd.DataFrame(detailed_df)], ignore_index = True
  )
  return offloading, detailed_offloading


def run(
    config: dict, log_on_file: bool = False, disable_plotting: bool = False
  ):
  base_solution_folder = config["base_solution_folder"]
  seed = config["seed"]
  limits = config["limits"]
  trace_type = config["limits"]["load"].get("trace_type", "fixed_sum")
  max_steps = config["max_steps"]
  min_run_time = config.get("min_run_time", 0)
  max_run_time = config.get("max_run_time", max_steps)
  run_time_step = config.get("run_time_step", 1)
  checkpoint_interval = config["checkpoint_interval"]
  plot_interval = config.get("plot_interval", max_steps)
  solver_name = config["solver_name"]
  solver_options = config.get("solver_options", {}).get("general", {})
  verbose = config.get("verbose", 0)
  # generate solution folder
  now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')
  solution_folder = f"{base_solution_folder}/{now}"
  os.makedirs(solution_folder, exist_ok = True)
  with open(os.path.join(solution_folder, "config.json"), "w") as ostream:
    ostream.write(json.dumps(config, indent = 2))
  # initialize log stream (if required)
  log_stream = sys.stdout
  if log_on_file:
    log_stream = open(os.path.join(solution_folder, "out.log"), "w")
  # initialize models
  models = [LoadManagementModel()]
  # generate base instance data and load traces
  base_instance_data, input_requests_traces, agents = init_problem(
    limits, trace_type, max_steps, seed, solution_folder
  )
  Nn = base_instance_data[None]["Nn"][None]
  Nf = base_instance_data[None]["Nf"][None]
  # run models
  obj_dict = {}
  tc_dict = {}
  runtime_dict = {}
  for M in models:
    if verbose > 0:
      print(f"### solving model {M.name}", file = log_stream, flush = True)
    complete_solution = init_complete_solution()
    obj_values = []
    termination_conditions = []
    runtimes = []
    r = None
    ub = (
      max_run_time + run_time_step
    ) if max_run_time == min_run_time else max_run_time
    for t in range(min_run_time, ub, run_time_step):
      if verbose > 0:
        print(f"  t = {t}", file = log_stream, flush = True)
      # get current load
      incoming_load = get_current_load(input_requests_traces, agents, t)
      # update data
      data = update_data(base_instance_data, {"incoming_load": incoming_load})
      # solve
      x, y, z, r, xi, omega, rho, U, obj, runtime, tc = solve_instance(
        M, data, solver_name, solver_options
      )
      complete_solution = decode_solution(
        x, y, z, r, xi, rho, U, complete_solution
      )
      obj_values.append(obj)
      termination_conditions.append(tc)
      runtimes.append(runtime)
      # save checkpoint
      if t % checkpoint_interval == 0 or t == max_steps - 1:
        save_checkpoint(
          complete_solution, os.path.join(solution_folder, M.name), t
        )
      # plot (if needed)
      if 0 < t < ub - 1 and t % plot_interval == 0:
        solution, offloaded, detailed_fwd_solution = join_complete_solution(
          complete_solution
        )
        if not disable_plotting and Nf <= 10 and Nn <= 10:
          plot_folder = os.path.join(solution_folder, M.name, f"{t}_plot")
          os.makedirs(plot_folder)
          plot_history(
            input_requests_traces, 
            min_run_time, 
            t, 
            run_time_step, 
            solution, 
            complete_solution["utilization"], 
            complete_solution["replicas"], 
            offloaded,
            obj_values,
            os.path.join(plot_folder, f"{M.name}.png")
          )
    obj_dict[M.name] = obj_values
    tc_dict[M.name] = termination_conditions
    runtime_dict[M.name] = runtimes
    # join
    solution, offloaded, detailed_fwd_solution = join_complete_solution(
      complete_solution
    )
    # plot and save solution
    if not disable_plotting and Nf <= 10 and Nn <= 10:
      plot_history(
        input_requests_traces, 
        min_run_time, 
        max_run_time, 
        run_time_step, 
        solution, 
        complete_solution["utilization"], 
        complete_solution["replicas"], 
        offloaded,
        obj_values,
        os.path.join(solution_folder, f"{M.name}.png")
      )
    save_solution(
      solution,
      offloaded,
      complete_solution,
      detailed_fwd_solution,
      M.name,
      solution_folder
    )
  # save objective function values and models runtime
  pd.DataFrame(obj_dict).to_csv(
    os.path.join(solution_folder, "obj.csv"), index = False
  )
  pd.DataFrame(tc_dict).to_csv(
    os.path.join(solution_folder, "termination_condition.csv"), index = False
  )
  pd.DataFrame(runtime_dict).to_csv(
    os.path.join(solution_folder, "runtime.csv"), index = False
  )
  if verbose > 0:
    print(
      f"All solutions saved in: {solution_folder}", 
      file = log_stream, 
      flush = True
    )
  # close log stream if needed
  if log_on_file:
    log_stream.close()
  return solution_folder


if __name__ == "__main__":
  args = parse_arguments()
  config_file = args.config
  disable_plotting = args.disable_plotting
  # load configuration file
  config = load_configuration("config.json")
  # run
  run(config, disable_plotting = disable_plotting)
