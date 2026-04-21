from RL4CC.environment import BaseEnvironment
from RL4CC.callbacks import BaseCallbacks

from generators.generate_load import generate_load_traces
from utils.common import (
  load_base_instance, load_configuration, load_requests_traces
)
from utils.faasmacro import compute_centralized_objective
from utils.centralized import get_current_load

from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.env.env_context import EnvContext
from gymnasium.spaces import Box, Dict
from copy import deepcopy
from typing import Tuple
import numpy as np
import os


def _convert_arrival_rate_dist(arrival_rate, action_dist):
  """
  Distribute the arrival rate to the given action distribution.
  Returns a tuple with the arrival rate for each action:
  - First element: local processing rate
  - Middle elements: forwarding rates to specific neighbors
  - Last element: rejection rate
  All values are integers.
  This function expects an action distribution that sums to 1."""
  # Calculate the raw rates based on the action distribution.
  raw_rates = [int(prob * arrival_rate) for prob in action_dist]
  # Calculate how many requests are currently assigned.
  total_assigned = sum(raw_rates)
  # Distribute the remaining requests due to integer rounding.
  remaining = int(arrival_rate - total_assigned)
  if remaining > 0:
    # Find the actions with the highest fractional parts.
    fractions = [
      (
        prob * arrival_rate
      ) - raw_rates[i] for i, prob in enumerate(action_dist)
    ]
    # Sort indices by fraction.
    sorted_indices = sorted(
      range(len(fractions)), key=lambda i: fractions[i], reverse=True
    )
    # Assign remaining requests to actions with highest fractional parts.
    for i in range(remaining):
      raw_rates[sorted_indices[i % len(sorted_indices)]] += 1
  assert sum(raw_rates) == arrival_rate
  return tuple(raw_rates)


class FaaSRLEnvironment(BaseEnvironment):
  
  def load_configuration(self, env_config: EnvContext) -> int:
    # simulation time management
    self.min_time = env_config["min_time"]
    self.max_time = env_config["max_time"]
    self.time_step = env_config["time_step"]
    self.current_time = self.min_time
    # load base instance data
    self.instance_data, self.load_limits = load_base_instance(
      env_config["opt_folder"]
    )
    self.instance_data = self.instance_data[None]
    # -- nodes and functions
    self.nodes = list(range(1, self.instance_data["Nn"][None] + 1))
    self.functions = list(range(1, self.instance_data["Nf"][None] + 1))
    self.Nn = len(self.nodes)
    self.Nf = len(self.functions)
    # -- build neighborhood
    self.agent_neighbors = {
      i: [
        j for j in self.nodes if self.instance_data["neighborhood"][(i,j)]
      ] for i in self.nodes
    }
    # -- compute maximum number of replicas
    self.max_n_replicas = np.zeros((self.Nn,self.Nf))
    for n in self.nodes:
      for f in self.functions:
        self.max_n_replicas[n-1][f-1] = (
          self.instance_data[
            "memory_capacity"
          ][n] // self.instance_data[
            "memory_requirement"
          ][f]
        )
    # load requests traces (in evaluation)
    self.is_evaluation = env_config.get("is_evaluation", False)
    if self.is_evaluation:
      self.workload_trace, mt, Mt, ts = load_requests_traces(
        env_config["opt_folder"]
      )
      if self.min_time < mt or self.max_time > Mt or self.time_step != ts:
        raise RuntimeError("Inconsistent time management")
    # initialize random number generator
    self.opt_config = load_configuration(
      os.path.join(env_config["opt_folder"], "config.json")
    )
    self.rng = np.random.default_rng(seed = self.opt_config["seed"])
    return int(self.rng.integers(low = 0, high = 4850 * 4850))
  
  def define_observation_space(self):
    """
    Define the environment observation space
    """
    self.absolute_max_workload = 10000
    self.observation_space = Dict({
      "input_rate": Box(
        low = 0,
        high = self.absolute_max_workload,
        shape = (self.Nn,self.Nf),
        dtype = np.int32
      ),
      "previous_input_rate": Box(
        low = 0,
        high = self.absolute_max_workload,
        shape = (self.Nn,self.Nf),
        dtype = np.int32
      ),
      # previous local/forwarding/cloud offloading utility
      "previous_loc_utility": Box(
        low = 0.0,
        high = 1.0,
        shape = (1,),
        dtype = np.float32
      ),
      "previous_fwd_utility": Box(
        low = 0.0,
        high = 1.0,
        shape = (1,),
        dtype = np.float32
      ),
      "previous_cloud_penalty": Box(
        low = -1.0,
        high = 0.0,
        shape = (1,),
        dtype = np.float32
      ),
      # previous CPU utilization
      "previous_cpu_utilization": Box(
        low = 0.0,
        high = np.inf,
        shape = (self.Nn,self.Nf),
        dtype = np.float32
      ),
      # previous number of replicas
      "previous_r": Box(
        low = 0,
        high = self.max_n_replicas.max(),
        shape = (self.Nn,self.Nf),
        dtype = np.int32
      ),
      # demand
      "demand": Box(
        low = 0.0,
        high = max(self.instance_data["demand"].values()),
        shape = (self.Nn,self.Nf),
        dtype = np.float32
      )
    })
    # initialize a random dummy observation to return at the end
    self._dummy_terminal_obs = self.observation_space.sample()
  
  def define_action_space(self):
    """
    Define the environment action space
    ---
    For each node,
      * xyz:  proportion of requests enqueued, fowarded (to each neighbor) or 
              offloaded to the cloud
      * r:    number of replicas
    """
    self.action_space = Dict({
      n: Dict({
        # -- load distribution
        **{
          f"xyz_{f}": Simplex(
            shape = (1 + len(self.agent_neighbors[n]) + 1,)
          ) for f in self.functions
        },
        # -- replicas
        "r": Simplex(
          shape = (self.Nf,)
        )
      }) for n in self.nodes
    })
  
  def init_agent_metrics(self):
    self.info = {
      # -- input rate, demand and CPU utilization
      "input_rate": np.zeros((self.Nn,self.Nf), dtype = np.int32),
      "demand": np.zeros((self.Nn,self.Nf), dtype = np.float32),
      "cpu_utilization": np.zeros((self.Nn,self.Nf), dtype = np.float32),
      # -- action
      "action": {},
      "loc": np.zeros((self.Nn,self.Nf), np.int32),           # x
      "fwd": np.zeros((self.Nn,self.Nn,self.Nf), np.int32),   # y
      "total_fwd": np.zeros((self.Nn,self.Nf), np.int32),     # omega
      "rej": np.zeros((self.Nn,self.Nf), dtype = np.int32),   # z
      "n_replicas": np.zeros((self.Nn,self.Nf)),              # r
      # -- utility
      "loc_utility": 0.0,
      "fwd_utility": 0.0,
      "cloud_penalty": 0.0,
      "cobj": 0.0,
      # -- feasibility
      "feasible": True,
      "why_feasible": ""
    }
  
  def observation(self):
    """
    Return the next observation (and the corresponding info dictionary)
    """
    obs = {
      "input_rate": np.zeros((self.Nn, self.Nf), dtype = np.int32),
      "previous_input_rate": deepcopy(self.info["input_rate"]),
      "previous_loc_utility": np.array(
        [self.info["loc_utility"]], dtype = np.float32
      ),
      "previous_fwd_utility": np.array(
        [self.info["fwd_utility"]], dtype = np.float32
      ),
      "previous_cloud_penalty": np.array(
        [self.info["cloud_penalty"]], dtype = np.float32
      ),
      "previous_r": deepcopy(self.info["n_replicas"]),
      "previous_cpu_utilization": deepcopy(self.info["cpu_utilization"]),
      "demand": np.zeros((self.Nn, self.Nf), dtype = np.float32)
    }
    # loop over agents
    for n in self.nodes:
      for f in self.functions:
        # -- workload
        obs["input_rate"][n-1,f-1] = int(
          self.workload_trace[f-1][n-1][self.current_time]
        )
        # -- demand
        obs["demand"][n-1,f-1] = float(
          self.instance_data["demand"][(n,f)]
        )
    obs_info = {
      "input_rate": obs["input_rate"],
      "demand": obs["demand"]
    }
    # update info
    old_info = deepcopy(self.info)
    for key, val in old_info.items():
      if key in obs_info and key != "current_time":
        self.info[f"previous_{key}"] = val
        self.info[key] = obs_info[key]
    self.info["current_time"] = self.current_time
    return obs, self.info
  
  def reset(self, seed: int = None, options = None):
    # initialize info dictionary
    self.init_agent_metrics()
    # set seed from the parent class
    if seed is None:
      seed = self.rng.integers(low = 0, high = 4850 * 4850)
    # restart time
    self.current_time = self.min_time
    # generate workload trace (if not in evaluation)
    if not self.is_evaluation:
      self.workload_trace = generate_load_traces(
        self.load_limits, 
        self.max_time, 
        seed, 
        self.opt_config["limits"]["load"].get("trace_type", "fixed_sum"), 
        solution_folder = None,
        enable_plotting = False
      )
    # define observation
    obs, obs_info = self.observation()
    return obs, obs_info
  
  def step(self, action_dict):
    """
    Applies the action chosen by each agent, moves to the next state and 
    computes the reward
    ---
    Returns a tuple containing:
      1) new observations for each ready agent, 
      2) reward values for each ready agent. If the episode is just started, 
      the value will be None. 
      3) Terminated values for each ready agent. The special key “__all__” 
      (required) is used to indicate env termination. 
      4) Truncated values for each ready agent. 
      5) Info values for each agent id (may be empty dicts)
    """
    # apply action
    self.simulate_action(action_dict)
    # compute reward
    reward = self.compute_reward()
    # update time
    self.current_time += self.time_step
    # check if we are in the last step of the episod should be truncated
    done = self.current_time >= self.max_time
    truncated = done
    # define observation
    obs = None
    obs_info = {}
    if self.current_time < self.max_time:
      obs, obs_info = self.observation()
    else:
      # -- ignore the last step
      obs = self._dummy_terminal_obs
    return obs, reward, done, truncated, obs_info
  
  def check_feasibility(self) -> bool:
    for n in self.nodes:
      for f in self.functions:
        # no traffic loss
        managed_load = (
          self.info["loc"][n-1,f-1] + 
            self.info["total_fwd"][n-1,f-1] + 
              self.info["rej"][n-1,f-1]
        )
        load = self.workload_trace[f-1][n-1][self.current_time]
        if abs(managed_load - load) > 1e-3:
          return False, f"no traffic loss ({n},{f}): {managed_load} != {load}"
        # max utilization
        utilization = self.info["cpu_utilization"][n-1,f-1]
        max_utilization = self.instance_data["max_utilization"][f]
        if utilization - max_utilization > 1e-5:
          return False, f"max utilization ({n},{f}): {utilization}"
    # memory capacity
    for n, ram in self.instance_data["memory_capacity"].items():
      used_memory = 0
      for f, req_memory in self.instance_data["memory_requirement"].items():
        used_memory += self.info["n_replicas"][n-1,f-1] * req_memory
        if used_memory - ram > 1e-5:
          return False, f"memory capacity ({n},{f}): {used_memory} > {ram}"
    return True, ""
  
  def compute_reward(self):
    # check feasibility
    feasible, why_feasible = self.check_feasibility()
    self.info["feasible"] = feasible
    self.info["why_feasible"] = why_feasible
    # compute centralized objective
    sp_data = {None: deepcopy(self.instance_data)}
    sp_data[None]["incoming_load"] = {
      (n,f): self.info["input_rate"][n-1,f-1] \
        for n in self.nodes for f in self.functions
    }
    cobj = compute_centralized_objective(
      sp_data, self.info["loc"], self.info["fwd"], self.info["rej"]
    )
    self.info["cobj"] = cobj
    # first attempt: reward only if feasible
    reward = float(cobj) if feasible else 0.0
    self.info["loc_utility"] = 0.0
    self.info["fwd_utility"] = 0.0
    self.info["cloud_penalty"] = 0.0
    return reward
  
  def simulate_action(self, action_dict):
    self.info["action"] = action_dict
    # tot_incoming_rate: total incoming requests for each agent (enqueued 
    # requests + those sent by neighbors)
    x = np.zeros((self.Nn,self.Nf), dtype = np.int32)
    omega = np.zeros((self.Nn,self.Nf), dtype = np.int32)
    y = np.zeros((self.Nn,self.Nn,self.Nf), dtype = np.int32)
    z = np.zeros((self.Nn,self.Nf), dtype = np.int32)
    r = np.zeros((self.Nn,self.Nf), dtype = np.int32)
    for n in self.nodes:
      for f in self.functions:
        # -- convert the action proportions into actual number of requests
        action = _convert_arrival_rate_dist(
          self.info["input_rate"][n-1,f-1], action_dict[n][f"xyz_{f}"]
        )
        x[n-1,f-1] = action[0]
        for jidx, j in enumerate(self.agent_neighbors[n]):
          y[n-1,j-1,f-1] = action[jidx+1]
        omega[n-1,f-1] = y[n-1,:,f-1].sum()
        z[n-1,f-1] = action[-1]
        if n not in self.info["action"]:
          self.info["action"][n] = {}
        # -- get the number of replicas
        r[n-1,f-1] = (
          action_dict[n]["r"] * self.instance_data["memory_capacity"][n]
        ) // self.instance_data["memory_requirement"][f]
    self.info["loc"] = x
    self.info["fwd"] = y
    self.info["total_fwd"] = omega
    self.info["rej"] = z
    self.info["n_replicas"] = r
    # simulate
    utilization = np.zeros((self.Nn,self.Nf), dtype = np.float32)
    for n in self.nodes:
      for f in self.functions:
        incoming_rate_total = x[n-1,f-1] + y[:, n-1, f-1].sum()
        utilization[n-1,f-1] = (
          self.instance_data["demand"][(n,f)] * incoming_rate_total
        ) / r[n-1,f-1]
    self.info["cpu_utilization"] = utilization


class FaaSRLCallbacks(BaseCallbacks):
  """
  User defined callbacks for the DFaaS environment.

  These callbacks can be used with other environments, both multi-agent and
  single-agent.

  See the Ray's API documentation for DefaultCallbacks, the custom class
  overrides (and uses) only a subset of callbacks and keyword arguments.
  """
  def on_episode_start(
      self, *, worker, base_env, policies, episode, env_index, **kwargs
    ):
    """
    Callback run right after an episode has started.
    Only the episode and base_env keyword arguments are used, other
    arguments are ignored.
    """
    try:
      env = base_env.envs[0]
    except AttributeError:
      # With single-agent environment the wrapper env is an instance of
      # VectorEnvWrapper and it doesn't have envs attribute. With
      # multi-agent the wrapper is MultiAgentEnvWrapper.
      env = base_env.get_sub_environments()[0]
    self.RELEVANT_KEYS = set()
    for key in env.info:
      self.RELEVANT_KEYS.add(key)
    super().on_episode_start(
      worker = worker, 
      base_env = base_env, 
      policies = policies, 
      episode = episode, 
      env_index = env_index, 
      **kwargs
    )
