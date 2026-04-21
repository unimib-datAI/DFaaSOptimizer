from RL4CC.environment import BaseMultiAgentEnvironment
from RL4CC.callbacks import BaseCallbacks

from generators.generate_load import generate_load_traces
from utils.common import (
  load_base_instance, load_configuration, load_requests_traces
)
from sa_environment import _convert_arrival_rate_dist

from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.env.env_context import EnvContext
from gymnasium.spaces import Box, Dict, Discrete
from copy import deepcopy
from typing import Tuple
import numpy as np
import os


def _get_n_f(agent: str) -> Tuple[int,int]:
  n, f = agent.split("_")
  return int(n), int(f)


class FaaSMARLEnvironment(BaseMultiAgentEnvironment):
  
  def load_configuration(self, env_config: EnvContext) -> int:
    # simulation time management
    self.min_time = env_config["min_time"]
    self.max_time = env_config["max_time"]
    self.time_step = env_config["time_step"]
    self.current_time = self.min_time
    # initialize agents
    # -- each agent's name is {i}_{f} where i\in{1,Nn+1}, f\in{1,Nf+1}
    self.agents = env_config["agents"]
    # load base instance data
    self.instance_data, self.load_limits = load_base_instance(
      env_config["opt_folder"]
    )
    self.instance_data = self.instance_data[None]
    # -- nodes and functions
    self.nodes = list(range(1, self.instance_data["Nn"][None] + 1))
    self.functions = list(range(1, self.instance_data["Nf"][None] + 1))
    # -- build neighborhood
    self.agent_neighbors = {
      i: [
        j for j in self.nodes if self.instance_data["neighborhood"][(i,j)]
      ] for i in self.nodes
    }
    # -- compute maximum number of replicas
    self.max_n_replicas = {}
    for agent in self.agents:
      n,f = _get_n_f(agent)
      self.max_n_replicas[agent] = (
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
  
  def define_observation_spaces(self):
    """
    Define the environment observation space(s)
    """
    self.absolute_max_workload = 10000
    self.observation_space = Dict({
      agent: Dict({
        "input_rate": Box(
          low = 0,
          high = self.absolute_max_workload,
          shape = (1,),
          dtype = np.int32
        ),
        "previous_input_rate": Box(
          low = 0,
          high = self.absolute_max_workload,
          shape = (1,),
          dtype = np.int32
        ),
        # rate of requests forwarded to all neighbors in the previous step
        **{
          f"previous_y_{j}": Box(
            low = 0,
            high = self.absolute_max_workload,
            shape=(1,),
            dtype = np.int32
          ) for j in self.agent_neighbors[_get_n_f(agent)[0]]
        },
        # previous local/forwarding/cloud offloading utility
        "previous_loc_utility": Box(
          low = 0.0,
          high = 1.0,
          shape=(1,),
          dtype = np.float32
        ),
        "previous_fwd_utility": Box(
          low = 0.0,
          high = 1.0,
          shape=(1,),
          dtype = np.float32
        ),
        "previous_cloud_utility": Box(
          low = 0.0,
          high = 1.0,
          shape=(1,),
          dtype = np.float32
        ),
        # previous CPU utilization
        "previous_cpu_utilization": Box(
          low = 0.0,
          high = np.inf,
          shape=(1,),
          dtype = np.float32
        ),
        # previous number of replicas
        "previous_r": Box(
          low = 0,
          high = self.max_n_replicas[agent],
          shape=(1,),
          dtype = np.int32
        )
      }) for agent in self.agents
    })
    # initialize a random dummy observation to return at the end
    self._dummy_terminal_obs = self.observation_space.sample()
  
  def define_action_spaces(self):
    """
    Define the environment action space(s)
    ---
    For each agent,
      * xyz:  proportion of requests enqueued, fowarded (to each neighbor) or 
              offloaded to the cloud
      * r:    number of replicas
    """
    self.action_space = Dict({
      agent: Dict({
        "xyz": Simplex(
          shape = (1 + len(self.agent_neighbors[_get_n_f(agent)[0]]) + 1,)
        ),
        "r": Discrete(self.max_n_replicas[agent])
      }) for agent in self.agents
    })
  
  def init_agent_metrics(self):
    self.info = {}
    # loop over agents
    for agent in self.agents:
      n,f = _get_n_f(agent)
      self.info[agent] = {
        # -- input rate and CPU utilization
        "input_rate": 0,
        "cpu_utilization": 0.0,
        # -- action
        "action": [],
        "loc": 0,
        "fwd": [],
        "total_fwd": 0,
        "rej": 0,
        "n_replicas": 0,
        # -- utility
        "loc_utility": 0.0,
        "fwd_utility": 0.0,
        "cloud_utility": 0.0
      }
      # -- forward
      for j in self.agent_neighbors[n]:
        neighbor = f"{j}_{f}"
        self.info[agent][f"fwd_to_{neighbor}"] = 0
  
  def observation(self):
    """
    Return the next observation (and the corresponding info dictionary)
    """
    obs = {agent: {} for agent in self.agents}
    obs_info = {agent: {} for agent in self.agents}
    # loop over agents
    for agent in self.agents:
      n,f = _get_n_f(agent)
      # -- workload
      obs[agent]["input_rate"] = np.array(
        [self.workload_trace[f-1][n-1][self.current_time]], dtype = np.int32
      )
      obs_info[agent]["input_rate"] = int(
        self.workload_trace[f-1][n-1][self.current_time]
      )
      obs[agent]["previous_input_rate"] = np.array(
        [self.info[agent]["input_rate"]], dtype = np.int32
      )
      # -- utilities
      obs[agent]["previous_loc_utility"] = np.array(
        [self.info[agent]["loc_utility"]], dtype = np.float32
      )
      obs[agent]["previous_fwd_utility"] = np.array(
        [self.info[agent]["fwd_utility"]], dtype = np.float32
      )
      obs[agent]["previous_cloud_utility"] = np.array(
        [self.info[agent]["cloud_utility"]], dtype = np.float32
      )
      # CPU utilization
      obs[agent]["previous_cpu_utilization"] = np.array(
        [self.info[agent]["cpu_utilization"]], dtype = np.float32
      )
      # number of replicas
      obs[agent]["previous_r"] = np.array(
        [self.info[agent]["n_replicas"]], dtype = np.int32
      )
      # -- forward to neighbors
      for j in self.agent_neighbors[n]:
        neighbor = f"{j}_{f}"
        obs[agent][f"previous_y_{j}"] = np.array(
          [self.info[agent][f"fwd_to_{neighbor}"]], dtype = np.int32
        )
    obs_info["__common__"] = {
      "current_time": self.current_time
    }
    # update info
    old_info = deepcopy(self.info)
    for agent, agent_info in old_info.items():
      for key, val in agent_info.items():
        if key in obs_info[agent]:
          self.info[agent][f"previous_{key}"] = val
          self.info[agent][key] = obs_info[agent][key]
    self.info["__common__"] = obs_info["__common__"]
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
    done = {
      agent: self.current_time >= self.max_time for agent in self.agents
    }
    done["__all__"] = all(done.values())
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
  
  def compute_reward(self):
    reward = {}
    # loop over agents
    for agent in self.agents:
      n,f = _get_n_f(agent)
      loc_utility, fwd_utility, cloud_utility = 0.0, 0.0, 0.0
      # check if the solution is feasible
      if self.info[agent]["cpu_utilization"] <= self.instance_data[
          "max_utilization"
        ][f]:
        # -- local processing
        loc_utility = (
          self.instance_data["alpha"][(n,f)] * self.info[agent]["loc"]
        ) / self.info[agent]["input_rate"]
        self.info[agent]["loc_utility"] = float(loc_utility)
        # -- forwarding to neighbors
        for j in self.agent_neighbors[n]:
          neighbor = f"{j}_{f}"
          if self.info[neighbor]["cpu_utilization"] <= self.instance_data[
              "max_utilization"
            ][f]:
            fwd_utility += (
              self.instance_data["beta"][(n,j,f)] * self.info[agent][
                f"fwd_to_{neighbor}"
              ]
            ) / self.info[agent]["input_rate"]
        self.info[agent]["fwd_utility"] = float(fwd_utility)
        # -- offloading to cloud
        cloud_utility = (
          self.instance_data["gamma"][(n,f)] * self.info[agent]["rej"]
        ) / self.info[agent]["input_rate"]
        self.info[agent]["cloud_utility"] = float(cloud_utility)
      # reward
      reward[agent] = loc_utility + fwd_utility + cloud_utility
    return reward
  
  def simulate_action(self, action_dict):
    # tot_incoming_rate: total incoming requests for each agent (enqueued 
    # requests + those sent by neighbors)
    tot_incoming_requests = {agent: [] for agent in action_dict}
    senders = {agent: [] for agent in action_dict}
    for agent, agent_action_dict in action_dict.items():
      n,f = _get_n_f(agent)
      # -- convert the action proportions into actual number of requests
      action = _convert_arrival_rate_dist(
        self.info[agent]["input_rate"], agent_action_dict["xyz"]
      )
      self.info[agent]["action"] = agent_action_dict["xyz"]
      self.info[agent]["loc"] = action[0]
      self.info[agent]["fwd"] = action[1:-1]
      self.info[agent]["total_fwd"] = sum(action[1:-1])
      self.info[agent]["rej"] = action[-1]
      # -- get the number of replicas
      self.info[agent]["n_replicas"] = int(agent_action_dict["r"] + 1)
      # -- local processing
      tot_incoming_requests[agent].append(action[0])
      # -- forward
      for neigh_idx, j in enumerate(self.agent_neighbors[n]):
        neighbor = f"{j}_{f}"
        tot_incoming_requests[neighbor].append(action[1 + neigh_idx])
        senders[neighbor].append(agent)
        self.info[agent][f"fwd_to_{neighbor}"] = action[1 + neigh_idx]
    # simulate
    for agent in tot_incoming_requests:
      n,f = _get_n_f(agent)
      incoming_rate_total = sum(tot_incoming_requests[agent])
      utilization = (
        self.instance_data["demand"][(n,f)] * incoming_rate_total
      ) / self.info[agent]["n_replicas"]
      self.info[agent]["cpu_utilization"] = float(utilization)


class FaaSMARLCallbacks(BaseCallbacks):
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
    for agent in env.agents + ["__common__"]:
      for key in env.info[agent]:
        self.RELEVANT_KEYS.add(key)
    super().on_episode_start(
      worker = worker, 
      base_env = base_env, 
      policies = policies, 
      episode = episode, 
      env_index = env_index, 
      **kwargs
    )
  
  def on_episode_end(
      self, *, worker, base_env, policies, episode, env_index, **kwargs,
    ):
    try:
      env = base_env.envs[0]
      for agent in env.agents:
        for key in self.RELEVANT_KEYS:
          if f"{key}_{agent}" in episode.user_data:
            _ = episode.hist_data.pop(f"{key}_{agent}")
            if len(episode.user_data[f"{key}_{agent}"]) > 0:
              episode.hist_data[f"{key}-{agent}"] = episode.user_data[
                f"{key}_{agent}"
              ][:-1]
              try:
                episode.custom_metrics[
                  f"{key}-{agent}_avg"
                ] = np.mean(episode.user_data[f"{key}_{agent}"][:-1])
              except Exception:
                episode.custom_metrics[
                  f"{key}-{agent}_avg"
                ] = None
    except AttributeError:
      for key in self.RELEVANT_KEYS:
        episode.hist_data[key] = episode.user_data[key]
        episode.custom_metrics[f"{key}_avg"] = np.mean(episode.user_data[key])
    # add worker index
    episode.hist_data["worker_index"] = episode.user_data["worker_index"]
