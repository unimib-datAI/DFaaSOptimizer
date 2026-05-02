from rlagents.ma_environment import FaaSMARLEnvironment, FaaSMARLEnvironment2
from rlagents.sa_environment import FaaSRLEnvironment
from ray.tune.registry import register_env
register_env("FaaSMARLEnvironment", lambda config: FaaSMARLEnvironment(config))
register_env("FaaSMARLEnvironment2", lambda config: FaaSMARLEnvironment2(config))
register_env("FaaSRLEnvironment", lambda config: FaaSRLEnvironment(config))

from RL4CC.experiments.train import TrainingExperiment


exp = TrainingExperiment(exp_config_file = "config_files/exp_config_ppo.json")
exp.run()
