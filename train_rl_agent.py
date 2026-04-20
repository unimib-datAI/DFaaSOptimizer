from rlagents.environment import FaaSMARLEnvironment
from ray.tune.registry import register_env
register_env("FaaSMARLEnvironment", lambda config: FaaSMARLEnvironment(config))

from RL4CC.experiments.train import TrainingExperiment


exp = TrainingExperiment(exp_config_file = "config_files/exp_config_ppo.json")
exp.run()
