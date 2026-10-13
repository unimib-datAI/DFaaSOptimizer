"""Train an RL agent using an explicitly supplied RL4CC experiment config."""
import argparse


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('-c', '--config', required=True, help='RL4CC experiment JSON file')
  args = parser.parse_args()

  from rlagents.ma_environment import FaaSMARLEnvironment, FaaSMARLEnvironment2
  from rlagents.sa_environment import FaaSRLEnvironment
  from ray.tune.registry import register_env
  from RL4CC.experiments.train import TrainingExperiment

  register_env("FaaSMARLEnvironment", lambda config: FaaSMARLEnvironment(config))
  register_env("FaaSMARLEnvironment2", lambda config: FaaSMARLEnvironment2(config))
  register_env("FaaSRLEnvironment", lambda config: FaaSRLEnvironment(config))
  TrainingExperiment(exp_config_file=args.config).run()


if __name__ == '__main__':
  main()
