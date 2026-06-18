import argparse
import json

import numpy as np

from dream_env import make_env
from model import make_model


def simulate(controller, episodes=1, seed=1, max_steps=500):
  rewards = []
  lengths = []
  for episode in range(episodes):
    env = make_env(seed=seed + episode, max_steps=max_steps)
    z = env.reset(seed=seed + episode)
    controller.reset()
    total_reward = 0.0
    for t in range(max_steps):
      controller.rnn_state = env.rnn_state
      action = controller.get_action(z)
      z, reward, done, info = env.step(action)
      controller.rnn_state = env.rnn_state
      total_reward += reward
      if done:
        break
    env.close()
    rewards.append(total_reward)
    lengths.append(t + 1)
  return rewards, lengths


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("model_json", nargs="?", default=None)
  parser.add_argument("--episodes", type=int, default=100)
  parser.add_argument("--max-steps", type=int, default=500)
  parser.add_argument("--seed", type=int, default=1)
  args = parser.parse_args()

  controller = make_model(load_model=True)
  if args.model_json:
    controller.load_model(args.model_json)
  else:
    rng = np.random.default_rng(args.seed)
    controller.set_model_params(rng.standard_normal(controller.param_count) * 0.01)

  rewards, lengths = simulate(controller, episodes=args.episodes, seed=args.seed, max_steps=args.max_steps)
  print("rewards", rewards)
  print("mean_reward", float(np.mean(rewards)), "std", float(np.std(rewards)), "mean_length", float(np.mean(lengths)))


if __name__ == "__main__":
  main()
