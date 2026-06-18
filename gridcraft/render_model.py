import argparse

import numpy as np

from model import GridcraftController
from render_env import make_env


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("model_json", nargs="?", default=None)
  parser.add_argument("--episodes", type=int, default=1)
  parser.add_argument("--max-steps", type=int, default=500)
  parser.add_argument("--seed", type=int, default=1)
  args = parser.parse_args()

  controller = GridcraftController(load_world_model=True)
  if args.model_json:
    controller.load_model(args.model_json)
  else:
    rng = np.random.default_rng(args.seed)
    controller.set_model_params(rng.standard_normal(controller.param_count) * 0.01)

  rewards = []
  for episode in range(args.episodes):
    env = make_env(seed=args.seed + episode, max_steps=args.max_steps)
    obs = env.reset(seed=args.seed + episode)
    controller.reset()
    total_reward = 0.0
    for t in range(args.max_steps):
      z = controller.encode_obs(obs)
      action = controller.get_action(z)
      obs, reward, done, info = env.step(action)
      controller.observe_transition(z, action)
      total_reward += reward
      env.render()
      if done:
        break
    env.close()
    rewards.append(total_reward)
  print("rewards", rewards)
  print("mean_reward", float(np.mean(rewards)))


if __name__ == "__main__":
  main()
