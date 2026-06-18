import argparse
import os

import numpy as np

from env import ACTION_SIZE, make_env


def extract(episodes, max_steps, seed, out_dir):
  os.makedirs(out_dir, exist_ok=True)
  rng = np.random.default_rng(seed)
  for episode in range(episodes):
    episode_seed = int(rng.integers(0, 2**31 - 1))
    env = make_env(seed=episode_seed, render_mode=False, max_steps=max_steps)
    obs = env.reset(seed=episode_seed)
    obs_list = []
    action_list = []
    reward_list = []
    done_list = []
    for _ in range(max_steps):
      action = int(rng.integers(0, ACTION_SIZE))
      next_obs, reward, done, info = env.step(action)
      obs_list.append(obs)
      action_list.append(action)
      reward_list.append(reward)
      done_list.append(done)
      obs = next_obs
      if done:
        break
    env.close()
    path = os.path.join(out_dir, f"{episode_seed}.npz")
    np.savez_compressed(
      path,
      obs=np.asarray(obs_list, dtype=np.float32),
      action=np.asarray(action_list, dtype=np.int16),
      reward=np.asarray(reward_list, dtype=np.float32),
      done=np.asarray(done_list, dtype=np.bool_),
    )
    print("saved", path, "steps", len(action_list), "reward", float(np.sum(reward_list)))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--episodes", type=int, default=5000)
  parser.add_argument("--max-steps", type=int, default=500)
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--out-dir", default="record")
  args = parser.parse_args()
  extract(args.episodes, args.max_steps, args.seed, args.out_dir)


if __name__ == "__main__":
  main()
