import argparse
import os
import time

import numpy as np

from env import ACTION_SIZE, make_env
from experiment_logging import add_wandb_args, logger_from_args
from wandb_schema import GENERAL, WORLD_MODEL_TRAINING


def extract(episodes, max_steps, seed, out_dir, logger=None, log_every=50):
  os.makedirs(out_dir, exist_ok=True)
  rng = np.random.default_rng(seed)
  rewards = []
  lengths = []
  start_time = time.time()
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
    episode_reward = float(np.sum(reward_list))
    episode_length = len(action_list)
    rewards.append(episode_reward)
    lengths.append(episode_length)
    print("saved", path, "steps", episode_length, "reward", episode_reward)
    if logger is not None and (episode == 0 or (episode + 1) % log_every == 0 or episode + 1 == episodes):
      elapsed = max(1e-6, time.time() - start_time)
      logger.log({
        "extraction_episodes": episode + 1,
        "extraction_reward_mean": float(np.mean(rewards[-log_every:])),
        "extraction_reward_global_mean": float(np.mean(rewards)),
        "extraction_episode_length_mean": float(np.mean(lengths[-log_every:])),
        "extraction_steps_total": int(np.sum(lengths)),
        "extraction_episodes_per_second": float((episode + 1) / elapsed),
      }, step=episode + 1, namespace="wm_training")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--episodes", type=int, default=5000)
  parser.add_argument("--max-steps", type=int, default=500)
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--out-dir", default="record")
  parser.add_argument("--log-every", type=int, default=50)
  add_wandb_args(parser)
  args = parser.parse_args()
  logger = logger_from_args(
    args,
    config=vars(args),
    default_group="extract",
    default_name=f"gridcraft-extract-seed{args.seed}",
    tags=["gridcraft", "extract"],
    info_sections=[GENERAL, WORLD_MODEL_TRAINING],
    out_dir=args.out_dir,
  )
  extract(args.episodes, args.max_steps, args.seed, args.out_dir, logger=logger, log_every=args.log_every)
  logger.finish()


if __name__ == "__main__":
  main()
