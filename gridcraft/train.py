import argparse
import json
import os
import time

import numpy as np

from es import CMAES, OpenES, PEPG, SimpleGA
from model import make_model, simulate


def make_optimizer(name, num_params, popsize, sigma):
  if name == "cma":
    return CMAES(num_params, sigma_init=sigma, popsize=popsize)
  if name == "ga":
    return SimpleGA(num_params, sigma_init=sigma, popsize=popsize)
  if name == "openes":
    return OpenES(num_params, sigma_init=sigma, popsize=popsize)
  return PEPG(num_params, sigma_init=sigma, popsize=popsize)


def save_json(path, payload):
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w") as f:
    json.dump(payload, f, indent=2)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-n", "--num_episode", type=int, default=16)
  parser.add_argument("-t", "--num_worker", type=int, default=64)
  parser.add_argument("-o", "--optimizer", choices=["cma", "pepg", "openes", "ga"], default="cma")
  parser.add_argument("--generations", type=int, default=100)
  parser.add_argument("--sigma", type=float, default=0.1)
  parser.add_argument("--seed_start", type=int, default=1)
  parser.add_argument("--max_len", type=int, default=-1)
  parser.add_argument("--max-steps", type=int, default=500)
  args = parser.parse_args()

  np.random.seed(args.seed_start)
  model = make_model(load_model=True)
  num_params = model.param_count
  popsize = args.num_worker
  optimizer = make_optimizer(args.optimizer, num_params, popsize, args.sigma)
  filebase = f"log/gridcraftrnn.{args.optimizer}.{args.num_episode}.{popsize}"
  max_steps = args.max_len if args.max_len > 0 else args.max_steps

  history = []
  history_best = []
  best_reward = -1e9
  best_params = None

  print("size of model", num_params)
  print("filebase", filebase)

  for generation in range(args.generations):
    start = time.time()
    solutions = optimizer.ask()
    rewards = []
    lengths = []
    for idx, params in enumerate(solutions):
      model.set_model_params(params)
      reward_list, t_list = simulate(
        model,
        train_mode=True,
        render_mode=False,
        num_episode=args.num_episode,
        seed=args.seed_start + generation * 1000 + idx * 10,
        max_len=max_steps,
      )
      reward = float(np.mean(reward_list))
      rewards.append(reward)
      lengths.append(float(np.mean(t_list)))
      if reward > best_reward:
        best_reward = reward
        best_params = np.asarray(params)
    optimizer.tell(rewards)

    avg_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    mean_time_step = float(np.mean(lengths))
    elapsed = time.time() - start
    row = [generation, elapsed, avg_reward, float(np.min(rewards)), float(np.max(rewards)), std_reward, float(optimizer.rms_stdev()), mean_time_step]
    history.append(row)
    history_best.append([generation, best_reward])

    current = optimizer.current_param()
    save_json(filebase + ".json", [np.asarray(current).round(6).tolist()])
    if best_params is not None:
      save_json(filebase + ".best.json", [best_params.round(6).tolist()])
    save_json(filebase + ".hist.json", history)
    save_json(filebase + ".hist_best.json", history_best)
    print("generation", generation, "avg", avg_reward, "best", best_reward, "steps", mean_time_step)

  print("saved", filebase + ".best.json")


if __name__ == "__main__":
  main()
