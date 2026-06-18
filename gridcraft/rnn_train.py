import argparse
import json
import os

import numpy as np

from rnn.rnn import GridcraftRNN


def load_series(path):
  data = np.load(path)
  episodes = []
  lengths = data["length"] if "length" in data.files else np.full((len(data["z"]),), data["z"].shape[1])
  for i, length in enumerate(lengths):
    n = int(length)
    episodes.append((data["z"][i, :n].astype(np.float32),
                     data["action"][i, :n].astype(np.int64),
                     data["reward"][i, :n].astype(np.float32),
                     data["done"][i, :n].astype(np.bool_)))
  return episodes


def sample_batch(episodes, batch_size, seq_len, rng):
  candidates = [ep for ep in episodes if len(ep[0]) > seq_len]
  if not candidates:
    raise RuntimeError("no episodes long enough for requested sequence length")
  z_batch = []
  action_batch = []
  reward_batch = []
  done_batch = []
  for _ in range(batch_size):
    z, action, reward, done = candidates[int(rng.integers(0, len(candidates)))]
    start = int(rng.integers(0, len(z) - seq_len))
    end = start + seq_len + 1
    z_batch.append(z[start:end])
    action_batch.append(action[start:end])
    reward_batch.append(np.pad(reward[start:start + seq_len], (0, 1)))
    done_batch.append(np.pad(done[start:start + seq_len], (0, 1)))
  return (np.asarray(z_batch, dtype=np.float32),
          np.asarray(action_batch, dtype=np.int64),
          np.asarray(reward_batch, dtype=np.float32),
          np.asarray(done_batch, dtype=np.bool_))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--series", default="series/series.npz")
  parser.add_argument("--initial-z", default="series/initial_z.npz")
  parser.add_argument("--out-dir", default="rnn")
  parser.add_argument("--initial-z-out", default="initial_z/initial_z.json")
  parser.add_argument("--steps", type=int, default=10000)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--seq-len", type=int, default=32)
  parser.add_argument("--seed", type=int, default=1)
  args = parser.parse_args()

  rng = np.random.default_rng(args.seed)
  episodes = load_series(args.series)
  model = GridcraftRNN()
  os.makedirs(args.out_dir, exist_ok=True)

  initial_z = np.load(args.initial_z)["z"].astype(np.float32)
  os.makedirs(os.path.dirname(args.initial_z_out), exist_ok=True)
  with open(args.initial_z_out, "w") as f:
    json.dump(initial_z.tolist(), f)

  for step in range(args.steps):
    batch = sample_batch(episodes, args.batch_size, args.seq_len, rng)
    loss, z_loss, reward_loss, done_loss = model.train_batch(*batch)
    if step == 0 or (step + 1) % 100 == 0:
      print("step", step + 1, "loss", loss, "z", z_loss, "reward", reward_loss, "done", done_loss)

  model.save_json(os.path.join(args.out_dir, "rnn.json"))
  print("saved", os.path.join(args.out_dir, "rnn.json"))


if __name__ == "__main__":
  main()
