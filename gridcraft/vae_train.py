import argparse
import os

import numpy as np

from vae.vae import GridcraftVAE


def load_dataset(record_dir, limit=None):
  files = sorted([f for f in os.listdir(record_dir) if f.endswith(".npz")])
  if limit is not None:
    files = files[:limit]
  batches = []
  for filename in files:
    data = np.load(os.path.join(record_dir, filename))
    batches.append(data["obs"].astype(np.float32))
  if not batches:
    raise RuntimeError(f"no .npz files found in {record_dir}")
  return np.concatenate(batches, axis=0)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--record-dir", default="record")
  parser.add_argument("--out-dir", default="vae")
  parser.add_argument("--steps", type=int, default=10000)
  parser.add_argument("--batch-size", type=int, default=256)
  parser.add_argument("--limit", type=int, default=None)
  parser.add_argument("--seed", type=int, default=1)
  args = parser.parse_args()

  rng = np.random.default_rng(args.seed)
  dataset = load_dataset(args.record_dir, args.limit)
  model = GridcraftVAE()
  os.makedirs(args.out_dir, exist_ok=True)

  for step in range(args.steps):
    indices = rng.integers(0, len(dataset), size=args.batch_size)
    loss, grid_loss, self_loss, kl = model.train_batch(dataset[indices])
    if step == 0 or (step + 1) % 100 == 0:
      print("step", step + 1, "loss", loss, "grid", grid_loss, "self", self_loss, "kl", kl)

  model.save_json(os.path.join(args.out_dir, "vae.json"))
  print("saved", os.path.join(args.out_dir, "vae.json"))


if __name__ == "__main__":
  main()
