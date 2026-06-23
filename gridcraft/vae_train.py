import argparse
import os

import numpy as np

from experiment_logging import add_wandb_args, logger_from_args
from vae.vae import GridcraftVAE
from wandb_schema import GENERAL, WORLD_MODEL_TRAINING


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
  add_wandb_args(parser)
  args = parser.parse_args()

  rng = np.random.default_rng(args.seed)
  dataset = load_dataset(args.record_dir, args.limit)
  model = GridcraftVAE()
  os.makedirs(args.out_dir, exist_ok=True)
  logger = logger_from_args(
    args,
    config={**vars(args), "dataset_size": int(len(dataset))},
    default_group="vae",
    default_name=f"gridcraft-vae-seed{args.seed}",
    tags=["gridcraft", "vae"],
    info_sections=[GENERAL, WORLD_MODEL_TRAINING],
    out_dir=args.out_dir,
  )
  logger.log({"vae_dataset_size": int(len(dataset))}, step=0, namespace="wm_training")

  for step in range(args.steps):
    indices = rng.integers(0, len(dataset), size=args.batch_size)
    loss, grid_loss, self_loss, kl = model.train_batch(dataset[indices])
    if step == 0 or (step + 1) % 100 == 0:
      print("step", step + 1, "loss", loss, "grid", grid_loss, "self", self_loss, "kl", kl)
      logger.log({
        "vae_loss": loss,
        "vae_grid_loss": grid_loss,
        "vae_self_loss": self_loss,
        "vae_kl_loss": kl,
      }, step=step + 1, namespace="wm_training")

  model.save_json(os.path.join(args.out_dir, "vae.json"))
  print("saved", os.path.join(args.out_dir, "vae.json"))
  logger.log_summary({
    "vae_loss": loss,
    "vae_grid_loss": grid_loss,
    "vae_self_loss": self_loss,
    "vae_kl_loss": kl,
  }, namespace="wm_training")
  logger.finish()


if __name__ == "__main__":
  main()
