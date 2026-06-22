import argparse
import os

import numpy as np

from vae.vae import GridcraftVAE


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--record-dir", default="record")
  parser.add_argument("--model-dir", default="vae")
  parser.add_argument("--out-dir", default="series")
  parser.add_argument("--limit", type=int, default=None)
  parser.add_argument("--sample-z", action="store_true")
  args = parser.parse_args()

  files = sorted([f for f in os.listdir(args.record_dir) if f.endswith(".npz")])
  if args.limit is not None:
    files = files[:args.limit]
  if not files:
    raise RuntimeError(f"no .npz files found in {args.record_dir}")

  vae = GridcraftVAE()
  vae.load_json(os.path.join(args.model_dir, "vae.json"))
  os.makedirs(args.out_dir, exist_ok=True)

  z_list = []
  obs_list = []
  action_list = []
  reward_list = []
  done_list = []
  length_list = []
  initial_z = []

  for filename in files:
    data = np.load(os.path.join(args.record_dir, filename))
    obs = data["obs"].astype(np.float32)
    mu, logvar = vae.encode_mu_logvar(obs)
    if args.sample_z:
      z = mu + np.exp(logvar / 2.0) * np.random.randn(*logvar.shape)
    else:
      z = mu
    z_list.append(z.astype(np.float32))
    obs_list.append(obs.astype(np.float32))
    action_list.append(data["action"].astype(np.int16))
    reward_list.append(data["reward"].astype(np.float32))
    done_list.append(data["done"].astype(np.bool_))
    length_list.append(len(z))
    if len(z):
      initial_z.append(z[0].astype(np.float32))
    print("encoded", filename, "steps", len(z))

  max_len = max(length_list)
  z_array = np.zeros((len(z_list), max_len, z_list[0].shape[1]), dtype=np.float32)
  obs_array = np.zeros((len(obs_list), max_len, obs_list[0].shape[1]), dtype=np.float32)
  action_array = np.zeros((len(z_list), max_len), dtype=np.int16)
  reward_array = np.zeros((len(z_list), max_len), dtype=np.float32)
  done_array = np.ones((len(z_list), max_len), dtype=np.bool_)
  mask_array = np.zeros((len(z_list), max_len), dtype=np.bool_)
  for i, (z, obs, action, reward, done) in enumerate(zip(z_list, obs_list, action_list, reward_list, done_list)):
    n = len(z)
    z_array[i, :n] = z
    obs_array[i, :n] = obs
    action_array[i, :n] = action
    reward_array[i, :n] = reward
    done_array[i, :n] = done
    mask_array[i, :n] = True

  np.savez_compressed(
    os.path.join(args.out_dir, "series.npz"),
    z=z_array,
    obs=obs_array,
    action=action_array,
    reward=reward_array,
    done=done_array,
    mask=mask_array,
    length=np.asarray(length_list, dtype=np.int32),
  )
  np.savez_compressed(os.path.join(args.out_dir, "initial_z.npz"), z=np.asarray(initial_z, dtype=np.float32))
  print("saved", os.path.join(args.out_dir, "series.npz"))


if __name__ == "__main__":
  main()
