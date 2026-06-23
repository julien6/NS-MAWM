import argparse
import json
import os
import time

import numpy as np
import tensorflow as tf

from dream_env import GridcraftDreamEnv
from env import ACTION_SIZE, make_env
from experiment_logging import add_wandb_args, logger_from_args, should_log_wandb_videos
from exp_config import OBS_SIZE, Z_SIZE
from ns_symbolic import NS_VARIANTS, apply_symbolic_projection, tabular_to_vector
from progress_logging import append_progress
from rnn.rnn import GridcraftRNN, rnn_init_state
from vae.vae import GridcraftVAE
from video_logging import (
  record_actor_policy_evaluation_video,
  record_mpc_cem_evaluation_video,
)
from wandb_schema import GENERAL, MARL_EVALUATION, MARL_TRAINING


class ActorCritic(tf.keras.Model):
  def __init__(self, obs_size, action_size=ACTION_SIZE, hidden_size=128):
    super().__init__()
    self.net = tf.keras.Sequential([
      tf.keras.layers.Input(shape=(obs_size,)),
      tf.keras.layers.Dense(hidden_size, activation="tanh"),
      tf.keras.layers.Dense(hidden_size, activation="tanh"),
    ])
    self.logits = tf.keras.layers.Dense(action_size)
    self.value = tf.keras.layers.Dense(1)

  def call(self, x):
    h = self.net(x)
    return self.logits(h), tf.squeeze(self.value(h), axis=-1)

  def act(self, obs, rng, deterministic=False):
    logits, value = self(np.asarray(obs, dtype=np.float32).reshape(1, -1), training=False)
    logits = logits.numpy()[0]
    if deterministic:
      action = int(np.argmax(logits))
    else:
      probs = softmax(logits)
      action = int(rng.choice(np.arange(len(probs)), p=probs))
    logprob = float(log_softmax(logits)[action])
    return action, logprob, float(value.numpy()[0])


def softmax(logits):
  logits = np.asarray(logits, dtype=np.float64)
  logits = logits - np.max(logits)
  exp = np.exp(logits)
  return exp / np.sum(exp)


def log_softmax(logits):
  logits = np.asarray(logits, dtype=np.float64)
  logits = logits - np.max(logits)
  logsum = np.log(np.sum(np.exp(logits)))
  return logits - logsum


def discounted_returns(rewards, gamma):
  returns = []
  value = 0.0
  for reward in reversed(rewards):
    value = float(reward) + gamma * value
    returns.append(value)
  return np.asarray(list(reversed(returns)), dtype=np.float32)


def make_policy_env(kind, seed, max_steps, rnn_json, initial_z_json):
  if kind == "real":
    return make_env(seed=seed, render_mode=False, max_steps=max_steps)
  if kind == "imagined":
    return GridcraftDreamEnv(seed=seed, max_steps=max_steps, rnn_path=rnn_json, initial_z_path=initial_z_json)
  raise ValueError(f"unknown policy env kind: {kind}")


class GridcraftNSDreamEnv:
  def __init__(self, seed, max_steps, vae_json, rnn_json, initial_z_json, ns_variant, symbolic_coverage):
    self.base = GridcraftDreamEnv(seed=seed, max_steps=max_steps, rnn_path=rnn_json, initial_z_path=initial_z_json)
    self.vae = GridcraftVAE()
    self.vae.load_json(vae_json)
    self.ns_variant = ns_variant
    self.symbolic_coverage = symbolic_coverage
    self.max_steps = max_steps

  def reset(self, seed=None):
    self.z = self.base.reset(seed=seed)
    self.current_obs = self.vae.decode_tabular(self.z)
    return self.z

  def step(self, action):
    next_z, reward, done, info = self.base.step(action)
    imagined_obs = self.vae.decode_tabular(next_z)
    projected_obs, _ = apply_symbolic_projection(
      imagined_obs,
      self.current_obs,
      action,
      self.ns_variant,
      coverage=self.symbolic_coverage,
    )
    if projected_obs is not imagined_obs:
      vec = tabular_to_vector(projected_obs)
      mu, _ = self.vae.encode_mu_logvar(vec.reshape(1, -1))
      next_z = mu[0]
      self.base.z = next_z
    self.z = next_z
    self.current_obs = projected_obs
    return self.z, reward, done, info

  def close(self):
    self.base.close()


def train_actor_critic(args):
  rng = np.random.default_rng(args.seed)
  obs_size = OBS_SIZE if args.train_env == "real" else Z_SIZE
  model = ActorCritic(obs_size=obs_size, hidden_size=args.hidden_size)
  optimizer = tf.keras.optimizers.Adam(args.learning_rate)
  logger = logger_from_args(
    args,
    config=vars(args),
    default_group=args.baseline_id,
    default_name=args.run_name or f"{args.baseline_id}_{args.policy_baseline}_seed{args.seed}",
    tags=["gridcraft", "policy", args.policy_baseline, args.train_env],
    info_sections=[GENERAL, MARL_TRAINING, MARL_EVALUATION],
    out_dir=args.out_dir,
  )

  os.makedirs(args.out_dir, exist_ok=True)
  training_rewards = []
  for update in range(args.updates):
    obs_batch = []
    action_batch = []
    return_batch = []
    reward_totals = []
    lengths = []
    start = time.time()
    for episode in range(args.episodes_per_update):
      env = make_train_env(args.train_env, args.seed + update * 1000 + episode, args)
      obs = env.reset(seed=args.seed + update * 1000 + episode)
      rewards = []
      observations = []
      actions = []
      total = 0.0
      for t in range(args.max_steps):
        action, _, _ = model.act(obs, rng)
        next_obs, reward, done, _ = env.step(action)
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        total += reward
        obs = next_obs
        if done:
          break
      env.close()
      returns = discounted_returns(rewards, args.gamma)
      obs_batch.extend(observations)
      action_batch.extend(actions)
      return_batch.extend(returns)
      reward_totals.append(total)
      lengths.append(t + 1)

    metrics = update_actor_critic(model, optimizer, obs_batch, action_batch, return_batch, args.entropy_coef, args.value_coef)
    metrics.update({
      f"training_{args.train_env}_reward": float(np.mean(reward_totals)),
      "episode_length": float(np.mean(lengths)),
      "step_time_ms": float((time.time() - start) * 1000.0),
    })
    training_rewards.append(metrics[f"training_{args.train_env}_reward"])
    logger.log(metrics, step=update + 1, namespace="marl_training")
    append_progress(args.progress_log, metrics, step=update + 1, namespace="marl_training")
    if update == 0 or (update + 1) % args.eval_every == 0:
      checkpoint_path = os.path.join(args.out_dir, f"policy_step_{update + 1}.weights.h5")
      os.makedirs(args.out_dir, exist_ok=True)
      model.save_weights(checkpoint_path)
      eval_metrics = evaluate_policy(model, args, update + 1)
      eval_metrics["policy_checkpoint_step"] = update + 1
      logger.log(eval_metrics, step=update + 1, namespace="marl_evaluation")
      append_progress(args.progress_log, eval_metrics, step=update + 1, namespace="marl_evaluation")
      if should_log_wandb_videos(args):
        try:
          frames = record_actor_policy_evaluation_video(
            model,
            policy_baseline=args.policy_baseline,
            vae_json=args.vae_json,
            rnn_json=args.rnn_json,
            ns_variant=args.ns_variant,
            symbolic_coverage=args.symbolic_coverage,
            seed=args.seed + 70000 + update,
            episodes=args.video_episodes,
            max_steps=args.video_max_steps,
          )
          logged = logger.log_video(
            "video_policy_rollout",
            frames,
            fps=args.video_fps,
            step=update + 1,
            namespace="marl_evaluation",
          )
          logger.log({
            "video_policy_rollout_logged": int(bool(logged)),
            "video_policy_rollout_skipped_media_dependency": int(not bool(logged)),
          }, step=update + 1, namespace="marl_evaluation")
        except Exception as exc:
          print(f"Skipping W&B policy video after generation failure: {exc}", flush=True)
          logger.log({
            "video_policy_rollout_logged": 0,
            "video_policy_rollout_generation_failed": 1,
          }, step=update + 1, namespace="marl_evaluation")
      else:
        logger.log({
          "video_policy_rollout_logged": 0,
          "video_policy_rollout_skipped_disabled": 1,
        }, step=update + 1, namespace="marl_evaluation")
      save_json(os.path.join(args.out_dir, f"policy_eval_step_{update + 1}.json"), eval_metrics)
      print("update", update + 1, metrics, eval_metrics, flush=True)

  model.save_weights(os.path.join(args.out_dir, "policy.weights.h5"))
  summary = {
    "training_reward_mean": float(np.mean(training_rewards)) if training_rewards else 0.0,
    "training_reward_final": float(training_rewards[-1]) if training_rewards else 0.0,
  }
  save_json(os.path.join(args.out_dir, "policy_summary.json"), summary)
  logger.log_summary(summary, namespace="marl_training")
  logger.save_json(os.path.join(args.out_dir, "policy_summary.json"), summary)
  logger.finish()
  return summary


def update_actor_critic(model, optimizer, observations, actions, returns, entropy_coef, value_coef):
  observations = tf.convert_to_tensor(np.asarray(observations, dtype=np.float32), dtype=tf.float32)
  actions = tf.convert_to_tensor(np.asarray(actions, dtype=np.int32), dtype=tf.int32)
  returns = tf.convert_to_tensor(np.asarray(returns, dtype=np.float32), dtype=tf.float32)
  with tf.GradientTape() as tape:
    logits, values = model(observations, training=True)
    log_probs = tf.nn.log_softmax(logits)
    probs = tf.nn.softmax(logits)
    selected = tf.gather(log_probs, actions, batch_dims=1)
    advantages = returns - tf.stop_gradient(values)
    policy_loss = -tf.reduce_mean(selected * advantages)
    value_loss = tf.reduce_mean(tf.square(returns - values))
    entropy = -tf.reduce_mean(tf.reduce_sum(probs * log_probs, axis=1))
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
  return {
    "policy_loss": float(policy_loss.numpy()),
    "value_loss": float(value_loss.numpy()),
    "entropy": float(entropy.numpy()),
    "policy_total_loss": float(loss.numpy()),
  }


def evaluate_policy(model, args, step):
  real_rewards, real_lengths = rollout_policy(model, "real", args, deterministic=True)
  metrics = {
    "eval_real_reward": float(np.mean(real_rewards)),
    "eval_real_reward_std": float(np.std(real_rewards)),
    "eval_real_episode_length": float(np.mean(real_lengths)),
  }
  if args.train_env == "imagined":
    imagined_rewards, imagined_lengths = rollout_policy(model, "imagined", args, deterministic=True)
    metrics.update({
      "eval_imagined_reward": float(np.mean(imagined_rewards)),
      "eval_imagined_reward_std": float(np.std(imagined_rewards)),
      "eval_imagined_episode_length": float(np.mean(imagined_lengths)),
      "real_imagined_reward_gap": float(np.mean(real_rewards) - np.mean(imagined_rewards)),
    })
  return metrics


def rollout_policy(model, kind, args, deterministic=True):
  rewards = []
  lengths = []
  rng = np.random.default_rng(args.seed + 9999)
  vae = None
  if kind == "real" and model.net.input_shape[-1] == Z_SIZE:
    vae = GridcraftVAE()
    vae.load_json(args.vae_json)
  for episode in range(args.eval_episodes):
    env = make_train_env(kind, args.seed + 50000 + episode, args)
    obs = env.reset(seed=args.seed + 50000 + episode)
    total = 0.0
    for t in range(args.max_steps):
      policy_obs = obs
      if vae is not None:
        mu, _ = vae.encode_mu_logvar(obs.reshape(1, -1))
        policy_obs = mu[0]
      action, _, _ = model.act(policy_obs, rng, deterministic=deterministic)
      obs, reward, done, _ = env.step(action)
      total += reward
      if done:
        break
    env.close()
    rewards.append(total)
    lengths.append(t + 1)
  return rewards, lengths


def run_mpc_cem(args):
  logger = logger_from_args(
    args,
    config=vars(args),
    default_group=args.baseline_id,
    default_name=args.run_name or f"{args.baseline_id}_mpc_cem_seed{args.seed}",
    tags=["gridcraft", "policy", "mpc-cem", args.ns_variant],
    info_sections=[GENERAL, MARL_EVALUATION],
    out_dir=args.out_dir,
  )
  vae = GridcraftVAE()
  vae.load_json(args.vae_json)
  rnn = GridcraftRNN()
  rnn.load_json(args.rnn_json)
  rng = np.random.default_rng(args.seed)
  rewards = []
  imagined_returns = []
  entropies = []
  planning_times = []
  lengths = []
  for episode in range(args.eval_episodes):
    env = make_env(seed=args.seed + episode, render_mode=False, max_steps=args.max_steps)
    obs = env.reset(seed=args.seed + episode)
    state = rnn_init_state(rnn)
    current_tabular = env.last_obs
    total = 0.0
    for t in range(args.max_steps):
      mu, _ = vae.encode_mu_logvar(obs.reshape(1, -1))
      planning_start = time.time()
      if args.batched_cem:
        action, imagined_return, entropy = cem_action_batched(rnn, vae, mu[0], state, current_tabular, rng, args)
      else:
        action, imagined_return, entropy = cem_action(rnn, vae, mu[0], state, current_tabular, rng, args)
      planning_step_time_ms = float((time.time() - planning_start) * 1000.0)
      logmix, mean, logstd, reward_pred, done_logit, state = rnn.step(mu[0], action, state)
      obs, reward, done, _ = env.step(action)
      current_tabular = env.last_obs
      total += reward
      imagined_returns.append(imagined_return)
      entropies.append(entropy)
      planning_times.append(planning_step_time_ms)
      if done:
        break
    env.close()
    rewards.append(total)
    lengths.append(t + 1)
    metrics = {
      "planning_real_return": float(total),
      "planning_imagined_return": float(np.mean(imagined_returns)) if imagined_returns else 0.0,
      "real_imagined_reward_gap": float(total - np.mean(imagined_returns)) if imagined_returns else float(total),
      "planning_action_entropy": float(np.mean(entropies)) if entropies else 0.0,
      "planning_step_time_ms": float(np.mean(planning_times)) if planning_times else 0.0,
      "planning_batch_size": float(args.cem_samples if args.batched_cem else 1),
      "planning_horizon": float(args.planning_horizon),
      "planning_device": planning_device(),
      "episode_length": float(t + 1),
    }
    logger.log(metrics, step=episode + 1, namespace="marl_evaluation")
    append_progress(args.progress_log, metrics, step=episode + 1, namespace="marl_evaluation")
  summary = {
    "eval_real_reward": float(np.mean(rewards)),
    "eval_real_reward_std": float(np.std(rewards)),
    "planning_imagined_return": float(np.mean(imagined_returns)) if imagined_returns else 0.0,
    "real_imagined_reward_gap": float(np.mean(rewards) - np.mean(imagined_returns)) if imagined_returns else float(np.mean(rewards)),
    "planning_action_entropy": float(np.mean(entropies)) if entropies else 0.0,
    "planning_step_time_ms": float(np.mean(planning_times)) if planning_times else 0.0,
    "planning_batch_size": float(args.cem_samples if args.batched_cem else 1),
    "planning_horizon": float(args.planning_horizon),
    "planning_device": planning_device(),
    "eval_real_episode_length": float(np.mean(lengths)),
  }
  os.makedirs(args.out_dir, exist_ok=True)
  logger.save_json(os.path.join(args.out_dir, "mpc_cem_summary.json"), summary)
  append_progress(args.progress_log, summary, namespace="marl_evaluation")
  if should_log_wandb_videos(args):
    try:
      frames = record_mpc_cem_evaluation_video(
        vae_json=args.vae_json,
        rnn_json=args.rnn_json,
        ns_variant=args.ns_variant,
        symbolic_coverage=args.symbolic_coverage,
        seed=args.seed + 70000,
        episodes=args.video_episodes,
        max_steps=args.video_max_steps,
        planning_horizon=args.planning_horizon,
        cem_samples=args.cem_samples,
        cem_elite_frac=args.cem_elite_frac,
        gamma=args.gamma,
      )
      logged = logger.log_video(
        "video_policy_rollout",
        frames,
        fps=args.video_fps,
        namespace="marl_evaluation",
      )
      logger.log({
        "video_policy_rollout_logged": int(bool(logged)),
        "video_policy_rollout_skipped_media_dependency": int(not bool(logged)),
      }, namespace="marl_evaluation")
    except Exception as exc:
      print(f"Skipping W&B policy video after generation failure: {exc}", flush=True)
      logger.log({
        "video_policy_rollout_logged": 0,
        "video_policy_rollout_generation_failed": 1,
      }, namespace="marl_evaluation")
  else:
    logger.log({
      "video_policy_rollout_logged": 0,
      "video_policy_rollout_skipped_disabled": 1,
    }, namespace="marl_evaluation")
  logger.log_summary(summary, namespace="marl_evaluation")
  logger.finish()
  print(json.dumps(summary, indent=2))
  return summary


def make_train_env(kind, seed, args):
  if kind == "imagined" and args.ns_variant in ("projection", "residual"):
    return GridcraftNSDreamEnv(
      seed=seed,
      max_steps=args.max_steps,
      vae_json=args.vae_json,
      rnn_json=args.rnn_json,
      initial_z_json=args.initial_z_json,
      ns_variant=args.ns_variant,
      symbolic_coverage=args.symbolic_coverage,
    )
  return make_policy_env(kind, seed, args.max_steps, args.rnn_json, args.initial_z_json)


def cem_action(rnn, vae, z, state, current_obs, rng, args):
  sequences = rng.integers(0, ACTION_SIZE, size=(args.cem_samples, args.planning_horizon))
  returns = np.zeros((args.cem_samples,), dtype=np.float32)
  for i, sequence in enumerate(sequences):
    rollout_z = np.asarray(z, dtype=np.float32)
    rollout_state = [tf.identity(state[0]), tf.identity(state[1])]
    rollout_obs = current_obs
    total = 0.0
    discount = 1.0
    for action in sequence:
      logmix, mean, logstd, reward, done_logit, rollout_state = rnn.step(rollout_z, int(action), rollout_state)
      mix = np.exp(logmix - np.max(logmix, axis=1, keepdims=True))
      mix = mix / np.sum(mix, axis=1, keepdims=True)
      rollout_z = np.sum(mix * mean, axis=1).astype(np.float32)
      if args.ns_variant in ("projection", "residual"):
        imagined_obs = vae.decode_tabular(rollout_z)
        projected_obs, _ = apply_symbolic_projection(
          imagined_obs,
          rollout_obs,
          int(action),
          args.ns_variant,
          coverage=args.symbolic_coverage,
        )
        if projected_obs is not imagined_obs:
          vec = tabular_to_vector(projected_obs)
          mu, _ = vae.encode_mu_logvar(vec.reshape(1, -1))
          rollout_z = mu[0]
        rollout_obs = projected_obs
      total += discount * float(reward)
      discount *= args.gamma
      if done_logit > 0:
        break
    returns[i] = total
  elite_count = max(1, int(args.cem_elite_frac * args.cem_samples))
  elite_idx = np.argsort(returns)[-elite_count:]
  first_actions = sequences[elite_idx, 0]
  counts = np.bincount(first_actions, minlength=ACTION_SIZE).astype(np.float32)
  probs = counts / np.sum(counts)
  action = int(np.argmax(probs))
  entropy = float(-np.sum(probs[probs > 0] * np.log(probs[probs > 0])))
  return action, float(np.mean(returns[elite_idx])), entropy


def cem_action_batched(rnn, vae, z, state, current_obs, rng, args):
  sequences = rng.integers(0, ACTION_SIZE, size=(args.cem_samples, args.planning_horizon), dtype=np.int32)
  rollout_z = np.repeat(np.asarray(z, dtype=np.float32).reshape(1, -1), args.cem_samples, axis=0)
  rollout_state = [
    tf.repeat(tf.convert_to_tensor(state[0], dtype=tf.float32), args.cem_samples, axis=0),
    tf.repeat(tf.convert_to_tensor(state[1], dtype=tf.float32), args.cem_samples, axis=0),
  ]
  returns = np.zeros((args.cem_samples,), dtype=np.float32)
  active = np.ones((args.cem_samples,), dtype=np.float32)
  discount = 1.0
  rollout_obs = None
  if args.ns_variant in ("projection", "residual"):
    rollout_obs = [
      {
        "grid": np.asarray(current_obs["grid"]).copy(),
        "self": np.asarray(current_obs["self"]).copy(),
      }
      for _ in range(args.cem_samples)
    ]

  for t in range(args.planning_horizon):
    actions = sequences[:, t]
    next_z, reward, done_logit, rollout_state, _ = rnn.step_batch(
      rollout_z,
      actions,
      rollout_state,
      deterministic=args.batched_cem_deterministic_z,
      rng=rng,
    )
    reward_np = reward.numpy().astype(np.float32)
    done_np = (done_logit.numpy() > 0).astype(np.float32)
    rollout_z = next_z.numpy().astype(np.float32)

    if args.ns_variant in ("projection", "residual"):
      rollout_z = apply_batched_symbolic_projection(vae, rollout_z, rollout_obs, actions, args)

    returns += active * discount * reward_np
    active *= 1.0 - done_np
    discount *= args.gamma
    if not np.any(active):
      break

  elite_count = max(1, int(args.cem_elite_frac * args.cem_samples))
  elite_idx = np.argsort(returns)[-elite_count:]
  first_actions = sequences[elite_idx, 0]
  counts = np.bincount(first_actions, minlength=ACTION_SIZE).astype(np.float32)
  probs = counts / np.sum(counts)
  action = int(np.argmax(probs))
  entropy = float(-np.sum(probs[probs > 0] * np.log(probs[probs > 0])))
  return action, float(np.mean(returns[elite_idx])), entropy


def apply_batched_symbolic_projection(vae, rollout_z, rollout_obs, actions, args):
  if args.batched_cem_symbolic_mode != "cpu_projection":
    raise ValueError(f"unsupported batched CEM symbolic mode: {args.batched_cem_symbolic_mode}")
  decoded = vae.decode_tabular(rollout_z)
  if isinstance(decoded, dict):
    decoded = [decoded]
  projected_z = np.asarray(rollout_z, dtype=np.float32).copy()
  encoded_rows = []
  encoded_indices = []
  for i, imagined_obs in enumerate(decoded):
    projected_obs, _ = apply_symbolic_projection(
      imagined_obs,
      rollout_obs[i],
      int(actions[i]),
      args.ns_variant,
      coverage=args.symbolic_coverage,
    )
    rollout_obs[i] = projected_obs
    vec = tabular_to_vector(projected_obs)
    encoded_rows.append(vec)
    encoded_indices.append(i)
  if encoded_rows:
    mu, _ = vae.encode_mu_logvar(np.asarray(encoded_rows, dtype=np.float32))
    projected_z[np.asarray(encoded_indices, dtype=np.int64)] = mu.astype(np.float32)
  return projected_z


def planning_device():
  gpus = tf.config.list_logical_devices("GPU")
  if gpus:
    return gpus[0].name
  return "CPU"


def save_json(path, payload):
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w") as f:
    json.dump(payload, f, indent=2)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--policy-baseline", choices=["real_mappo", "imagined_mappo", "mpc_cem"], required=True)
  parser.add_argument("--baseline-id", default="B00")
  parser.add_argument("--run-name", default=None)
  parser.add_argument("--out-dir", default="runs/policy")
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--max-steps", type=int, default=500)
  parser.add_argument("--updates", type=int, default=100)
  parser.add_argument("--episodes-per-update", type=int, default=8)
  parser.add_argument("--eval-every", type=int, default=10)
  parser.add_argument("--eval-episodes", type=int, default=10)
  parser.add_argument("--hidden-size", type=int, default=128)
  parser.add_argument("--learning-rate", type=float, default=3e-4)
  parser.add_argument("--gamma", type=float, default=0.99)
  parser.add_argument("--entropy-coef", type=float, default=0.01)
  parser.add_argument("--value-coef", type=float, default=0.5)
  parser.add_argument("--vae-json", default="vae/vae.json")
  parser.add_argument("--rnn-json", default="rnn/rnn.json")
  parser.add_argument("--initial-z-json", default="initial_z/initial_z.json")
  parser.add_argument("--ns-variant", choices=NS_VARIANTS, default="neural")
  parser.add_argument("--symbolic-coverage", type=float, default=1.0)
  parser.add_argument("--planning-horizon", type=int, default=15)
  parser.add_argument("--cem-samples", type=int, default=64)
  parser.add_argument("--cem-elite-frac", type=float, default=0.2)
  parser.add_argument("--batched-cem", action="store_true")
  batch_z_group = parser.add_mutually_exclusive_group()
  batch_z_group.add_argument("--batched-cem-deterministic-z", dest="batched_cem_deterministic_z", action="store_true", default=True)
  batch_z_group.add_argument("--batched-cem-sample-z", dest="batched_cem_deterministic_z", action="store_false")
  parser.add_argument("--batched-cem-symbolic-mode", choices=["cpu_projection"], default="cpu_projection")
  parser.add_argument("--progress-log", default=None)
  add_wandb_args(parser)
  args = parser.parse_args()
  if args.policy_baseline == "mpc_cem":
    run_mpc_cem(args)
  else:
    args.train_env = "real" if args.policy_baseline == "real_mappo" else "imagined"
    train_actor_critic(args)


if __name__ == "__main__":
  main()
