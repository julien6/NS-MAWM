import numpy as np
import tensorflow as tf

from env import ACTION_SIZE, make_env
from exp_config import OBS_SIZE, Z_SIZE
from model import (
  GridcraftController,
  TabularRenderWorker,
  decode_tabular_observation,
  predict_rnn_next_z,
  world_snapshot,
)
from ns_symbolic import apply_symbolic_projection, tabular_to_vector
from rnn.rnn import GridcraftRNN, rnn_init_state
from vae.vae import GridcraftVAE


ACTION_NAMES = [
  "stay",
  "move_n",
  "move_s",
  "move_w",
  "move_e",
  "harvest",
  "pickup",
  "attack",
  "eat",
  "craft_plank",
  "craft_stick",
  "craft_wood_sword",
  "craft_stone_sword",
  "craft_wood_pickaxe",
  "craft_stone_pickaxe",
]


def record_world_model_comparison_video(
    vae_json="vae/vae.json",
    rnn_json="rnn/rnn.json",
    ns_variant="neural",
    symbolic_coverage=1.0,
    seed=1,
    episodes=1,
    max_steps=100,
    imagination_mode="mean",
):
  controller = GridcraftController(vae_path=vae_json, rnn_path=rnn_json, load_world_model=True)
  renderer = TabularRenderWorker(display=False)
  frames = []
  try:
    for episode in range(max(1, int(episodes))):
      episode_seed = int(seed) + episode
      rng = np.random.default_rng(episode_seed)
      env = make_env(seed=episode_seed, render_mode=False, max_steps=max_steps)
      obs = env.reset(seed=episode_seed)
      controller.reset()
      z = controller.encode_obs(obs)
      imagined_obs = _copy_tabular_obs(env.last_obs)
      cumulative_reward = 0.0
      frames.append(_as_rgb(renderer.render(
        {"agent_0": imagined_obs},
        world=world_snapshot(env),
        overlay_info=_overlay(0, "initial", 0.0, cumulative_reward, False),
      )))
      for step_index in range(max_steps):
        action = int(rng.integers(0, ACTION_SIZE))
        current_imagined_obs = imagined_obs
        imagined_z = predict_rnn_next_z(controller, z, action, rng=rng, mode=imagination_mode)
        obs, reward, done, info = env.step(action)
        imagined_obs = decode_tabular_observation(controller, imagined_z)["agent_0"]
        imagined_obs, _ = apply_symbolic_projection(
          imagined_obs,
          current_imagined_obs,
          action,
          ns_variant,
          coverage=symbolic_coverage,
        )
        z = _encode_tabular(controller, imagined_obs) if ns_variant in ("projection", "residual") else imagined_z
        cumulative_reward += float(reward)
        frames.append(_as_rgb(renderer.render(
          {"agent_0": imagined_obs},
          world=world_snapshot(env),
          overlay_info=_overlay(step_index + 1, _action_name(action), reward, cumulative_reward, done),
        )))
        if done:
          break
      env.close()
  finally:
    renderer.close()
  return _stack_frames(frames)


def record_real_policy_video(policy_fn=None, seed=1, episodes=1, max_steps=100):
  renderer = TabularRenderWorker(display=False)
  frames = []
  try:
    for episode in range(max(1, int(episodes))):
      episode_seed = int(seed) + episode
      rng = np.random.default_rng(episode_seed)
      env = make_env(seed=episode_seed, render_mode=False, max_steps=max_steps)
      obs = env.reset(seed=episode_seed)
      for _ in range(max_steps):
        frames.append(_as_rgb(renderer.render(None, world=world_snapshot(env))))
        action = int(policy_fn(obs)) if policy_fn is not None else int(rng.integers(0, ACTION_SIZE))
        obs, reward, done, info = env.step(action)
        if done:
          frames.append(_as_rgb(renderer.render(None, world=world_snapshot(env))))
          break
      env.close()
  finally:
    renderer.close()
  return _stack_frames(frames)


def record_actor_policy_evaluation_video(
    actor,
    policy_baseline,
    vae_json="vae/vae.json",
    rnn_json="rnn/rnn.json",
    ns_variant="neural",
    symbolic_coverage=1.0,
    seed=1,
    episodes=1,
    max_steps=100,
):
  if policy_baseline == "real_mappo":
    return record_real_policy_video(
      policy_fn=lambda obs: actor.act(obs, np.random.default_rng(seed), deterministic=True)[0],
      seed=seed,
      episodes=episodes,
      max_steps=max_steps,
    )

  rng = np.random.default_rng(seed)
  vae = GridcraftVAE()
  vae.load_json(vae_json)
  controller = GridcraftController(vae_path=vae_json, rnn_path=rnn_json, load_world_model=True)
  renderer = TabularRenderWorker(display=False)
  frames = []
  try:
    for episode in range(max(1, int(episodes))):
      episode_seed = int(seed) + episode
      env = make_env(seed=episode_seed, render_mode=False, max_steps=max_steps)
      obs = env.reset(seed=episode_seed)
      controller.reset()
      z, _ = vae.encode_mu_logvar(obs.reshape(1, -1))
      z = z[0]
      imagined_obs = _copy_tabular_obs(env.last_obs)
      cumulative_reward = 0.0
      frames.append(_as_rgb(renderer.render(
        {"agent_0": imagined_obs},
        world=world_snapshot(env),
        overlay_info=_overlay(0, "initial", 0.0, cumulative_reward, False),
      )))
      for step_index in range(max_steps):
        action = actor.act(z, rng, deterministic=True)[0]
        current_imagined_obs = imagined_obs
        imagined_z = predict_rnn_next_z(controller, z, action, rng=rng, mode="mean")
        obs, reward, done, info = env.step(action)
        imagined_obs = decode_tabular_observation(controller, imagined_z)["agent_0"]
        imagined_obs, _ = apply_symbolic_projection(imagined_obs, current_imagined_obs, action, ns_variant, coverage=symbolic_coverage)
        z = _encode_tabular(controller, imagined_obs) if ns_variant in ("projection", "residual") else imagined_z
        cumulative_reward += float(reward)
        frames.append(_as_rgb(renderer.render(
          {"agent_0": imagined_obs},
          world=world_snapshot(env),
          overlay_info=_overlay(step_index + 1, _action_name(action), reward, cumulative_reward, done),
        )))
        if done:
          break
      env.close()
  finally:
    renderer.close()
  return _stack_frames(frames)


def record_mpc_cem_evaluation_video(
    vae_json="vae/vae.json",
    rnn_json="rnn/rnn.json",
    ns_variant="neural",
    symbolic_coverage=1.0,
    seed=1,
    episodes=1,
    max_steps=100,
    planning_horizon=15,
    cem_samples=64,
    cem_elite_frac=0.2,
    gamma=0.99,
):
  rng = np.random.default_rng(seed)
  vae = GridcraftVAE()
  vae.load_json(vae_json)
  rnn = GridcraftRNN()
  rnn.load_json(rnn_json)
  controller = GridcraftController(vae_path=vae_json, rnn_path=rnn_json, load_world_model=True)
  renderer = TabularRenderWorker(display=False)
  frames = []
  try:
    for episode in range(max(1, int(episodes))):
      episode_seed = int(seed) + episode
      env = make_env(seed=episode_seed, render_mode=False, max_steps=max_steps)
      obs = env.reset(seed=episode_seed)
      state = rnn_init_state(rnn)
      z, _ = vae.encode_mu_logvar(obs.reshape(1, -1))
      z = z[0]
      imagined_obs = _copy_tabular_obs(env.last_obs)
      cumulative_reward = 0.0
      frames.append(_as_rgb(renderer.render(
        {"agent_0": imagined_obs},
        world=world_snapshot(env),
        overlay_info=_overlay(0, "initial", 0.0, cumulative_reward, False),
      )))
      for step_index in range(max_steps):
        current_imagined_obs = imagined_obs
        action, _ = _cem_action(
          rnn,
          vae,
          z,
          state,
          current_imagined_obs,
          rng,
          ns_variant,
          symbolic_coverage,
          planning_horizon,
          cem_samples,
          cem_elite_frac,
          gamma,
        )
        controller.rnn_state = state
        imagined_z = predict_rnn_next_z(controller, z, action, rng=rng, mode="mean")
        state = controller.rnn_state
        obs, reward, done, info = env.step(action)
        imagined_obs = controller.vae.decode_tabular(imagined_z)
        imagined_obs, _ = apply_symbolic_projection(imagined_obs, current_imagined_obs, action, ns_variant, coverage=symbolic_coverage)
        z = _encode_tabular(controller, imagined_obs) if ns_variant in ("projection", "residual") else imagined_z
        cumulative_reward += float(reward)
        frames.append(_as_rgb(renderer.render(
          {"agent_0": imagined_obs},
          world=world_snapshot(env),
          overlay_info=_overlay(step_index + 1, _action_name(action), reward, cumulative_reward, done),
        )))
        if done:
          break
      env.close()
  finally:
    renderer.close()
  return _stack_frames(frames)


def _cem_action(rnn, vae, z, state, current_obs, rng, ns_variant, symbolic_coverage, planning_horizon, cem_samples, cem_elite_frac, gamma):
  sequences = rng.integers(0, ACTION_SIZE, size=(cem_samples, planning_horizon))
  returns = np.zeros((cem_samples,), dtype=np.float32)
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
      if ns_variant in ("projection", "residual"):
        imagined_obs = vae.decode_tabular(rollout_z)
        projected_obs, _ = apply_symbolic_projection(imagined_obs, rollout_obs, int(action), ns_variant, coverage=symbolic_coverage)
        if projected_obs is not imagined_obs:
          vec = tabular_to_vector(projected_obs)
          mu, _ = vae.encode_mu_logvar(vec.reshape(1, -1))
          rollout_z = mu[0]
        rollout_obs = projected_obs
      total += discount * float(reward)
      discount *= gamma
      if done_logit > 0:
        break
    returns[i] = total
  elite_count = max(1, int(cem_elite_frac * cem_samples))
  elite_idx = np.argsort(returns)[-elite_count:]
  first_actions = sequences[elite_idx, 0]
  counts = np.bincount(first_actions, minlength=ACTION_SIZE).astype(np.float32)
  probs = counts / np.sum(counts)
  return int(np.argmax(probs)), float(np.mean(returns[elite_idx]))


def _as_rgb(frame):
  frame = np.asarray(frame, dtype=np.uint8)
  if frame.ndim != 3:
    raise ValueError("rendered frame must be an RGB array")
  if frame.shape[2] == 4:
    frame = frame[:, :, :3]
  return frame


def _stack_frames(frames):
  if not frames:
    raise RuntimeError("no frames were generated for W&B video")
  return np.stack(frames, axis=0).astype(np.uint8)


def _copy_tabular_obs(obs):
  return {
    "grid": np.asarray(obs["grid"], dtype=np.int8).copy(),
    "self": np.asarray(obs["self"], dtype=np.int16).copy(),
  }


def _encode_tabular(controller, obs):
  vec = tabular_to_vector(obs).reshape(1, -1)
  mu, _ = controller.vae.encode_mu_logvar(vec)
  return mu[0]


def _action_name(action):
  action = int(action)
  return ACTION_NAMES[action] if 0 <= action < len(ACTION_NAMES) else str(action)


def _overlay(step, action, reward, cumulative_reward, done):
  return {
    "step": int(step),
    "action": action,
    "reward": float(reward),
    "cumulative_reward": float(cumulative_reward),
    "done": bool(done),
  }
