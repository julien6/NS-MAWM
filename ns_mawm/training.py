from __future__ import annotations

from dataclasses import asdict
import json
import random
import time
from collections import OrderedDict
from functools import wraps
import numpy as np
import torch
from torch.nn import functional as F
from .models import BACKBONES, WorldModel, observation_loss
from .rules import Context, PSWM
from .diagnostics import Diagnostics, error_split, union_blocks
from .statistics import threshold_crossing


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def episode_contexts(episode, schema, engine):
    history, memory, previous = [], {}, None
    output = []
    for t, action in enumerate(episode.actions):
        c = Context(schema.decode(episode.observations[t]), tuple(action.tolist()), tuple(history), previous, memory, "real", t)
        symbolic = engine(c)
        output.append((c, symbolic))
        memory, previous = symbolic.memory, tuple(action.tolist())
        history = (history + [c.observation])[-50:]
    return output


def build_network(schema, action_size, config, control=False):
    wm = config["world_model"]
    if wm.get("backbone") in ("pswm_only", "llm_code"):
        return None
    network = BACKBONES[wm.get("backbone", "jopm_lstm")](schema, action_size,
        **wm.get("sizes", {}), residual=wm.get("strategy") == "residual", control=control)
    if control and not callable(getattr(network, "heads", None)):
        raise TypeError("Control requires the separate reward/termination heads capability")
    network.control = control
    return network


def isolated_rng(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            return function(*args, **kwargs)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
    return wrapped


@isolated_rng
def train(dataset, schema, library, action_size, config, run=None, *, control=False, network=None):
    if getattr(dataset, "frozen", None) is not None:
        raise PermissionError("Training after test protocol freeze is disabled")
    wm = config["world_model"]
    optim = wm.get("optim", {})
    seed = config["run"].get("seed", 0)
    seed_all(seed)
    strategy = wm.get("strategy", "none")
    network = network if network is not None else build_network(schema, action_size, config, control)
    if network is None:
        return None, {"updates": 0, "epochs": 0, "dataset_passes": 0, "seconds": 0}
    device = config.get("device", "cpu")
    network.to(device).train()
    optimizer = torch.optim.Adam(network.parameters(), lr=float(optim.get("lr", 3e-4)))
    episodes = dataset.get("train", purpose="train")
    engine = PSWM(library)
    cached = OrderedDict()
    rng = np.random.default_rng(seed + 10000)
    updates, batch = int(optim.get("updates", 20000)), int(optim.get("batch_seqs", 32))
    length, burn = int(optim.get("seq_len", 50)), int(optim.get("burn_in", 10))
    if min(updates, batch, length) < 1 or burn < 0:
        raise ValueError("Invalid optimizer budget")
    total_data = sum(len(e.actions) for e in episodes)
    processed, durations, validations = 0, [], []
    start_time = time.perf_counter()
    for update in range(1, updates + 1):
        start_time_update = time.perf_counter()
        optimizer.zero_grad()
        total_loss = 0.
        # Variable episode lengths are handled explicitly; sequences never cross
        # reset boundaries. One update means one optimizer.step, not one sequence.
        for _ in range(batch):
            index = int(rng.integers(len(episodes)))
            episode = episodes[index]
            if index not in cached:
                cached[index] = episode_contexts(episode, schema, engine)
                while len(cached) > int(optim.get("symbolic_cache_episodes", 4)):
                    cached.popitem(last=False)
            cached.move_to_end(index)
            start = int(rng.integers(len(episode.actions)))
            state = network.init_state()
            for t in range(max(0, start - burn), start):
                s = cached[index][t][1]
                with torch.no_grad():
                    _, state = network.step(state, episode.observations[t:t+1].to(device), episode.actions[t:t+1].to(device),
                        (s.mask[None].to(device), s.target[None].to(device)))
            loss = torch.zeros((), device=device)
            end = min(len(episode.actions), start + length)
            for t in range(start, end):
                s = cached[index][t][1]
                logits, state = network.step(state, episode.observations[t:t+1].to(device), episode.actions[t:t+1].to(device),
                    (s.mask[None].to(device), s.target[None].to(device)))
                loss = loss + observation_loss(schema, logits, episode.observations[t+1:t+2].to(device),
                    (s.mask[None].to(device), s.target[None].to(device)), strategy, wm.get("lambda", 1.))
                if getattr(network, "auxiliary_loss", None) is not None:
                    loss = loss + wm.get("latent_kl_weight", .1) * network.auxiliary_loss
                if control:
                    reward, done = network.heads(state)
                    loss = loss + F.mse_loss(reward, episode.rewards[t:t+1].to(device)) + F.binary_cross_entropy_with_logits(done, episode.terminated[t:t+1].float().to(device))
            loss = loss / (end - start) / batch
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite world-model training loss")
            loss.backward()
            total_loss += float(loss.detach())
            processed += end - start
        torch.nn.utils.clip_grad_norm_(network.parameters(), 10.)
        optimizer.step()
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        duration = time.perf_counter() - start_time_update
        if update > 1:
            durations.append(duration)
        if run:
            run.record("train_loss", total_loss, update)
            run.record("seconds_per_update", duration, update, unit="seconds")
        interval = int(optim.get("checkpoint_interval", updates))
        if update % interval == 0 or update == updates:
            if run:
                torch.save({"network": network.state_dict(), "optimizer": optimizer.state_dict(), "update": update,
                    "schema": schema.to_dict(), "config": config, "library_hash": library.hash, "split_hash": dataset.hash,
                    "action_size": action_size, "control": control}, run.path / f"update-{update}.pt")
            if config.get("evaluation", {}).get("validation_target") is not None:
                score = evaluate(dataset, schema, library, network, config, split="select")["metrics"]["mse"]
                validations.append((update, score, time.perf_counter() - start_time))
                network.train()
    timing = {"updates": updates, "epochs": 0, "dataset_passes": processed / total_data,
              "sampling": "random_sequences_with_replacement", "seconds": time.perf_counter() - start_time,
              "seconds_per_update_after_warmup": float(np.mean(durations)) if durations else None}
    target = config.get("evaluation", {}).get("validation_target")
    if target is not None:
        crossing = threshold_crossing([v[0] for v in validations], [v[1] for v in validations], target)
        timing["updates_to_target"] = crossing
        timing["seconds_to_target"] = next((v[2] for v in validations if v[0] == crossing), None)
        timing["attained"] = crossing is not None
    return network.eval(), timing


@isolated_rng
def evaluate(dataset, schema, library, network, config, *, split="select", diagnostic_library=None, compared_libraries=()):
    if split == "test" and dataset.frozen != {"library_hash": library.hash, "configuration_hash": __import__('ns_mawm.rules', fromlist=['digest']).digest(config)}:
        raise PermissionError("Evaluation differs from the frozen test protocol")
    seed_all(config["run"].get("seed", 0) + 30000)
    episodes = dataset.get(split, purpose="evaluate")
    reference = PSWM(diagnostic_library or library)
    wm = config["world_model"]
    strategy = wm.get("strategy", "none")
    if wm.get("backbone") in ("pswm_only", "llm_code"):
        strategy = wm["backbone"]
    model = WorldModel(network, PSWM(library), strategy)
    if network is not None:
        network.eval()
    diagnostic = Diagnostics(reference, enforced_hash=library.hash, strategy=strategy)
    horizon = int(config.get("evaluation", {}).get("horizon", 25))
    burn = int(wm.get("optim", {}).get("burn_in", 10))
    reference_contexts = [episode_contexts(e, schema, reference) for e in episodes]
    fixed = set()
    for episode, contexts in zip(episodes, reference_contexts):
        for c, output in contexts:
            fixed.update(output.provenance)
        for lib in compared_libraries:
            fixed.update(union_blocks([s for _, s in episode_contexts(episode, schema, PSWM(lib))]))
    union_path = config.get("evaluation", {}).get("fixed_union")
    if union_path:
        from .rules import digest
        record = json.loads(__import__('pathlib').Path(union_path).read_text())
        claimed = record.pop("hash")
        if digest(record) != claimed or record["split_hash"] != dataset.hash or config.get("evaluation",{}).get("fixed_union_hash",claimed) != claimed:
            raise ValueError("Fixed union provenance mismatch")
        expected = sorted({library.hash, *[lib.hash for lib in compared_libraries]})
        if record["libraries"] != expected:
            raise ValueError("Fixed union library set mismatch")
        fixed = set(record["blocks"])
        if not fixed <= set(schema.by_name):
            raise ValueError("Fixed union contains unknown blocks")
    imagined_diagnostics = {}
    splits, curves, inference = [], {}, []
    for episode, contexts in zip(episodes, reference_contexts):
        state = network.init_state() if network is not None else None
        enforcement = episode_contexts(episode, schema, model.engine)
        for t, (c, reference_output) in enumerate(contexts):
            start = time.perf_counter()
            p = model.step(state, enforcement[t][0], diagnostics=True)
            state = p.state
            if t:
                inference.append(time.perf_counter() - start)
            diagnostic.add(c, p.output, episode.observations[t+1], pre=p.pre if strategy == "projection" else None,
                           split=split, evidence_id=f"{episode.id}:{t}")
        stride = int(config.get("evaluation", {}).get("stride", horizon))
        for start in range(0, len(episode.actions), stride):
            prefix = [(episode.observations[t], episode.actions[t].tolist()) for t in range(max(0, start-burn), start)]
            predictions = model.rollout(prefix, episode.observations[start], episode.actions[start:].tolist(), horizon=horizon)
            imagined_memory, imagined_history, previous_action = {}, [], None
            for t, (prefix_obs, prefix_action) in enumerate(prefix):
                pc = Context(schema.decode(prefix_obs), tuple(prefix_action), tuple(imagined_history), previous_action, imagined_memory, "real", t)
                po = reference(pc)
                imagined_memory, previous_action = po.memory, tuple(prefix_action)
                imagined_history.append(pc.observation)
            imagined_observation = schema.decode(episode.observations[start])
            for step, p in enumerate(predictions):
                target = episode.observations[start + step + 1]
                if config.get("evaluation", {}).get("diagnostics_mode") == "both":
                    ic = Context(imagined_observation, tuple(episode.actions[start+step].tolist()), tuple(imagined_history[-50:]), previous_action, imagined_memory, "imagined", len(prefix)+step)
                    diag = imagined_diagnostics.setdefault(step+1, Diagnostics(reference, enforced_hash=library.hash, strategy=strategy))
                    io = diag.add(ic, p.output, target, pre=p.pre if strategy == "projection" else None, split=split, evidence_id=f"{episode.id}:{start+step}")
                    imagined_memory, previous_action = io.memory, ic.joint_action
                    imagined_history.append(imagined_observation)
                    imagined_observation = schema.decode(p.output)
                reference_output = contexts[start + step][1]
                errors = error_split(schema, p.output, target, reference_output.mask, fixed)
                per_block = schema.losses(p.output, target, training=False)
                other = torch.tensor([b.owner != 0 for b in schema.blocks])
                errors["other_shared_error"] = float(per_block[other].mean()) if other.any() else None
                splits.append(errors)
                curves.setdefault(step + 1, []).append(errors["mse"])
    metrics = {name: float(np.mean([s[name] for s in splits if s[name] is not None])) if any(s[name] is not None for s in splits) else None
               for name in splits[0]} if splits else {}
    metrics["inference_seconds_per_joint_step"] = float(np.mean(inference)) if inference else None
    return {"metrics": metrics, "per_step_error": {k: float(np.mean(v)) for k, v in curves.items()},
            "diagnostics": diagnostic.report(), "fixed_blocks": sorted(fixed), "split": split,
            "horizon": horizon, "rollout_steps": len(splits),
            "imagined_diagnostics": {k: {**v.report(), "context_origin": "imagined", "rdd_reference": "recorded trajectory; not a counterfactual simulator label"} for k,v in imagined_diagnostics.items()}}
