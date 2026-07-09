from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vGridcraft"))

from vgridcraft.config import VGridcraftConfig
from vgridcraft.env import (
    ACTION_NAMES,
    BLOCK_EMPTY,
    BLOCK_STONE,
    BLOCK_TREE,
    EVENT_INDEX,
    EVENT_NAMES,
    ITEM_PLANK,
    ITEM_STICK,
    ITEM_STONE,
    ITEM_STONE_SWORD,
    ITEM_WOOD,
    ITEM_WOOD_PICKAXE,
    ITEM_WOOD_SWORD,
    REWARD_COMPONENT_NAMES,
    TERRAIN_GRASS,
    VectorizedGridcraftEnv,
)


POLICY_LEVELS = {
    "idle": 0,
    "explore": 1,
    "harvest_wood": 2,
    "craft_plank": 3,
    "craft_stick": 4,
    "craft_wood_tool": 5,
    "harvest_stone": 6,
    "craft_stone_tool": 7,
    "armed_hunter": 8,
}

ACTION_ID = {name: index for index, name in enumerate(ACTION_NAMES)}


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose Gridcraft reward/task hierarchy.")
    parser.add_argument("--protocol", choices=("controlled", "natural", "all"), default="all")
    parser.add_argument("--policies", nargs="*", default=list(POLICY_LEVELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--controlled-steps", type=int, default=8)
    parser.add_argument("--natural-steps", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default="reward_hierarchy_diagnosis")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="ns-mawm-gridcraft")
    parser.add_argument("--wandb-name", default="gridcraft-reward-hierarchy-diagnosis")
    parser.add_argument(
        "--policy-checkpoints",
        nargs="*",
        default=[],
        help="Optional BenchMARL checkpoint files (for example final B00/B10 checkpoints).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    unknown = sorted(set(args.policies) - set(POLICY_LEVELS))
    if unknown:
        raise SystemExit(f"Unknown policies: {', '.join(unknown)}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            job_type="reward_hierarchy_diagnosis",
            config=vars(args),
        )

    rows = []
    summaries = []
    global_step = 0
    protocols = ("controlled", "natural") if args.protocol == "all" else (args.protocol,)
    for protocol in protocols:
        for policy in args.policies:
            for seed in args.seeds:
                episode_rows, summary = run_episode(
                    policy=policy,
                    protocol=protocol,
                    seed=seed,
                    num_agents=args.num_agents,
                    max_steps=args.controlled_steps if protocol == "controlled" else args.natural_steps,
                    device=args.device,
                )
                rows.extend(episode_rows)
                summaries.append(summary)
                if wandb_run is not None:
                    for row in episode_rows:
                        wandb_run.log(
                            {
                                "Reward hierarchy diagnosis/cumulative_reward": row["cumulative_reward"],
                                "Reward hierarchy diagnosis/cumulative_complexity": row["cumulative_complexity"],
                                "Reward hierarchy diagnosis/cumulative_exponential_complexity": row[
                                    "cumulative_exponential_complexity"
                                ],
                                "Reward hierarchy diagnosis/task_level_max": row["task_level_max"],
                                "Reward hierarchy diagnosis/policy_level": row["expected_level"],
                                "Reward hierarchy diagnosis/protocol": protocol,
                                "Reward hierarchy diagnosis/policy": policy,
                                "Reward hierarchy diagnosis/seed": seed,
                            },
                            step=global_step,
                        )
                        global_step += 1

    for checkpoint in args.policy_checkpoints:
        checkpoint_rows, checkpoint_summaries = evaluate_learned_checkpoint(
            checkpoint=checkpoint,
            seeds=args.seeds,
            num_agents=args.num_agents,
            max_steps=args.natural_steps,
            device=args.device,
        )
        rows.extend(checkpoint_rows)
        summaries.extend(checkpoint_summaries)
        if wandb_run is not None:
            for row in checkpoint_rows:
                wandb_run.log(
                    {
                        "Reward hierarchy diagnosis/cumulative_reward": row["cumulative_reward"],
                        "Reward hierarchy diagnosis/cumulative_complexity": row[
                            "cumulative_complexity"
                        ],
                        "Reward hierarchy diagnosis/cumulative_exponential_complexity": row[
                            "cumulative_exponential_complexity"
                        ],
                        "Reward hierarchy diagnosis/task_level_max": row["task_level_max"],
                        "Reward hierarchy diagnosis/policy_level": -1,
                        "Reward hierarchy diagnosis/protocol": "learned",
                        "Reward hierarchy diagnosis/policy": row["policy"],
                        "Reward hierarchy diagnosis/seed": row["seed"],
                    },
                    step=global_step,
                )
                global_step += 1

    analysis = analyse(summaries)
    write_outputs(out_dir, rows, summaries, analysis)
    if wandb_run is not None:
        import wandb

        wandb_run.log(
            {
                "Reward hierarchy diagnosis/reward_complexity_spearman": analysis[
                    "reward_complexity_spearman"
                ],
                "Reward hierarchy diagnosis/summary": wandb.Table(
                    columns=list(summaries[0]) if summaries else [],
                    data=[list(row.values()) for row in summaries],
                ),
                "Reward hierarchy diagnosis/policy_event_matrix": event_matrix_table(
                    wandb, summaries
                ),
                "Reward hierarchy diagnosis/controlled_reward_curves": wandb.plot.line_series(
                    **line_series_args(rows, "cumulative_reward", "Cumulative reward")
                ),
                "Reward hierarchy diagnosis/controlled_complexity_curves": wandb.plot.line_series(
                    **line_series_args(
                        rows, "cumulative_exponential_complexity", "Cumulative complexity"
                    )
                ),
            },
            step=global_step,
        )
        wandb_run.finish()
    print(json.dumps(analysis, indent=2))
    print(f"Wrote diagnostic outputs to {out_dir.resolve()}")


def run_episode(policy, protocol, seed, num_agents, max_steps, device):
    config = VGridcraftConfig(
        width=16,
        height=16,
        num_agents=num_agents,
        max_steps=max_steps,
        seed=seed,
        mob_spawn_rate=0 if protocol == "controlled" else 10,
        mob_move_prob=0.0 if protocol == "controlled" else 0.8,
        tree_apple_drop_chance=0.0 if protocol == "controlled" else 0.5,
    )
    env = VectorizedGridcraftEnv(
        num_envs=1, num_agents=num_agents, device=device, seed=seed, config=config
    )
    if protocol == "controlled":
        setup_controlled(env, policy)
    cumulative_reward = 0.0
    action_totals = torch.zeros(len(ACTION_NAMES))
    event_totals = torch.zeros(len(EVENT_NAMES))
    component_totals = torch.zeros(len(REWARD_COMPONENT_NAMES))
    rows = []
    for step in range(max_steps):
        actions = (
            controlled_actions(policy, num_agents, step, env.device)
            if protocol == "controlled"
            else natural_actions(env, POLICY_LEVELS[policy], step)
        )
        _, reward, done, truncated, info = env.step(actions)
        error = float(info["reward_decomposition_error"].abs().max().detach().cpu())
        if error > 1e-4:
            raise RuntimeError(f"reward decomposition mismatch: {error}")
        step_reward = float(reward.sum().detach().cpu())
        cumulative_reward += step_reward
        step_events = info["event_success"][0].sum(dim=0).detach().cpu()
        action_totals += info["action_attempts"][0].sum(dim=0).detach().cpu()
        event_totals += step_events
        component_totals += info["reward_components"][0].sum(dim=0).detach().cpu()
        successful = [
            f"{EVENT_NAMES[index]}:{value:g}"
            for index, value in enumerate(step_events.tolist())
            if value > 0
        ]
        rows.append(
            {
                "protocol": protocol,
                "policy": policy,
                "seed": seed,
                "step": step,
                "expected_level": POLICY_LEVELS[policy],
                "reward": step_reward,
                "cumulative_reward": cumulative_reward,
                "successful_event": ",".join(successful),
                "cumulative_complexity": float(info["complexity_cumulative"].sum().cpu()),
                "cumulative_exponential_complexity": float(
                    info["complexity_exponential_cumulative"].sum().cpu()
                ),
                "complexity_unique": float(info["complexity_unique"].sum().cpu()),
                "task_level_max": int(info["task_level_max"].max().cpu()),
                "inventory": json.dumps(env.inventory[0].detach().cpu().tolist()),
                "health": json.dumps(env.hp[0].detach().cpu().tolist()),
                "hunger": json.dumps(env.hunger[0].detach().cpu().tolist()),
            }
        )
        if bool((done | truncated)[0]):
            break
    env.close()
    summary = {
        "protocol": protocol,
        "policy": policy,
        "seed": seed,
        "expected_level": POLICY_LEVELS[policy],
        "observed_level": max(row["task_level_max"] for row in rows),
        "cumulative_reward": cumulative_reward,
        "cumulative_complexity": rows[-1]["cumulative_complexity"],
        "cumulative_exponential_complexity": rows[-1]["cumulative_exponential_complexity"],
        "complexity_unique": rows[-1]["complexity_unique"],
        **{f"event_{name}": float(event_totals[index]) for index, name in enumerate(EVENT_NAMES)},
        **{
            f"attempt_{name}": float(action_totals[index])
            for index, name in enumerate(ACTION_NAMES)
        },
        **{
            f"reward_{name}": float(component_totals[index])
            for index, name in enumerate(REWARD_COMPONENT_NAMES)
        },
    }
    kill_reward = summary["reward_attack_hit"] + summary["reward_mob_kill"]
    summary["kill_reward_fraction"] = kill_reward / max(abs(cumulative_reward), 1e-8)
    return rows, summary


def setup_controlled(env, policy):
    env.terrain[:] = TERRAIN_GRASS
    env.blocks[:] = BLOCK_EMPTY
    env.mob_alive[:] = False
    env.item_alive[:] = False
    env.inventory[:] = 0
    env.equipped[:] = -1
    env.hp[:] = env.config.hp_max
    env.hunger[:] = env.config.hunger_max
    y = 5
    for agent in range(env.num_agents):
        x = 2 + agent * 4
        env.agent_x[0, agent] = x
        env.agent_y[0, agent] = y
        env.visited[0, agent] = False
        env.visited[0, agent, y, x] = True
        if policy == "harvest_wood":
            env.blocks[0, y, x + 1] = BLOCK_TREE
        elif policy == "craft_plank":
            env.inventory[0, agent, ITEM_WOOD] = 1
        elif policy == "craft_stick":
            env.inventory[0, agent, ITEM_PLANK] = 2
        elif policy == "craft_wood_tool":
            env.inventory[0, agent, ITEM_STICK] = 1
            env.inventory[0, agent, ITEM_PLANK] = 1
        elif policy == "harvest_stone":
            env.inventory[0, agent, ITEM_WOOD_PICKAXE] = 1
            env.equipped[0, agent] = ITEM_WOOD_PICKAXE
            env.blocks[0, y, x + 1] = BLOCK_STONE
        elif policy == "craft_stone_tool":
            env.inventory[0, agent, ITEM_STICK] = 1
            env.inventory[0, agent, ITEM_STONE] = 1
        elif policy == "armed_hunter":
            env.inventory[0, agent, ITEM_WOOD_SWORD] = 1
            env.equipped[0, agent] = ITEM_WOOD_SWORD
            env.mob_alive[0, agent] = True
            env.mob_hp[0, agent] = 3
            env.mob_x[0, agent] = x + 1
            env.mob_y[0, agent] = y


def controlled_actions(policy, num_agents, step, device):
    action_name = {
        "idle": "stay",
        "explore": "move_e",
        "harvest_wood": "harvest",
        "craft_plank": "craft_plank",
        "craft_stick": "craft_stick",
        "craft_wood_tool": "craft_wood_sword",
        "harvest_stone": "harvest",
        "craft_stone_tool": "craft_stone_sword",
        "armed_hunter": "attack",
    }[policy]
    action = ACTION_ID[action_name] if step == 0 else ACTION_ID["stay"]
    return torch.full((1, num_agents), action, dtype=torch.long, device=device)


def natural_actions(env, target_level, step):
    actions = torch.zeros((1, env.num_agents), dtype=torch.long, device=env.device)
    for agent in range(env.num_agents):
        actions[0, agent] = natural_action_for_agent(env, agent, target_level, step)
    return actions


def natural_action_for_agent(env, agent, target_level, step):
    inv = env.inventory[0, agent]
    if target_level == 0:
        return ACTION_ID["stay"]
    if int(env.task_level_max[0, agent]) >= target_level:
        return ACTION_ID["stay"]
    if target_level == 1:
        return ACTION_ID[("move_n", "move_e", "move_s", "move_w")[step % 4]]
    if target_level >= 8 and inv[ITEM_WOOD_SWORD] > 0:
        mob = nearest_position(
            env, agent, env.mob_x[0][env.mob_alive[0]], env.mob_y[0][env.mob_alive[0]]
        )
        if mob is not None:
            return interaction_or_move(env, agent, mob, "attack")
    if target_level >= 7 and inv[ITEM_STONE] > 0 and inv[ITEM_STICK] > 0:
        return ACTION_ID["craft_stone_sword"]
    if target_level >= 6 and env.equipped[0, agent] in (ITEM_WOOD_PICKAXE,):
        stone_y, stone_x = torch.nonzero(env.blocks[0] == BLOCK_STONE, as_tuple=True)
        target = nearest_position(env, agent, stone_x, stone_y)
        if target is not None:
            return interaction_or_move(env, agent, target, "harvest")
    if target_level >= 5 and inv[ITEM_STICK] > 0 and inv[ITEM_PLANK] > 0:
        tool = "craft_wood_sword" if target_level >= 8 else "craft_wood_pickaxe"
        return ACTION_ID[tool]
    if target_level >= 4 and inv[ITEM_STICK] == 0 and inv[ITEM_PLANK] >= 2:
        return ACTION_ID["craft_stick"]
    needs_plank = target_level >= 3 and (
        inv[ITEM_PLANK] == 0 or (target_level >= 5 and inv[ITEM_STICK] == 0)
    )
    if needs_plank and inv[ITEM_WOOD] > 0:
        return ACTION_ID["craft_plank"]
    tree_y, tree_x = torch.nonzero(env.blocks[0] == BLOCK_TREE, as_tuple=True)
    target = nearest_position(env, agent, tree_x, tree_y)
    if target is not None:
        return interaction_or_move(env, agent, target, "harvest")
    return ACTION_ID[("move_n", "move_e", "move_s", "move_w")[step % 4]]


def nearest_position(env, agent, xs, ys):
    if len(xs) == 0:
        return None
    ax = env.agent_x[0, agent]
    ay = env.agent_y[0, agent]
    distances = (xs - ax).abs() + (ys - ay).abs()
    index = int(distances.argmin())
    return int(xs[index]), int(ys[index])


def interaction_or_move(env, agent, target, interaction):
    ax = int(env.agent_x[0, agent])
    ay = int(env.agent_y[0, agent])
    tx, ty = target
    if abs(tx - ax) + abs(ty - ay) == 1:
        return ACTION_ID[interaction]
    candidates = []
    if tx > ax:
        candidates.append(("move_e", ax + 1, ay))
    if tx < ax:
        candidates.append(("move_w", ax - 1, ay))
    if ty > ay:
        candidates.append(("move_s", ax, ay + 1))
    if ty < ay:
        candidates.append(("move_n", ax, ay - 1))
    for name, x, y in candidates:
        tx_tensor = torch.tensor([x], device=env.device)
        ty_tensor = torch.tensor([y], device=env.device)
        if bool(env.is_walkable(tx_tensor, ty_tensor)[0]):
            return ACTION_ID[name]
    return ACTION_ID["stay"]


def analyse(summaries):
    controlled = [row for row in summaries if row["protocol"] == "controlled"]
    reward = [row["cumulative_reward"] for row in controlled]
    complexity = [row["complexity_unique"] for row in controlled]
    expected = [row["expected_level"] for row in controlled]
    level_failures = [
        {
            "policy": row["policy"],
            "seed": row["seed"],
            "expected": row["expected_level"],
            "observed": row["observed_level"],
        }
        for row in controlled
        if row["expected_level"] != row["observed_level"]
    ]
    return {
        "reward_complexity_spearman": spearman(reward, complexity),
        "reward_expected_level_spearman": spearman(reward, expected),
        "controlled_level_failures": level_failures,
        "controlled_reward_monotonic": monotonic_policy_means(controlled, "cumulative_reward"),
        "controlled_complexity_monotonic": monotonic_policy_means(controlled, "complexity_unique"),
        "unarmed_kills": sum(row["event_mob_kill_unarmed"] for row in summaries),
        "armed_kills": sum(row["event_mob_kill_armed"] for row in summaries),
    }


def evaluate_learned_checkpoint(checkpoint, seeds, num_agents, max_steps, device):
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    sys.path.insert(0, str(ROOT / "BenchMARL"))
    from benchmarl.experiment import Experiment
    from vgridcraft.torchrl_env import GridcraftTorchRLEnv

    experiment = Experiment.reload_from_file(str(checkpoint))
    policy = experiment.policy
    label = checkpoint.parent.parent.name
    all_rows = []
    summaries = []
    try:
        for seed in seeds:
            config = VGridcraftConfig(
                num_agents=num_agents,
                max_steps=max_steps,
                seed=seed,
            )
            env = GridcraftTorchRLEnv(
                num_envs=1, device=device, seed=seed, config=config
            )
            rollout = env.rollout(
                max_steps=max_steps,
                policy=policy,
                auto_reset=True,
                break_when_any_done=False,
            )
            rows, summary = rows_from_learned_rollout(rollout, label, seed)
            all_rows.extend(rows)
            summaries.append(summary)
            env.close()
    finally:
        experiment.close()
    return all_rows, summaries


def rows_from_learned_rollout(rollout, label, seed):
    rewards = rollout.get(("next", "agents", "reward")).detach().cpu()
    attempts = rollout.get(("next", "agents", "action_attempts")).detach().cpu()
    events = rollout.get(("next", "agents", "event_success")).detach().cpu()
    components = rollout.get(("next", "agents", "reward_components")).detach().cpu()
    levels = rollout.get(("next", "agents", "task_level_max")).detach().cpu()
    complexity = rollout.get(("next", "agents", "complexity_cumulative")).detach().cpu()
    complexity_exp = rollout.get(
        ("next", "agents", "complexity_exponential_cumulative")
    ).detach().cpu()
    complexity_unique = rollout.get(("next", "agents", "complexity_unique")).detach().cpu()
    rewards = rewards.reshape(-1, rewards.shape[-2], rewards.shape[-1])
    attempts = attempts.reshape(-1, attempts.shape[-2], attempts.shape[-1])
    events = events.reshape(-1, events.shape[-2], events.shape[-1])
    components = components.reshape(-1, components.shape[-2], components.shape[-1])
    levels = levels.reshape(-1, levels.shape[-2], levels.shape[-1])
    complexity = complexity.reshape(-1, complexity.shape[-2], complexity.shape[-1])
    complexity_exp = complexity_exp.reshape(
        -1, complexity_exp.shape[-2], complexity_exp.shape[-1]
    )
    complexity_unique = complexity_unique.reshape(
        -1, complexity_unique.shape[-2], complexity_unique.shape[-1]
    )
    cumulative_reward = 0.0
    rows = []
    for step in range(rewards.shape[0]):
        step_reward = float(rewards[step].sum())
        cumulative_reward += step_reward
        step_events = events[step].sum(dim=0)
        successful = [
            f"{EVENT_NAMES[index]}:{value:g}"
            for index, value in enumerate(step_events.tolist())
            if value > 0
        ]
        rows.append(
            {
                "protocol": "learned",
                "policy": label,
                "seed": seed,
                "step": step,
                "expected_level": -1,
                "reward": step_reward,
                "cumulative_reward": cumulative_reward,
                "successful_event": ",".join(successful),
                "cumulative_complexity": float(complexity[step].sum()),
                "cumulative_exponential_complexity": float(complexity_exp[step].sum()),
                "complexity_unique": float(complexity_unique[step].sum()),
                "task_level_max": int(levels[step].max()),
                "inventory": "",
                "health": "",
                "hunger": "",
            }
        )
    event_totals = events.sum(dim=(0, 1))
    action_totals = attempts.sum(dim=(0, 1))
    component_totals = components.sum(dim=(0, 1))
    summary = {
        "protocol": "learned",
        "policy": label,
        "seed": seed,
        "expected_level": -1,
        "observed_level": int(levels.max()),
        "cumulative_reward": cumulative_reward,
        "cumulative_complexity": float(complexity[-1].sum()),
        "cumulative_exponential_complexity": float(complexity_exp[-1].sum()),
        "complexity_unique": float(complexity_unique[-1].sum()),
        **{f"event_{name}": float(event_totals[index]) for index, name in enumerate(EVENT_NAMES)},
        **{
            f"attempt_{name}": float(action_totals[index])
            for index, name in enumerate(ACTION_NAMES)
        },
        **{
            f"reward_{name}": float(component_totals[index])
            for index, name in enumerate(REWARD_COMPONENT_NAMES)
        },
    }
    kill_reward = summary["reward_attack_hit"] + summary["reward_mob_kill"]
    summary["kill_reward_fraction"] = kill_reward / max(abs(cumulative_reward), 1e-8)
    return rows, summary


def monotonic_policy_means(rows, key):
    means = []
    for policy, level in sorted(POLICY_LEVELS.items(), key=lambda item: item[1]):
        values = [row[key] for row in rows if row["policy"] == policy]
        if values:
            means.append(sum(values) / len(values))
    return all(left < right for left, right in zip(means, means[1:]))


def spearman(xs, ys):
    if len(xs) < 2 or len(xs) != len(ys):
        return float("nan")
    rx = ranks(xs)
    ry = ranks(ys)
    mean_x = sum(rx) / len(rx)
    mean_y = sum(ry) / len(ry)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(rx, ry))
    denominator = math.sqrt(
        sum((x - mean_x) ** 2 for x in rx) * sum((y - mean_y) ** 2 for y in ry)
    )
    return numerator / denominator if denominator else 0.0


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + end - 1) / 2 + 1
        for index in order[cursor:end]:
            result[index] = rank
        cursor = end
    return result


def write_outputs(out_dir, rows, summaries, analysis):
    (out_dir / "step_traces.json").write_text(json.dumps(rows, indent=2))
    (out_dir / "policy_summary.json").write_text(json.dumps(summaries, indent=2))
    (out_dir / "analysis.json").write_text(json.dumps(analysis, indent=2))
    for name, data in (("step_traces.csv", rows), ("policy_summary.csv", summaries)):
        if not data:
            continue
        with (out_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    write_local_curves(out_dir, rows)


def event_matrix_table(wandb, summaries):
    columns = ["protocol", "policy", "seed", *EVENT_NAMES]
    data = []
    for row in summaries:
        data.append(
            [row["protocol"], row["policy"], row["seed"]]
            + [row[f"event_{name}"] for name in EVENT_NAMES]
        )
    return wandb.Table(columns=columns, data=data)


def line_series_args(rows, key, title):
    controlled = [row for row in rows if row["protocol"] == "controlled"]
    max_step = max((row["step"] for row in controlled), default=0)
    xs = list(range(max_step + 1))
    ys = []
    keys = []
    for policy in POLICY_LEVELS:
        curve = []
        for step in xs:
            values = [
                row[key]
                for row in controlled
                if row["policy"] == policy and row["step"] == step
            ]
            curve.append(sum(values) / len(values) if values else float("nan"))
        ys.append(curve)
        keys.append(policy)
    return {
        "xs": xs,
        "ys": ys,
        "keys": keys,
        "title": title,
        "xname": "timestep",
    }


def write_local_curves(out_dir, rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    controlled = [row for row in rows if row["protocol"] == "controlled"]
    if not controlled:
        return
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    for policy in POLICY_LEVELS:
        policy_rows = [row for row in controlled if row["policy"] == policy]
        steps = sorted({row["step"] for row in policy_rows})
        reward = []
        complexity = []
        for step in steps:
            step_rows = [row for row in policy_rows if row["step"] == step]
            reward.append(sum(row["cumulative_reward"] for row in step_rows) / len(step_rows))
            complexity.append(
                sum(row["cumulative_exponential_complexity"] for row in step_rows)
                / len(step_rows)
            )
        axes[0].plot(steps, reward, label=policy)
        axes[1].plot(steps, complexity, label=policy)
    axes[0].set_title("Cumulative reward")
    axes[1].set_title("Cumulative task complexity")
    for axis in axes:
        axis.set_xlabel("Timestep")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Reward")
    axes[1].set_ylabel("Complexity")
    axes[1].legend(fontsize=7, loc="upper left")
    figure.tight_layout()
    figure.savefig(out_dir / "reward_vs_complexity.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
