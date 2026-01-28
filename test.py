from co_mas.test import sample_action
from smac_pettingzoo import smacv2_pettingzoo_v1, smacv1_pettingzoo_v1
from loguru import logger

env = smacv1_pettingzoo_v1.parallel_env("8m")
obs, info = env.reset(seed=42)
step = 0
while True:
    obs, _, terminated, truncated, info = env.step(
        {agent: sample_action(env, obs, agent, info) for agent in env.agents}
    )
    step += 1
    if len(env.agents) <= 0:
        logger.debug(
            f"step {step}, terminated: {terminated} truncated: {truncated}")
        break
env.close()

env = smacv2_pettingzoo_v1.parallel_env("10gen_terran_10_vs_10")
obs, info = env.reset(seed=42)
step = 0
while True:
    obs, _, terminated, truncated, info = env.step(
        {agent: sample_action(env, obs, agent, info) for agent in env.agents}
    )
    step += 1
    if len(env.agents) <= 0:
        logger.debug(
            f"step {step}, terminated: {terminated} truncated: {truncated}")
        break
env.close()
