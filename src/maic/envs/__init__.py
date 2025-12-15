from functools import partial

# SMAC stuff
# from maic.smac.env import MultiAgentEnv, StarCraft2Env
# from .join1 import Join1Env
from maic.envs.multiagentenv import MultiAgentEnv
from maic.envs.lbforaging import ForagingEnv

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {
    # "sc2": partial(env_fn, env=StarCraft2Env),
    # "join1": partial(env_fn, env=Join1Env),
    "foraging": partial(env_fn, env=ForagingEnv),
}
