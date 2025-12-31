from functools import partial
from maic.envs.lbforaging.foraging_wrapper import ForagingEnvWrapper
from maic.envs.multiagentenvwrapper import MultiAgentEnvWrapper
from maic.envs.qplex_smac.smac.env.lbforaging import register_envs as register_foraging


def env_fn(env, **kwargs) -> MultiAgentEnvWrapper:
    return env(**kwargs)

REGISTRY = {
    "foraging": partial(env_fn, env=ForagingEnvWrapper),
}

register_foraging()
