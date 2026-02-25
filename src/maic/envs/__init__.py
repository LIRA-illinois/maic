from functools import partial
from .lbforaging.foraging_wrapper import ForagingEnvWrapper
from .qplex_smac.smac.env.lbforaging import register_envs as register_foraging

from .multiagentenvwrapper import MultiAgentEnvWrapper
from .join1 import Join1Env


def env_fn(env, **kwargs):
    return env(**kwargs)


REGISTRY = {
    "foraging": partial(env_fn, env=ForagingEnvWrapper),
    "join1": partial(env_fn, env=Join1Env),
}

register_foraging()
