from functools import partial
# non-gymnasium envs
from .multiagentenvwrapper import MultiAgentEnvWrapper
from .join1 import Join1Env
from .lbforaging.foraging_wrapper import ForagingEnvWrapper
from .qplex_smac.smac.env.lbforaging import register_envs as register_foraging

# gymnasium envs
from .join1_g import register_env as register_join1_g


def env_fn(env, **kwargs):
    return env(**kwargs)

REGISTRY = {
    "foraging": partial(env_fn, env=ForagingEnvWrapper),
    "join1": partial(env_fn, env=Join1Env),
}

def register_envs():
    register_join1_g()
    register_foraging()