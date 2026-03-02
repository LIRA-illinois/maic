from functools import partial

# taken directly from https://github.com/semitable/lb-foraging, v 2.0.0 compatible with Gymnasium
import lbforaging as lbf


# non-gymnasium envs
from .multiagentenvwrapper import MultiAgentEnvWrapper
from .join1 import Join1Env

# from .lbforaging.foraging_wrapper import ForagingEnvWrapper
# from .qplex_smac.smac.env.lbforaging import register_envs as register_foraging


# gymnasium envs
from .join1_g import register_env as register_join1_g


def env_fn(env, **kwargs):
    return env(**kwargs)


REGISTRY = {
    # "foraging": partial(env_fn, env=ForagingEnvWrapper),
    "join1": partial(env_fn, env=Join1Env),
}


def register_envs():
    register_join1_g()
    lbf.register_envs()

    # not using the grid obs for now, but may experiment with it later
    # lbf.register_grid_envs()
    # register_foraging()


def get_env_id(env: str, env_args: dict):
    """get the full ID used for gym registration when there is a difference between env's name in the yaml config file (for PYMARL) and its ID when registered in gymnasium in the env package

    env: str - name of the environment as specified in its .yaml config file
    """
    match env:
        case "foraging":
            id_args = {
                "s": env_args["field_size"],
                "p": env_args["players"],
                "f": env_args["max_num_food"],
                "c": env_args["force_coop"],
                "po": env_args["partially_observe"],
                "pen": env_args["penalty"],
                "mfl": (
                    env_args["max_food_level"]
                    if "max_food_level" in env_args.keys()
                    else None
                ),
            }

            env_id = lbf.get_env_id(**id_args)

    return env_id
