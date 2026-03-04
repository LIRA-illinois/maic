from functools import partial
import numpy as np
from numpy.typing import NDArray
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from maic.envs import register_envs
from maic.components.episode_buffer import EpisodeBatch
from maic.envs import REGISTRY as env_REGISTRY, get_env_id


class Runner:
    """general episode runner class"""

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        self.episode_limit: int = self.args.env_args.get("max_episode_steps")

        # register all envs we may want to run
        register_envs(env=self.args.env)

        self.env: gym.Env
        self.batch: EpisodeBatch
        self.new_batch: partial
        self.mac: object

        self.t = 0
        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -1000000

    def build_env(self) -> gym.Env:
        # make the env
        if self.args.env in env_REGISTRY:
            # old PYMARL way of doing it
            env = env_REGISTRY[self.args.env](**self.args.env_args)

        else:
            if self.args.env == "foraging-v2":
                # special way for foraging b/c they pre-register their envs with kwargs under specific names
                env_id = get_env_id(env=self.args.env, env_args=self.args.env_args)
                env = gym.make(env_id)

            else:
                # normal way that follows the Gymnasium website's example
                env = gym.make(self.args.env, **self.args.env_args)

        return env


    def check_env(self, env):
        if hasattr(env, "unwrapped"):
            tmp_env = env.unwrapped
        else:
            tmp_env = env

        try:
            check_env(tmp_env, skip_render_check=True)
        except Exception as e:
            print(f"Env has issues: {e}")

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(
            EpisodeBatch,
            scheme,
            groups,
            self.batch_size,
            self.episode_limit + 1,
            preprocess=preprocess,
            device=self.args.device,
        )
        self.mac = mac

    def _reset(self):
        self.batch = self.new_batch()
        self.t = 0

    def get_env_info(self):
        raise NotImplementedError

    def set_env_seed(self):
        raise NotImplementedError

def step(actions: NDArray, env: gym.Env):
    # actions must be size (1, n_agents)
    if hasattr(env, "unwrapped"):
        obs, reward, terminated, truncated, env_info = env.step(actions[0])
        terminated = bool(terminated)

    else:
        # only for the non-Gymnasium version of the env
        reward, terminated, env_info = env.step(actions[0])
        obs = None
        truncated = None

    return obs, reward, terminated, truncated, env_info


def get_state(env: gym.Env) -> NDArray:
    if hasattr(env, "unwrapped"):
        state = env.unwrapped.get_state()
    else:
        state = env.get_state()

    # expand 0th dimension to be size (n_samples=1, n_agents, n_features)
    return np.expand_dims(state, 0)


def get_avail_actions(env: gym.Env) -> list:
    if hasattr(env, "unwrapped"):
        return env.unwrapped.get_avail_actions()
    else:
        return env.get_avail_actions()


def get_obs(env: gym.Env) -> NDArray:
    if hasattr(env, "unwrapped"):
        obs = env.unwrapped.get_obs()
    else:
        obs = env.get_obs()

    # expand 0th dimension to be size (n_samples=1, n_agents, n_features)
    return np.expand_dims(obs, 0)
