from typing import Any
from functools import partial
import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from maic.envs import REGISTRY as env_REGISTRY, register_envs, get_env_id
from maic.components.episode_buffer import EpisodeBatch


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.episode_limit: int = self.args.env_args.get("max_episode_steps")

        register_envs()
        if self.args.env in env_REGISTRY:
            self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        else:

            if self.args.env == "foraging":
                # this env already has all its params defined when registered
                env_id = get_env_id(env=self.args.env, env_args=self.args.env_args)
                self.args.env = env_id
                self.env: gym.Env = gym.make(self.args.env)

            else:
                self.env: gym.Env = gym.make(self.args.env, **self.args.env_args)

        # initialize the env's PRNG
        self.env.reset(seed=self.args.seed)

        # run basic env checks to follow the Gymnasium API
        try:
            check_env(self.env.unwrapped, skip_render_check=True)
        except Exception as e:
            print(f"Env has issues: {e}")


        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

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

    def get_env_info(self) -> dict[str, Any]:
        if hasattr(self.env, "unwrapped"):
            info: dict = self.env.unwrapped.get_env_info()
        else:
            info: dict = self.env.get_env_info()

        info["episode_limit"] = self.episode_limit

        return info

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def get_state(self):
        if hasattr(self.env, "unwrapped"):
            return self.env.unwrapped.get_state()
        else:
            return self.env.get_state()

    def get_avail_actions(self):
        if hasattr(self.env, "unwrapped"):
            return self.env.unwrapped.get_avail_actions()
        else:
            return self.env.get_avail_actions()

    def get_obs(self):
        if hasattr(self.env, "unwrapped"):
            return self.env.unwrapped.get_obs()
        else:
            return self.env.get_obs()

    def reset(self):
        self.batch = self.new_batch()
        # do not use seed in calling reset() here, doing that will reset the PRNG to its initial state
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        truncated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not (terminated or truncated):
            pre_transition_data = {
                "state": [self.get_state()],
                "avail_actions": [self.get_avail_actions()],
                "obs": [self.get_obs()],
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(
                self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
            )

            # following the format from the parallel episode runner
            actions = actions.cpu().numpy()

            if hasattr(self.env, "unwrapped"):
                _, reward, terminated, truncated, env_info = self.env.step(actions[0])
                terminated = bool(terminated)
            else:
                # only for the non-Gymnasium version of the env
                reward, terminated, env_info = self.env.step(actions[0])

            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [
                    (terminated != env_info.get("episode_limit", False),)
                ],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.get_state()],
            "avail_actions": [self.get_avail_actions()],
            "obs": [self.get_obs()],
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(
            self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
        )
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update(
            {
                k: cur_stats.get(k, 0) + env_info.get(k, 0)
                for k in set(cur_stats) | set(env_info)
            }
        )
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat(
                    "epsilon", self.mac.action_selector.epsilon, self.t_env
                )
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_median", np.median(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(
                    prefix + k + "_mean", v / stats["n_episodes"], self.t_env
                )
        stats.clear()
