from typing import Any
import numpy as np
import gymnasium as gym

from .runner import Runner, step, get_state, get_avail_actions, get_obs


class EpisodeRunner(Runner):
    """class to run single episodes"""

    def __init__(self, args, logger):
        super().__init__(args, logger)
        assert self.batch_size == 1
        self.env = self.build_env()

        # initialize the env's PRNG
        self.set_env_seed()

        # run basic env checks to follow the Gymnasium API
        self.check_env(self.env)

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self._reset()
        # do not use seed in calling reset() here, doing that will reset the PRNG to its initial state
        self.env.reset()

    def get_env_info(self) -> dict[str, Any]:
        if hasattr(self.env, "unwrapped"):
            info: dict = self.env.unwrapped.get_env_info()
        else:
            info: dict = self.env.get_env_info()

        info["episode_limit"] = self.episode_limit
        return info

    def set_env_seed(self):
        print(f"setting env seed to {self.args.seed}")
        self.env.reset(seed=self.args.seed)

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        truncated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not (terminated or truncated):
            pre_transition_data = {
                "state": get_state(self.env),
                "avail_actions": [get_avail_actions(self.env)],
                "obs": get_obs(self.env),
            }
            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(
                self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
            )

            # following the format from the parallel episode runner
            actions = actions.cpu().numpy()
            _, reward, terminated, truncated, env_info = step(actions, env=self.env)

            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": get_state(self.env),
            "avail_actions": [get_avail_actions(self.env)],
            "obs": get_obs(self.env),
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
