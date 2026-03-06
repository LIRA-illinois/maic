from multiprocessing import Pipe, Process
import numpy as np
import gymnasium as gym

from .runner import Runner, step, get_state, get_avail_actions, get_obs


# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner(Runner):
    """class to run multiple episodes in parallel"""

    def __init__(self, args, logger):
        super().__init__(args, logger)

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(
            *[Pipe() for _ in range(self.batch_size)]
        )

        # env_args_list: list[dict] = [
        #     self.args.env_args.copy() for _ in range(self.batch_size)
        # ]

        self.processes: list[Process] = []
        for process_idx, worker_conn in enumerate(self.worker_conns):
        # for env_args, worker_conn in zip(env_args_list, self.worker_conns):
            # each process gets its a separate instance of the env
            env = self.build_env()

            process = Process(
                target=env_worker,
                args=(
                    worker_conn,
                    process_idx,
                    env,
                    self.args.seed,
                ),
            )

            self.processes.append(process)

        for p in self.processes:
            p.daemon = True
            p.start()

        # initialize each env's PRNG
        self.set_env_seed()

    def set_env_seed(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("set_env_seed", None))

    def get_env_info(self):
        self.parent_conns[0].send(("get_env_info", None))
        info = self.parent_conns[0].recv()
        info["episode_limit"] = self.episode_limit

        return info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        self._reset()

        self.env_steps_this_run = 0

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        pre_transition_data = {"state": [], "avail_actions": [], "obs": []}
        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])

        self.batch.update(pre_transition_data, ts=0)

    def run(self, test_mode=False):
        self.reset()

        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [
            b_idx for b_idx, termed in enumerate(terminated) if not termed
        ]
        final_env_infos = (
            []
        )  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

        while True:
            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            actions = self.mac.select_actions(
                self.batch,
                t_ep=self.t,
                t_env=self.t_env,
                bs=envs_not_terminated,
                test_mode=test_mode,
            )

            # ensure this is size (batch_size_run, 1, n_agents)
            actions = np.expand_dims(actions.cpu().numpy(), 1)

            self.batch.update(
                {"actions": actions},
                bs=envs_not_terminated,
                ts=self.t,
                mark_filled=False,
            )

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                # We produced actions for this env
                if idx in envs_not_terminated:
                    # Only send the actions to the env if it hasn't terminated
                    if not terminated[idx]:
                        parent_conn.send(("step", actions[action_idx]))
                    action_idx += 1  # actions is not a list over every env

            # Update envs_not_terminated
            envs_not_terminated: list[int] = [
                b_idx for b_idx, termed in enumerate(terminated) if not termed
            ]

            if all(terminated):
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {"reward": [], "terminated": []}
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {"state": [], "avail_actions": [], "obs": []}

            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((data["reward"],))

                    episode_returns[idx] += data["reward"]
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get(
                        "episode_limit", False
                    ):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])

            # Add post_transiton data into the batch
            self.batch.update(
                post_transition_data,
                bs=envs_not_terminated,
                ts=self.t,
                mark_filled=False,
            )

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(
                pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True
            )

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats", None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        infos = [cur_stats] + final_env_infos
        cur_stats.update(
            {
                k: sum(d.get(k, 0) for d in infos)
                for k in set.union(*[set(d) for d in infos])
            }
        )
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        cur_returns.extend(episode_returns)

        n_test_runs = (
            max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        )
        if test_mode and (len(self.test_returns) == n_test_runs):
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
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(
                    prefix + k + "_mean", v / stats["n_episodes"], self.t_env
                )
        stats.clear()


def env_worker(remote, process_idx: int, env: gym.Env, seed: int):

    while True:
        cmd, data = remote.recv()

        match cmd:
            case "step":
                actions = data
                # Take a step in the environment
                _, reward, terminated, truncated, env_info = step(actions, env)

                # "terminated" in the runner's scope is equivalent to "terminated or truncated" in env_worker's scope
                terminated = terminated or truncated

                # Return the observations, avail_actions and state to make the next action
                state = get_state(env)
                avail_actions = get_avail_actions(env)
                obs = get_obs(env)

                remote.send(
                    {
                        # Data for the next timestep needed to pick an action
                        "state": state,
                        "avail_actions": avail_actions,
                        "obs": obs,
                        # Rest of the data for the current timestep
                        "reward": reward,
                        "terminated": terminated,
                        "info": env_info,
                    }
                )

            case "set_env_seed":
                print(f"process {process_idx} setting env seed to {seed}")
                env.reset(seed=seed)

            case "reset":
                # print("resetting env")
                env.reset()
                # state = get_state(env)

                remote.send(
                    {
                        "state": get_state(env),
                        "avail_actions": get_avail_actions(env),
                        "obs": get_obs(env),
                    }
                )

            case "close":
                env.close()
                remote.close()
                break

            case "get_env_info":
                if hasattr(env, "unwrapped"):
                    info: dict = env.unwrapped.get_env_info()
                else:
                    info: dict = env.get_env_info()

                remote.send(info)

            case "get_stats":
                # some envs do not have a get_stats method
                if hasattr(env, "get_stats") and callable(
                    getattr(env, "get_stats")
                ):
                    if hasattr(env, "unwrapped"):
                        stats: dict = env.unwrapped.get_stats()
                    else:
                        stats: dict = env.get_env_info()
                else:
                    stats = {}
                remote.send(stats)

            case _:
                raise NotImplementedError
