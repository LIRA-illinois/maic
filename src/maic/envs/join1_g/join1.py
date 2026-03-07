import enum
from typing import Optional, Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray


class Actions(enum.IntEnum):
    still = 0
    left = 1
    right = 2


class Join1Env(gym.Env):
    """3-agent independent hallway env updated to match the Gymnasium API. All agents must reach the goal state at the same time to succeed."""

    def __init__(
        self,
        n_agents: int = 3,
        state_numbers: tuple[int, ...] = (
            2,
            6,
            10,
        ),
        reward_win: float = 10.0,
        obs_last_action: bool = False,
        state_last_action: bool = True,
        is_print: bool = False,
        print_rew: bool = False,
        print_steps: int = 1000,
    ):
        # Map arguments
        self.print_rew = print_rew
        self.is_print = is_print
        self.print_steps = print_steps
        self.n_agents = n_agents
        self.n_states = np.array(state_numbers, dtype=np.int_)

        # Observations and state
        self.obs_last_action = obs_last_action
        self.state_last_action = state_last_action

        # initialize agents
        self.state: NDArray

        self.observation_space = self._set_observation_space()

        # Rewards args
        self.reward_win = reward_win

        # Actions
        self.n_actions = len(Actions)
        self.action_space = spaces.MultiDiscrete(
            [len(Actions) for _ in range(self.n_agents)]
        )

        # Statistics
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self.battles_won = 0
        self.battles_game = 0

        self.p_step = 0
        self.rew_gather = []
        self.is_print_once = False

        self.last_action = np.zeros((self.n_agents, self.n_actions))

    def step(self, actions: NDArray[np.int_]):
        """transition env from s_t to s_{t+1} with team of agents taking joint action a_t
        Returns obs, reward, terminated, truncated, info."""
        self._total_steps += 1
        self._episode_steps += 1
        info = {}

        if self.is_print:
            print("t_steps: %d" % self._episode_steps)
            print(self.state)
            print(actions)

        # transition env state
        for agent_i, action in enumerate(actions):
            if action == Actions.still:
                pass
            elif action == Actions.left:
                self.state[agent_i] = max(0, self.state[agent_i] - 1)

            elif action == Actions.right:
                self.state[agent_i] = min(
                    self.n_states[agent_i], self.state[agent_i] + 1
                )

        # get obs for the new env state
        obs = self.get_obs()

        # get rewards for the transition, check if episode terminated
        reward = 0
        terminated = False
        info["battle_won"] = False

        # all agents reached their goal state at the same time
        if (self.state == 0).all():
            reward = self.reward_win
            terminated = True
            self.battles_won += 1
            info["battle_won"] = True

        # m agents reached the goal state where 1 < m < n_agents
        elif (self.state == 0).any():
            terminated = True

        if terminated:
            self._episode_count += 1
            self.battles_game += 1

        # truncated is handled by a Gymnasium wrapper when
        # max_episode_steps is passed as an arg to make()
        truncated: bool = False

        if self.print_rew:
            self.p_step += 1
            if terminated:
                self.rew_gather.append(reward)
            if self.p_step % self.print_steps == 0:
                print(
                    "steps: %d, average rew: %.3lf"
                    % (self.p_step, float(np.mean(self.rew_gather)) / self.reward_win)
                )
                self.is_print_once = True

        return obs, reward, terminated, truncated, info

    def get_obs(self) -> NDArray:
        """Returns team's joint observation, size (n_agents, obs_size)."""
        obs_list: list[NDArray] = [self.get_obs_agent(i) for i in range(self.n_agents)]
        obs = np.vstack(obs_list)
        return obs

    def get_obs_agent(self, agent_id) -> NDArray:
        """Returns observation for agent_id."""
        return np.array([self.state[agent_id]])

    def get_obs_size(self) -> int:
        # """Returns the size of each agent's observation."""
        return 1

    def _set_observation_space(self) -> spaces.Box:
        """size of the team's joint observation space"""
        obs_shape = (self.n_agents, self.get_obs_size())
        min_obs, max_obs = 0, max(self.n_states)

        obs_space = spaces.Box(
            low=min_obs * np.ones(obs_shape),
            high=max_obs * np.ones(obs_shape),
            dtype=np.int_,
        )

        return obs_space

    def get_state(self) -> NDArray:
        """Returns the global state."""
        # return deepcopy so changing self.state in step()
        # will not affect the returned state in the runner's pre_transition_data
        return self.state.copy()

    def get_state_size(self) -> int:
        """Returns the size of the global state."""
        return self.n_agents

    def get_avail_actions(self) -> tuple:
        """Returns the available actions of all agents."""
        avail_actions = [self.get_avail_agent_actions(i) for i in range(self.n_agents)]
        return tuple(avail_actions)

    def get_avail_agent_actions(self, agent_id) -> tuple[int, ...]:
        """Returns the available actions for agent_id."""
        avail_actions = [1] * self.n_actions
        return tuple(avail_actions)

    def get_total_actions(self) -> int:
        """Returns the size of a single agent's action space"""
        return self.n_actions

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[NDArray, dict]:
        """Returns initial observations and info."""

        super().reset(seed=seed)

        self._episode_steps = 0
        self.last_action = np.zeros((self.n_agents, self.n_actions))

        self.state = np.array(
            [
                self.np_random.integers(low=1, high=self.n_states[i] + 1)
                for i in range(self.n_agents)
            ],
            dtype=np.int_,
        )

        obs = self.get_obs()
        info = {"battle_won": False}

        return obs, info

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass

    def save_replay(self) -> None:
        """Save a replay."""
        pass

    def get_env_info(self) -> dict[str, int]:
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
        }
        return env_info

    def get_stats(self) -> dict[str, Any]:
        stats = {
            "battles_won": self.battles_won,
            "battles_game": self.battles_game,
            "win_rate": self.battles_won / self.battles_game,
        }
        return stats

    def clean(self) -> None:
        self.p_step = 0
        self.rew_gather = []
        self.is_print_once = False
