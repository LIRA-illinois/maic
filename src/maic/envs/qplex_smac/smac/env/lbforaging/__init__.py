from gymnasium.envs.registration import register
from itertools import product

from maic.envs.qplex_smac.smac.env.lbforaging.foraging import ForagingEnv


def register_envs():
    sizes = range(5, 20)
    players = range(2, 20)
    foods = range(1, 10)
    coop = [True, False]
    partial_obs = [True, False]

    for s, p, f, c, po in product(sizes, players, foods, coop, partial_obs):
        register(
            id="Foraging{4}-{0}x{0}-{1}p-{2}f{3}-v1".format(s, p, f, "-coop" if c else "", "-2s" if po else ""),
            entry_point="maic.envs.qplex_smac.smac.env.lbforaging:ForagingEnv",
            kwargs={
                "players": p,
                "max_player_level": 6,
                "field_size": (s, s),
                "max_food": f,
                "sight": 2 if po else s,
                "episode_limit": 50,
                "force_coop": c,
            },
        )
