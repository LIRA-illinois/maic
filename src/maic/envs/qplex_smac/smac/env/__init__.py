from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from maic.envs.qplex_smac.smac.env.multiagentenv import MultiAgentEnvWrapper
# do not import smac b/c it requires a specific version of protobuf that conflicts with tensorboard's requirements
# from maic.envs.qplex_smac.smac.env.starcraft2.starcraft2 import StarCraft2Env
from maic.envs.qplex_smac.smac.env.matrix_game_1 import Matrix_game1Env
from maic.envs.qplex_smac.smac.env.matrix_game_2 import Matrix_game2Env
from maic.envs.qplex_smac.smac.env.matrix_game_3 import Matrix_game3Env
from maic.envs.qplex_smac.smac.env.mmdp_game_1 import mmdp_game1Env
from maic.envs.qplex_smac.smac.env.lbforaging import ForagingEnv

__all__: list[str] = [
    "MultiAgentEnvWrapper",
    # "StarCraft2Env",
    "Matrix_game1Env",
    "Matrix_game2Env",
    "Matrix_game3Env",
    "mmdp_game1Env",
]
