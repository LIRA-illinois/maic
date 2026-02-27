from gymnasium.envs.registration import register

project_path = "maic.envs."

def register_env():
    register(id="join1-v0", entry_point=f"{project_path}join1_g.join1:Join1Env")