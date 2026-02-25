#!/bin/bash
# get parent directory for exp name
# parent_dir_name=$(basename "$(dirname "$filepath")")
exp_path=$(dirname "${BASH_SOURCE[0]}")
exp_name=$(basename $exp_path)

# experiment-specific setup
rl_algorithm="maic"
envs=("foraging" "join1")
# same order as the env list
params=(
    "with test_interval=25000 t_max=1000000 env_args.players=3 env_args.field_size=4 env_args.max_food=1"
    "with test_interval=25000 t_max=1000000 test_nepisode=50 env_args.n_agents=2 env_args.state_numbers=[2,2,2]"
)

# other params to loop over in the experiment
seeds=(0 2289 3608)

# Iterate over the setups to kick off experiments
# generic commands used for all runs
cmd_prefix="python3 src/main.py --config=$rl_algorithm"
activate_env="source .venv/bin/activate;"

for env_idx in "${!envs[@]}"; do
    for seed_idx in "${!seeds[@]}"; do
        seed=${seeds[$seed_index]}
        env_name=${envs[$env_idx]}
        param=${params[$env_idx]}
        screen -dmS "$exp_name-${envs[$env_idx]}-$seed_idx" bash -c "$activate_env $cmd_prefix --env-config=$env_name $param seed=$seed"
    done
done
