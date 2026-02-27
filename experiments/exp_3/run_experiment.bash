#!/bin/bash
###########################
# experiment config
###########################
# experiment-specific setup
rl_algorithm="maic"

# 1 env and 1 param setup per scenario
scenarios=(
    "sc_1"
    "sc_2"
    "sc_3"
    "sc_4"
    )

envs=(
    "join1"
    "join1"
    "join1-v0"
    "join1-v0"
)

# same order as the env list
params=(
    "with test_interval=50000 t_max=1000000 test_nepisode=50"
    "with test_interval=50000 t_max=1000000 test_nepisode=50 env_args.n_agents=2 env_args.state_numbers=[2,2,2]"
    "with test_interval=50000 t_max=1000000 test_nepisode=50"
    "with test_interval=50000 t_max=1000000 test_nepisode=50 env_args.n_agents=2 env_args.state_numbers=[2,2,2]"
)

# the code runs every seed within each scenario
seeds=(0 2289 3608)

###########################
# get parent directory for exp name
exp_path=$(dirname "${BASH_SOURCE[0]}")
exp_name=$(basename $exp_path)

# Iterate over the setups to kick off experiments
# generic commands used for all runs
screen_cmd="screen -dmS"
python_cmd_prefix="python3 src/main.py --config=$rl_algorithm"
activate_env="source .venv/bin/activate;"

run_dir="$exp_path/tmp"
mkdir -p $run_dir

current_datetime=$(date +"%Y-%m-%d_%H-%M-%S")
run_path="${run_dir}/${current_datetime}_run.bash"

bash_prefix="#!/bin/bash"
echo $bash_prefix >> $run_path

# output commands to a separate bash file with datetime filename
for scenario_idx in "${!scenarios[@]}"; do
    scenario=${scenarios[$scenario_idx]}
    env=${envs[$scenario_idx]}
    param_str=${params[$scenario_idx]}

    echo "# $scenario" >> $run_path
    exp_scen_str="${exp_name}_${scenario}"

    for seed_idx in "${!seeds[@]}"; do
        seed=${seeds[$seed_idx]}
        screen_name="${exp_scen_str}_${env}_seed_${seed}"

        cmd="
        $screen_cmd $screen_name\
        bash -c '$activate_env $python_cmd_prefix --env-config=$env\
        $param_str seed=$seed exp_name=$exp_scen_str'\
        "

        echo $cmd >> $run_path

    done

    # newline for readability
    echo >> $run_path

done

# run all the commands
bash $run_path

# delete the commands file (optional)
# bash rm -r $run_path