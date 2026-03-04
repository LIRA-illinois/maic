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
    "sc_5"
    "sc_6"
    "sc_7"
    "sc_8"
    "sc_9"
    )

envs=(
    "foraging-v2"
    "foraging-v2"
    "foraging-v2"
    "foraging-v2"
    "foraging-v2"
    "foraging-v2"
    "foraging-v2"
    "foraging-v2"
    "foraging-v2"
)

# same order as the env list
params=(
    "with test_interval=50000 t_max=2000000 test_nepisode=40 env_args.field_size=10 env_args.players=4 runner=parallel batch_size_run=1 batch_size=32"
    "with test_interval=50000 t_max=2000000 test_nepisode=40 env_args.field_size=10 env_args.players=4 runner=parallel batch_size_run=2 batch_size=64"
    "with test_interval=50000 t_max=2000000 test_nepisode=40 env_args.field_size=10 env_args.players=4 runner=parallel batch_size_run=5 batch_size=160"
    "with test_interval=50000 t_max=2000000 test_nepisode=40 env_args.field_size=10 env_args.players=4 runner=parallel batch_size_run=10 batch_size=320"
    "with test_interval=50000 t_max=2000000 test_nepisode=40 env_args.field_size=10 env_args.players=4 runner=parallel batch_size_run=20 batch_size=640"
    "with test_interval=50000 t_max=2000000 test_nepisode=40 env_args.field_size=10 env_args.players=4 runner=parallel batch_size_run=40 batch_size=1280"
    # it may also be the case that you need larger batch sizes for smaller number of added eps, so instead of batch_size_run*32, set batch_size=2*batch_size_run*32
    "with test_interval=50000 t_max=2000000 test_nepisode=40 env_args.field_size=10 env_args.players=4 runner=parallel batch_size_run=5 batch_size=320"
    "with test_interval=50000 t_max=2000000 test_nepisode=40 env_args.field_size=10 env_args.players=4 runner=parallel batch_size_run=10 batch_size=640"
    "with test_interval=50000 t_max=2000000 test_nepisode=40 env_args.field_size=10 env_args.players=4 runner=parallel batch_size_run=20 batch_size=1280"
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