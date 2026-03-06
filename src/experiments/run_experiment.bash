#!/bin/bash

# get the experiment config parsing the -e arg
while getopts e:g: option
do
    case "${option}"
        in
        e)exp_name=${OPTARG};;
    esac
done

# parse experiment.config
exp_path="experiments/${exp_name}"
source ${exp_path}/experiment.config

# set up bash runner file in a temporary dir
run_dir="$exp_path/tmp"
mkdir -p $run_dir
current_datetime=$(date +"%Y-%m-%d_%H-%M-%S")
run_path="${run_dir}/${current_datetime}_run.bash"

# get available GPU indices if computer has GPUs
if [[ ${#avail_gpus[@]} -eq 0 ]] ; then
    mapfile -t avail_gpus < <( nvidia-smi --query-gpu=index --format=csv,noheader,nounits )
fi

n_gpus=${#avail_gpus[@]}
echo "$n_gpus available GPUs with indices (${avail_gpus[@]})"

# output commands to runner file
# generic commands used for all runs
bash_prefix="#!/bin/bash"
echo $bash_prefix >> $run_path

screen_cmd="screen -dmS"
activate_env="source .venv/bin/activate;"

# check that you defined 1 alg and 1 env per set of scenario params
n_scenarios=${#scenario_params[@]}
n_rl_algs=${#rl_algs[@]}
n_envs=${#envs[@]}

if [ $n_scenarios -eq $n_rl_algs ] && [ $n_rl_algs -eq $n_envs ]; then
    # do nothing
    :
else
    echo "Error in experiment.config, rl_algs, envs, and scenario_params are different lengths."
    exit
fi

# loop over scenarios to generate 1 command per scenario
for ((scenario_idx = 0; scenario_idx < $n_scenarios; scenario_idx++)); do
    scenario_name=sc_$((scenario_idx+1))

    # read env and rl alg from the config file
    scenario_param=${scenario_params[$scenario_idx]}
    rl_alg=${rl_algs[$scenario_idx]}
    env=${envs[$scenario_idx]}

    # for GPU index just take scenario_idx mod len(avail_gpus)
    gpu_idx=$((scenario_idx % $n_gpus))

    echo "# ${scenario_name}" >> $run_path
    python_cmd_prefix="python3 src/main.py"
    exp_scen_str="${exp_name}_${scenario_name}"

    # 1 scenario involves running multiple seeds
    for seed in "${seeds[@]}"; do
        screen_name="${exp_scen_str}_${rl_alg}_${env}_seed_${seed}"

        cmd="
        $screen_cmd $screen_name\
        bash -c '$activate_env $python_cmd_prefix --config=$rl_alg --env-config=$env\
        with $same_params $scenario_param seed=$seed device_idx=$gpu_idx exp_name=$exp_scen_str'\
        "

        echo $cmd >> $run_path

    done

    # newline for readability
    echo >> $run_path

done

echo "Writing experiment commands to $run_path"

# check if user wants to run all commands in the runner file
read -rp "Run experiment now? (y/n) " run_now
# get lowercase input
run_now="${run_now,,}"

if [[ "$run_now" == "y" ]]; then
    echo "Running commands in $run_path"
    bash $run_path
else
    echo "Exiting without running experiment"
fi

# delete the runner file (optional)
# bash rm -r $run_path
