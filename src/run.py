import datetime
import os
from os.path import dirname, abspath
import pprint
import time
import threading
import torch as th
import json
from types import SimpleNamespace as SN


from maic.utils.logging import Logger
from maic.utils.timehelper import time_left, time_str
from maic.learners import REGISTRY as le_REGISTRY
from maic.runners import REGISTRY as r_REGISTRY
from maic.controllers import REGISTRY as mac_REGISTRY
from maic.components.episode_buffer import ReplayBuffer
from maic.components.transforms import OneHot


def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    if args.use_cuda:
        if args.device_idx:
            args.device = args.device_idx
        else:
            args.device = "cuda"
    else:
        args.device = "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    if len(args.comment) > 0:
        alg_name = "{}_{}".format(args.name, args.comment)
    else:
        alg_name = args.name

    curr_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")[:-3]
    if str(args.env).startswith("sc2"):
        unique_token = f"{args.exp_name}_{curr_time}_{args.env}_{args.env_args['map_name']}_{alg_name}_seed_{args.seed}"
    else:
        unique_token = (
            f"{args.exp_name}_{curr_time}_{args.env}_{alg_name}_seed_{args.seed}"
        )

    args.unique_token = unique_token

    if str(args.env).startswith("sc2"):
        tb_logs_direc = os.path.join(
            dirname(dirname(abspath(__file__))),
            "results",
            "tb_logs",
            args.env,
            args.env_args["map_name"],
            alg_name,
        )
    else:
        tb_logs_direc = os.path.join(
            dirname(dirname(abspath(__file__))),
            "results",
            "tb_logs",
            args.env,
            alg_name,
        )
    tb_exp_direc = os.path.join(tb_logs_direc, unique_token)
    if args.use_tensorboard:
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    if args.use_tensorboard:
        if str(args.env).startswith("sc2"):
            json_output_direc = os.path.join(
                dirname(dirname(abspath(__file__))),
                "results",
                "json_out",
                args.env,
                args.env_args["map_name"],
                alg_name,
            )
        else:
            json_output_direc = os.path.join(
                dirname(dirname(abspath(__file__))),
                "results",
                "json_out",
                args.env,
                alg_name,
            )
        json_exp_direc = os.path.join(json_output_direc, unique_token + ".json")
        print(
            f"Export tensorboard scalars at {tb_exp_direc} to json file {json_exp_direc}"
        )
        export_scalar_to_json(tb_exp_direc, json_output_direc, args)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def run_sequential(args, logger):
    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.episode_limit = env_info["episode_limit"]
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    if "unit_dim" in env_info:
        args.unit_dim = env_info["unit_dim"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": th.int,
        },
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

    env_name = args.env
    if env_name == "sc2":
        env_name += "/" + args.env_args["map_name"]

    buffer = ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.to(args.device)

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info(
                "Checkpoint directiory {} doesn't exist".format(args.checkpoint_path)
            )
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)

        if args.training_option == "single_epoch_update":
            if buffer.can_sample(args.batch_size):
                episode_sample = buffer.sample(args.batch_size)

                # Truncate batch to only filled timesteps
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != args.device:
                    episode_sample.to(args.device)

                learner.train(episode_sample, runner.t_env, episode)

        elif args.training_option == "multi_epoch_update":
            # sample the buffer batch_size_run times to perform multiple training updates from the same dataset
            # expected value of the gradient should be the same as a large batch size, but smaller batches usually work better for RL training
            # smaller batches will have higher variance which may be better for RL training to avoid local minima in the loss landscape
            # this is SIGNIFICANTLY slower than single_epoch_update since train() and loss.backward() take up a huge proportion of clock time in the training loop
            # this could be parallelized, but that isn't my focus
            if buffer.can_sample(args.batch_size):
                for i in range(args.batch_size_run):
                    episode_sample = buffer.sample(args.batch_size)

                    # Truncate batch to only filled timesteps
                    max_ep_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_ep_t]

                    if episode_sample.device != args.device:
                        episode_sample.to(args.device)

                    learner.train(episode_sample, runner.t_env, episode)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info(
                "t_env: {} / {}".format(runner.t_env, args.t_max)
            )
            logger.console_logger.info(
                "Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, runner.t_env, args.t_max),
                    time_str(time.time() - start_time),
                )
            )
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (
            runner.t_env - model_save_time >= args.save_model_interval
            or model_save_time == 0
        ):
            model_save_time = runner.t_env
            save_path = os.path.join(
                args.local_results_path, "models", args.unique_token, str(runner.t_env)
            )
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]
        ) * config["batch_size_run"]

    return config


def export_scalar_to_json(tensorboard_path, output_path, args):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    os.makedirs(output_path, exist_ok=True)
    filename = os.path.basename(tensorboard_path)
    output_path = os.path.join(output_path, filename + ".json")
    summary = EventAccumulator(tensorboard_path).Reload()
    scalar_list = summary.Tags()["scalars"]
    stone_dict = {}
    stone_dict["seed"] = args.seed
    for scalar_name in scalar_list:
        stone_dict["_".join([scalar_name, "T"])] = [
            scalar.step for scalar in summary.Scalars(scalar_name)
        ]
        stone_dict[scalar_name] = [
            scalar.value for scalar in summary.Scalars(scalar_name)
        ]
    json.dump(stone_dict, open(output_path, "w"), ensure_ascii=False)
