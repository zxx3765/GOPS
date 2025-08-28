#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: example for spil + mobilerobot + mlp + off_async
#  Update: 2021-03-05, Baiyu Peng: create example


import argparse
import os
import numpy as np
import multiprocessing

import sys

gops_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
sys.path.insert(0, gops_path)

from gops.create_pkg.create_alg import create_alg
from gops.create_pkg.create_buffer import create_buffer
from gops.create_pkg.create_env import create_env
from gops.create_pkg.create_evaluator import create_evaluator
from gops.create_pkg.create_sampler import create_sampler
from gops.create_pkg.create_trainer import create_trainer
from gops.utils.init_args import init_args
from gops.utils.plot_evaluation import plot_all
from gops.utils.tensorboard_setup import start_tensorboard, save_tb_to_csv


if __name__ == "__main__":
    # Parameters Setup
    parser = argparse.ArgumentParser()

    ################################################
    # Key Parameters for users
    parser.add_argument("--env_id", type=str, default="pyth_mobilerobot", help="id of environment")
    parser.add_argument("--algorithm", type=str, default="SPIL", help="RL algorithm")
    parser.add_argument("--enable_cuda", default=False, help="Enable CUDA")

    # 1. Parameters for environment
    parser.add_argument("--obsv_dim", type=int, default=None)
    parser.add_argument("--action_dim", type=int, default=None)
    parser.add_argument("--constrained_dim", type=int, default=None)
    parser.add_argument("--action_high_limit", type=list, default=None)
    parser.add_argument("--action_low_limit", type=list, default=None)
    parser.add_argument("--is_render", type=bool, default=False, help="Draw environment animation")
    parser.add_argument("--is_adversary", type=bool, default=False, help="Adversary training")

    ################################################
    # 2.1 Parameters of value approximate function
    parser.add_argument(
        "--value_func_name",
        type=str,
        default="StateValue",
        help="Options: StateValue/ActionValue/ActionValueDis/ActionValueDistri",
    )
    parser.add_argument("--value_func_type", type=str, default="MLP", help="Options: MLP/CNN/CNN_SHARED/RNN/POLY/GAUSS")
    value_func_type = parser.parse_known_args()[0].value_func_type
    parser.add_argument("--value_hidden_sizes", type=list, default=[64, 64])
    parser.add_argument(
        "--value_hidden_activation", type=str, default="relu", help="Options: relu/gelu/elu/selu/sigmoid/tanh"
    )
    parser.add_argument("--value_output_activation", type=str, default="linear", help="Options: linear/tanh")

    # 2.2 Parameters of policy approximate function
    parser.add_argument(
        "--policy_func_name",
        type=str,
        default="DetermPolicy",
        help="Options: None/DetermPolicy/FiniteHorizonPolicy/StochaPolicy",
    )
    parser.add_argument(
        "--policy_func_type", type=str, default="MLP", help="Options: MLP/CNN/CNN_SHARED/RNN/POLY/GAUSS"
    )
    parser.add_argument(
        "--policy_act_distribution",
        type=str,
        default="default",
        help="Options: default/TanhGaussDistribution/GaussDistribution",
    )
    policy_func_type = parser.parse_known_args()[0].policy_func_type
    parser.add_argument("--policy_hidden_sizes", type=list, default=[64, 64])
    parser.add_argument(
        "--policy_hidden_activation", type=str, default="relu", help="Options: relu/gelu/elu/selu/sigmoid/tanh"
    )

    ################################################
    # 3. Parameters for RL algorithm
    parser.add_argument("--value_learning_rate", type=float, default=2e-3)
    parser.add_argument("--policy_learning_rate", type=float, default=3e-4)

    # 4. Parameters for trainer
    parser.add_argument(
        "--trainer",
        type=str,
        default="off_async_trainer",
        help="Options: on_serial_trainer, on_sync_trainer, off_serial_trainer, off_async_trainer",
    )
    parser.add_argument("--max_iteration", type=int, default=4000)
    parser.add_argument("--ini_network_dir", type=str, default=None)
    trainer_type = parser.parse_known_args()[0].trainer

    # 4.1. Parameters for off_async_trainer
    parser.add_argument("--num_algs", type=int, default=2, help="number of algs")
    parser.add_argument("--num_samplers", type=int, default=1, help="number of samplers")
    parser.add_argument("--num_buffers", type=int, default=2, help="number of buffers")
    cpu_core_num = multiprocessing.cpu_count()
    num_core_input = (
        parser.parse_known_args()[0].num_algs
        + parser.parse_known_args()[0].num_samplers
        + parser.parse_known_args()[0].num_buffers
        + 2
    )
    if num_core_input > cpu_core_num:
        raise ValueError("The number of core is {}, but you want {}!".format(cpu_core_num, num_core_input))
    parser.add_argument("--alg_queue_max_size", type=int, default=1)
    parser.add_argument(
        "--buffer_name", type=str, default="replay_buffer", help="Options:replay_buffer/prioritized_replay_buffer"
    )
    # Size of collected samples before training
    parser.add_argument("--buffer_warm_size", type=int, default=10 * 1000)
    # Max size of reply buffer
    parser.add_argument("--buffer_max_size", type=int, default=400 * 1000)
    # Batch size of replay samples from buffer
    parser.add_argument("--replay_batch_size", type=int, default=1024)

    ################################################
    # 5. Parameters for sampler
    parser.add_argument("--sampler_name", type=str, default="off_sampler")
    # Batch size of sampler for buffer store
    parser.add_argument("--sample_batch_size", type=int, default=256)
    # Add noise to actions for better exploration
    parser.add_argument(
        "--noise_params",
        type=dict,
        default={"mean": np.array([0, 0], dtype=np.float32), "std": np.array([0.05, 0.05], dtype=np.float32),},
    )

    ################################################
    # 6. Parameters for evaluator
    parser.add_argument("--evaluator_name", type=str, default="evaluator")
    parser.add_argument("--num_eval_episode", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_save", type=str, default=False, help="save evaluation data")

    ################################################
    # 7. Data savings
    parser.add_argument("--save_folder", type=str, default=None)
    # Save value/policy every N updates
    parser.add_argument("--apprfunc_save_interval", type=int, default=1000)
    # Save key info every N updates
    parser.add_argument("--log_save_interval", type=int, default=100)

    ################################################
    # Get parameter dictionary
    args = vars(parser.parse_args())
    env = create_env(**args)
    args = init_args(env, **args)
    start_tensorboard(args["save_folder"])
    # Step 1: create algorithm and approximate function
    alg = create_alg(**args)  # create appr_model in algo **vars(args)
    # Step 2: create sampler in trainer
    sampler = create_sampler(**args)
    # Step 3: create buffer in trainer
    buffer = create_buffer(**args)
    # Step 4: create evaluator in trainer
    evaluator = create_evaluator(**args)
    # Step 5: create trainer
    trainer = create_trainer(alg, sampler, buffer, evaluator, **args)

    ################################################
    # Start training ... ...
    trainer.train()
    print("Training is finished!")

    ################################################
    # plot and save training curve
    plot_all(args["save_folder"])
    save_tb_to_csv(args["save_folder"])
    print("Plot & Save are finished!")
