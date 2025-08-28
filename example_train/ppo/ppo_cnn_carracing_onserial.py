#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: example for ppo + carracing + cnn + on_serial
#  Update Date: 2021-06-11, Li Jie: create example

from pathlib import Path
import sys
import argparse
import numpy as np

from gops.create_pkg.create_alg import create_alg
from gops.create_pkg.create_buffer import create_buffer
from gops.create_pkg.create_env import create_env
from gops.create_pkg.create_evaluator import create_evaluator
from gops.create_pkg.create_sampler import create_sampler
from gops.create_pkg.create_trainer import create_trainer
from gops.utils.init_args import init_args
from gops.utils.plot_evaluation import plot_all
from gops.utils.tensorboard_setup import start_tensorboard, save_tb_to_csv

PROJ_DIR = Path.cwd().parent.parent
sys.path.append(str(PROJ_DIR))

if __name__ == "__main__":
    # Parameters Setup
    parser = argparse.ArgumentParser()

    ################################################
    # Key Parameters for users
    parser.add_argument("--env_id", type=str, default="gym_carracing", help="id of environment")
    parser.add_argument("--algorithm", type=str, default="PPO", help="RL algorithm")
    parser.add_argument("--enable_cuda", default=True, help="Disable CUDA")

    ################################################
    # 1. Parameters for environment
    parser.add_argument("--is_render", type=bool, default=False, help="Draw environment animation")
    parser.add_argument("--is_adversary", type=bool, default=False, help="Adversary training")
    parser.add_argument("--is_constrained", type=bool, default=False, help="Constrained training")

    ################################################
    # 2.1 Parameters of value approximate function
    parser.add_argument(
        "--value_func_name",
        type=str,
        default="StateValue",
        help="Options: StateValue/ActionValue/ActionValueDis/ActionValueDistri",
    )
    parser.add_argument("--value_func_type", type=str, default="CNN", help="Options: MLP/CNN/CNN_SHARED/RNN/POLY/GAUSS")
    value_func_type = parser.parse_args().value_func_type
    parser.add_argument(
        "--value_hidden_activation", type=str, default="relu", help="Options: relu/gelu/elu/selu/sigmoid/tanh"
    )
    parser.add_argument("--value_output_activation", type=str, default="linear", help="Options: linear/tanh")
    parser.add_argument("--value_conv_type", type=str, default="type_2", help="Options: type_1/type_2")

    # 2.2 Parameters of policy approximate function
    parser.add_argument(
        "--policy_func_name",
        type=str,
        default="StochaPolicy",
        help="Options: None/DetermPolicy/FiniteHorizonPolicy/StochaPolicy",
    )
    parser.add_argument(
        "--policy_func_type", type=str, default="CNN", help="Options: MLP/CNN/CNN_SHARED/RNN/POLY/GAUSS"
    )
    parser.add_argument(
        "--policy_act_distribution",
        type=str,
        default="GaussDistribution",
        help="Options: default/TanhGaussDistribution/GaussDistribution",
    )
    policy_func_type = parser.parse_args().policy_func_type
    parser.add_argument(
        "--policy_hidden_activation", type=str, default="relu", help="Options: relu/gelu/elu/selu/sigmoid/tanh"
    )
    parser.add_argument("--policy_conv_type", type=str, default="type_2", help="Options: type_1/type_2")
    parser.add_argument("--policy_min_log_std", type=int, default=-6)
    parser.add_argument("--policy_max_log_std", type=int, default=6)

    ################################################
    # 3. Parameters for algorithm
    parser.add_argument("--loss_coefficient_value", type=float, default=1.0)
    parser.add_argument("--loss_coefficient_entropy", type=float, default=0.001)
    parser.add_argument("--loss_value_clip", type=bool, default=True)
    parser.add_argument("--learning_rate", type=float, default=3e-4)

    ################################################
    # 4. Parameters for trainer
    parser.add_argument(
        "--trainer",
        type=str,
        default="on_serial_trainer",
        help="Options: on_serial_trainer, on_sync_trainer, off_serial_trainer, off_async_trainer",
    )
    # Maximum iteration number
    parser.add_argument("--max_iteration", type=int, default=200)
    trainer_type = parser.parse_args().trainer
    parser.add_argument(
        "--ini_network_dir",
        type=str,
        default=None
    )

    # 4.1. Parameters for on_serial_trainer
    parser.add_argument("--num_repeat", type=int, default=10)
    parser.add_argument("--num_mini_batch", type=int, default=8)
    parser.add_argument("--mini_batch_size", type=int, default=256)
    parser.add_argument(
        "--num_epoch",
        type=int,
        default=parser.parse_args().num_repeat * parser.parse_args().num_mini_batch,
        help="# 50 gradient step per sample",
    )
    parser.add_argument(
        "--buffer_name", type=str, default="replay_buffer", help="Options:replay_buffer/prioritized_replay_buffer"
    )
    # Size of collected samples before training
    parser.add_argument("--buffer_warm_size", type=int, default=1000)
    # Max size of reply buffer
    parser.add_argument("--buffer_max_size", type=int, default=30000)

    ################################################
    # 5. Parameters for sampler
    parser.add_argument("--sampler_name", type=str, default="on_sampler", help="Options: on_sampler/off_sampler")
    # Batch size of sampler for buffer store
    parser.add_argument("--sample_batch_size", type=int, default=2048)
    assert (
        parser.parse_args().num_mini_batch * parser.parse_args().mini_batch_size
        == parser.parse_args().sample_batch_size
    ), "sample_batch_size error"
    # Add noise to actions for better exploration
    parser.add_argument(
        "--noise_params",
        type=dict,
        default={"mean": np.array([0], dtype=np.float32), "std": np.array([0], dtype=np.float32)},
        help="Add noise to actions for exploration",
    )

    ################################################
    # 6. Parameters for evaluator
    parser.add_argument("--evaluator_name", type=str, default="evaluator")
    parser.add_argument("--num_eval_episode", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--eval_save", type=str, default=False, help="save evaluation data")

    ################################################
    # 7. Data savings
    parser.add_argument("--save_folder", type=str, default=None)
    # Save value/policy every N updates
    parser.add_argument("--apprfunc_save_interval", type=int, default=500)
    # Save key info every N updates
    parser.add_argument("--log_save_interval", type=int, default=100)

    ################################################
    # Get parameter dictionary
    args = vars(parser.parse_args())
    env = create_env(**args)
    args = init_args(env, **args)

    start_tensorboard(args["save_folder"])
    # Step 1: create algorithm and approximate function
    alg = create_alg(**args)
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
    # Plot and save training figures
    plot_all(args["save_folder"])
    save_tb_to_csv(args["save_folder"])
    print("Plot & Save are finished!")
