#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: run a closed-loop system
#  Update: 2022-12-05, Congsheng Zhang: create file

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from gops.sys_simulator.PolicyRunnerCustom import PolicyRunnerCustom
import numpy as np
result_path = "D:/Project/GOPS/results/simu_quarter_sus_win/"
runner = PolicyRunnerCustom(
    log_policy_dir_list=[
                         result_path+"DDPGCustom_250831-080656",],
    trained_policy_iteration_list=['409000_opt',],
    is_init_info=True,
    init_info={"init_state": [0.0, 0.0, 0.0, 0.0], "ref_time": 0.0,
               "ref_num": 3}, # ref_num = [0, 1, 2,..., 7]
    save_render=False,
    legend_list=["409000_opt"],
    opt_args={
        "opt_controller_type": "OPT",
        "num_pred_step": 10,
        "gamma": 0.99,
        "mode": "shooting",
        "minimize_options": {
            "max_iter": 2000,
            "tol": 1e-4,
            "acceptable_tol": 1e-2,
            "acceptable_iter": 10,
        },
        "use_terminal_cost": False,
    },
    constrained_env=False,
    is_tracking=False,
    dt=0.01,
)

runner.run()
