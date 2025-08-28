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


from gops.sys_simulator.sys_run import PolicyRunner
import numpy as np
result_path = "D:/Project/GOPS/results/simu_quarter_sus_win/"
runner = PolicyRunner(
    log_policy_dir_list=[result_path+"DDPG_250827-150043",
                         result_path+"DDPG_250827-150043",
                         result_path+"DDPG_250827-150043",],
    trained_policy_iteration_list=["51220_opt","67500",'77000'],
    is_init_info=True,
    init_info={"init_state": [0.0, 0.0, 0.0], "ref_time": 0.0,
               "ref_num": 3}, # ref_num = [0, 1, 2,..., 7]
    save_render=False,
    legend_list=["51220_opt","67500",'77000'],
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
