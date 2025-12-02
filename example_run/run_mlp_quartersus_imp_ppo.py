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
result_path = "D:/Project/GOPS/results/simu_quarter_sus_vimp/"
runner = PolicyRunnerCustom(
    log_policy_dir_list=[result_path+"PPO_251129-201253",
                         result_path+"PPO_251130-135327",
                         result_path+"TD3_251201-214832",],
    trained_policy_iteration_list=['936_opt','356_opt','6654_opt'],
    is_init_info=True,
    init_info={"init_state": [0.0, 0.0, 0.0, 0.0], "ref_time": 0.0,
               "ref_num": 3}, # ref_num = [0, 1, 2,..., 7]
    save_render=False,
    legend_list=["936_PPO",'356_PPO','6654_TD3'],
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
    use_unified_env_config=True,  # 启用统一环境配置
    eval_G0=0.001024,  # 指定评估时使用的固定G0值 (Class A)
    eval_max_step=10000,  # 指定评估时的最大步长，增加采样点数以提高频率分辨率
)

runner.run()
