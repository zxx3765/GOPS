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
result_path = "D:/Project/GOPS/results/simu_quarter_sus_imp_force/"
runner = PolicyRunnerCustom(
    log_policy_dir_list=[
                         result_path+"TD3_251202-154807",
                         result_path+"TD3_251210-193618",
                         result_path+"TD3_251210-135209",
                         result_path+"TD3_251212-093413",
                         result_path+"TD3_251215-144815",],
    trained_policy_iteration_list=['46071_opt','475536_opt','345892_opt','101421_opt','270374_opt'],
    is_init_info=True,
    init_info={"init_state": [0.0, 0.0, 0.0, 0.0], "ref_time": 0.0,
               "ref_num": 3}, # ref_num = [0, 1, 2,..., 7]
    save_render=False,
    legend_list=['46071_TD3','475536_TD3','345892_TD3','101421_TD3','270374_TD3'],
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
