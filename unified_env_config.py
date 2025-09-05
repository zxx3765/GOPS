#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Unified environment configuration for consistent evaluation

"""
统一环境配置文件
用于在运行评估时确保所有策略使用相同的环境参数
"""

# Quarter-Car Suspension 统一环境参数
UNIFIED_QUARTER_SUSPENSION_CONFIG = {
    # 物理参数
    "Cs": 2000.0,      # 阻尼系数 (N⋅s/m)
    "Ks": 20000.0,     # 弹簧系数 (N/m) 
    "Ms": 400.0,       # 簧载质量 (kg)
    "Mu": 50.0,        # 非簧载质量 (kg) - 注意：统一为50.0而不是40.0
    "Kt": 200000.0,    # 轮胎刚度 (N/m)
    
    # 道路参数
    "G0": 0.001256,    # 随机路面参数 (Class A)
    "f0": 0.1,
    "u": 20.0,         # 车速 (m/s)
    "Road_Type": "Random",  # 道路类型: Sine/Chirp/Random/Bump
    "road_seed": 827538,    # 道路随机种子
    
    # 控制约束
    "act_max": 1000,    # 最大控制力 (N)
    "as_max": 1,        # 簧载质量最大加速度 (m/s²)
    "deflec_max": 0.04, # 最大悬架变形 (m)
    
    # 仿真参数
    "Max_step": 2000,   # 每个episode最大步数
    "act_repeat": 10,   # 动作重复次数
    "dt": 0.01,         # 时间步长 (s)
    
    # 缩放参数
    "obs_scaling": [5, 1, 0.03, 0.3],  # 观测缩放
    "act_scaling": 0.001,               # 动作缩放
    "rew_scaling": 1,                   # 奖励缩放
    
    # 奖励函数权重
    "punish_Q_acc_s": 7,      # 簧载质量加速度惩罚权重
    "punish_Q_flec": 1,       # 悬架变形惩罚权重  
    "punish_Q_F": 1,          # 控制力惩罚权重
    "punish_Q_flec_t": 1,     # 轮胎变形惩罚权重
    "punish_Q_acc_s_h": 2.5,  # 高频簧载质量加速度惩罚权重
    "punish_b_deflec": 0.01,  # 变形边界参数
    "punish_Q_b_defelc": -100, # 边界惩罚权重
    
    # 初始状态范围 [xs0, vs0, xu0, vu0]
    "init_state_max": [0.01, 0.1, 0.01, 0.1],
    "init_state_min": [-0.01, -0.1, -0.01, -0.1],
    
    # 随机化参数
    "rand_bias": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    "rand_center": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    
    # 其他环境设置
    "action_type": "continu",
    "is_render": False,
    "is_adversary": False,
    "is_constrained": False,
    "punish_done": 0.0,
    "rew_bias": 0,
    "rew_bound": 100.0,
}

def get_unified_config(env_id="simu_quarter_sus_win"):
    """
    获取指定环境的统一配置参数
    
    Args:
        env_id (str): 环境ID
        
    Returns:
        dict: 统一的环境配置参数
    """
    if env_id == "simu_quarter_sus_win":
        return UNIFIED_QUARTER_SUSPENSION_CONFIG.copy()
    else:
        raise ValueError(f"Unified config for environment '{env_id}' is not implemented yet")

def override_env_args(original_args, env_id="simu_quarter_sus_win"):
    """
    用统一配置覆盖原始环境参数
    
    Args:
        original_args (dict): 原始环境参数
        env_id (str): 环境ID
        
    Returns:
        dict: 更新后的环境参数
    """
    unified_config = get_unified_config(env_id)
    
    # 创建原始参数的副本
    updated_args = original_args.copy()
    
    # 用统一配置覆盖相应参数
    for key, value in unified_config.items():
        updated_args[key] = value
    
    return updated_args