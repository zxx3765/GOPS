#!/usr/bin/env python3
#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Test script for G0 randomization functionality
#  测试G0随机化功能的脚本

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import matplotlib.pyplot as plt
from gops.create_pkg.create_env import create_env

def get_G0_from_env(env):
    """
    从环境中获取当前G0值，处理环境包装层问题
    """
    # 尝试不同的访问路径
    try:
        # 方法1: 直接从环境获取
        if hasattr(env, 'get_current_G0'):
            return env.get_current_G0()
        
        # 方法2: 通过env属性链访问
        current_env = env
        while hasattr(current_env, 'env'):
            current_env = current_env.env
            if hasattr(current_env, 'get_current_G0'):
                return current_env.get_current_G0()
        
        # 方法3: 直接访问model_class
        if hasattr(current_env, 'model_class'):
            return current_env.model_class.quarter_sus_win_InstP.G0
            
        # 方法4: 通过env.env...链式访问
        test_env = env
        for i in range(10):  # 最多尝试10层包装
            try:
                if hasattr(test_env, 'model_class'):
                    return test_env.model_class.quarter_sus_win_InstP.G0
                if hasattr(test_env, 'env'):
                    test_env = test_env.env
                else:
                    break
            except AttributeError:
                break
                
        raise AttributeError("Cannot find G0 value in environment")
        
    except Exception as e:
        print(f"Error accessing G0 from environment: {e}")
        print(f"Environment type: {type(env)}")
        print(f"Available attributes: {[attr for attr in dir(env) if not attr.startswith('_')]}")
        raise

def test_G0_randomization():
    """
    测试G0随机化功能
    """
    print("=" * 60)
    print("Testing G0 Randomization Functionality")
    print("=" * 60)
    
    # 环境配置参数
    env_args = {
        "env_id": "simu_quarter_sus_win",
        "Max_step": 100,  # 短步数用于快速测试
        "act_repeat": 10,
        "obs_scaling": [5, 1, 0.03, 0.3],
        "act_scaling": 0.001,
        "rew_scaling": 1,
        "act_max": 1000,
        "punish_done": 0.0,
        "rew_bias": 0,
        "rew_bound": 100.0,
        "rand_bias": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        "rand_center": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        
        # 物理参数
        "Cs": 2000.0,
        "Ks": 20000.0,
        "Ms": 400.0,
        "Mu": 50.0,
        "Kt": 200000.0,
        
        # G0参数设置
        "G0": 0.001024,      # 默认G0值
        "G0_min": 0.0001,    # 最小G0值
        "G0_max": 0.002,     # 最大G0值
        "f0": 0.1,
        "u": 20.0,
        "as_max": 1,
        "deflec_max": 0.04,
        "road_seed": 827538,
        "Road_Type": "Random",
        
        # 初始状态范围
        "init_state_max": [0.01, 0.1, 0.01, 0.1],
        "init_state_min": [-0.01, -0.1, -0.01, -0.1],
        
        # 奖励函数参数
        "punish_Q_acc_s": 7,
        "punish_Q_flec": 1,
        "punish_Q_F": 1,
        "punish_Q_flec_t": 1,
        "punish_Q_acc_s_h": 2.5,
        "punish_b_deflec": 0.04,
        "punish_Q_b_defelc": -100,
        
        # 环境设置
        "action_type": "continu",
        "is_render": False,
        "is_adversary": False,
        "is_constrained": False,
    }
    
    # 创建环境
    print("Creating environment...")
    env = create_env(**env_args)
    print(f"Environment created successfully!")
    print(f"G0 range: [{env_args['G0_min']}, {env_args['G0_max']}]")
    print()
    
    # 测试1: 随机G0值测试（训练模式）
    print("Test 1: Random G0 values (Training mode)")
    print("-" * 40)
    
    random_G0_values = []
    n_tests = 10
    
    for i in range(n_tests):
        # 不传入init_G0，应该随机生成
        obs = env.reset()
        
        # 获取当前G0值（从Simulink模型中读取）
        current_G0 = get_G0_from_env(env)
        random_G0_values.append(current_G0)
        
        print(f"  Reset {i+1:2d}: G0 = {current_G0:.6f}")
        
        # 验证G0在指定范围内
        assert env_args['G0_min'] <= current_G0 <= env_args['G0_max'], \
            f"G0 {current_G0} not in range [{env_args['G0_min']}, {env_args['G0_max']}]"
    
    print(f"  Random G0 values: min={min(random_G0_values):.6f}, max={max(random_G0_values):.6f}")
    print(f"  All values within specified range: OK")
    print()
    
    # 测试2: 指定G0值测试（评估模式）
    print("Test 2: Specified G0 values (Evaluation mode)")
    print("-" * 40)
    
    test_G0_values = [0.0005, 0.001024, 0.0015]
    
    for test_G0 in test_G0_values:
        # 传入指定的init_G0值
        obs = env.reset(init_G0=test_G0)
        
        # 获取当前G0值
        current_G0 = get_G0_from_env(env)
        
        print(f"  Specified G0 = {test_G0:.6f}, Actual G0 = {current_G0:.6f}")
        
        # 验证G0值是否正确设置
        assert abs(current_G0 - test_G0) < 1e-8, \
            f"G0 mismatch: expected {test_G0}, got {current_G0}"
    
    print(f"  All specified G0 values set correctly: OK")
    print()
    
    # 测试3: 运行几步验证环境正常工作
    print("Test 3: Environment functionality test")
    print("-" * 40)
    
    # 使用指定G0值重置环境
    obs = env.reset(init_G0=0.001024)
    if isinstance(obs, tuple):
        obs = obs[0]  # 取观测值部分
    print(f"  Initial observation shape: {obs.shape}")
    print(f"  Initial observation: {obs}")
    
    # 运行几步
    for step in range(3):
        action = env.action_space.sample()  # 随机动作
        step_result = env.step(action)
        
        # 处理不同版本的gym/gymnasium返回值格式
        if len(step_result) == 5:
            obs, reward, done, truncated, info = step_result
        elif len(step_result) == 4:
            obs, reward, done, info = step_result
            truncated = False
        else:
            raise ValueError(f"Unexpected step result length: {len(step_result)}")
            
        print(f"  Step {step+1}: action={action[0]:.3f}, reward={reward:.3f}, done={done}")
        
        if done or truncated:
            break
    
    print(f"  Environment runs normally: OK")
    print()
    
    # 测试4: 可视化G0值分布
    print("Test 4: G0 distribution visualization")
    print("-" * 40)
    
    # 收集更多随机G0值用于统计
    large_sample_G0 = []
    n_large_sample = 100
    
    print(f"  Collecting {n_large_sample} random G0 samples...")
    for i in range(n_large_sample):
        env.reset()  # 不传init_G0，随机生成
        current_G0 = get_G0_from_env(env)
        large_sample_G0.append(current_G0)
    
    # 统计信息
    mean_G0 = np.mean(large_sample_G0)
    std_G0 = np.std(large_sample_G0)
    min_G0 = np.min(large_sample_G0)
    max_G0 = np.max(large_sample_G0)
    
    print(f"  Statistics for {n_large_sample} samples:")
    print(f"    Mean: {mean_G0:.6f}")
    print(f"    Std:  {std_G0:.6f}")
    print(f"    Min:  {min_G0:.6f}")
    print(f"    Max:  {max_G0:.6f}")
    print(f"    Expected range: [{env_args['G0_min']:.6f}, {env_args['G0_max']:.6f}]")
    
    # 保存分布图
    plt.figure(figsize=(10, 6))
    plt.hist(large_sample_G0, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(env_args['G0_min'], color='red', linestyle='--', label=f'G0_min = {env_args["G0_min"]}')
    plt.axvline(env_args['G0_max'], color='red', linestyle='--', label=f'G0_max = {env_args["G0_max"]}')
    plt.axvline(mean_G0, color='green', linestyle='-', label=f'Mean = {mean_G0:.6f}')
    plt.xlabel('G0 Value')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Random G0 Values (n={n_large_sample})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图片
    plot_path = "G0_distribution_test.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Distribution plot saved to: {plot_path}")
    print()
    
    # 测试总结
    print("=" * 60)
    print("ALL TESTS PASSED! [OK]")
    print("=" * 60)
    print("Summary:")
    print(f"  [OK] Random G0 generation works correctly")
    print(f"  [OK] Specified G0 setting works correctly") 
    print(f"  [OK] Environment functionality preserved")
    print(f"  [OK] G0 values distributed properly in [{env_args['G0_min']:.6f}, {env_args['G0_max']:.6f}]")
    print()
    print("Usage recommendations:")
    print("  - For training: Use env.reset() for random G0")
    print("  - For evaluation: Use env.reset(init_G0=specific_value)")
    print(f"  - Class A road range: G0 in [0.0001, 0.002]")
    print("=" * 60)

if __name__ == "__main__":
    test_G0_randomization()