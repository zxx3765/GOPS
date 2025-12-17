#!/usr/bin/env python3
"""
Simple unit test for G0 randomization functionality
G0随机化功能的简单单元测试
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import unittest
import numpy as np
from gops.create_pkg.create_env import create_env

def get_G0_from_env(env):
    """
    从环境中获取当前G0值，处理环境包装层问题
    """
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
        raise

class TestG0Randomization(unittest.TestCase):
    
    def setUp(self):
        """设置测试环境"""
        self.env_args = {
            "env_id": "simu_quarter_sus_win",
            "Max_step": 50,
            "act_repeat": 10,
            "obs_scaling": [5, 1, 0.03, 0.3],
            "act_scaling": 0.001,
            "rew_scaling": 1,
            "act_max": 1000,
            "punish_done": 0.0,
            "rew_bias": 0,
            "rew_bound": 100.0,
            "rand_bias": [0.01] * 10,
            "rand_center": [0] * 10,
            
            # 物理参数
            "Cs": 2000.0, "Ks": 20000.0, "Ms": 400.0, "Mu": 50.0, "Kt": 200000.0,
            
            # G0参数
            "G0": 0.001024, "G0_min": 0.0001, "G0_max": 0.002,
            "f0": 0.1, "u": 20.0, "as_max": 1, "deflec_max": 0.04,
            "road_seed": 827538, "Road_Type": "Random",
            
            # 初始状态和奖励参数
            "init_state_max": [0.01, 0.1, 0.01, 0.1],
            "init_state_min": [-0.01, -0.1, -0.01, -0.1],
            "punish_Q_acc_s": 7, "punish_Q_flec": 1, "punish_Q_F": 1,
            "punish_Q_flec_t": 1, "punish_Q_acc_s_h": 2.5,
            "punish_b_deflec": 0.04, "punish_Q_b_defelc": -100,
            
            # 环境设置
            "action_type": "continu", "is_render": False,
            "is_adversary": False, "is_constrained": False,
        }
        
        self.env = create_env(**self.env_args)
    
    def test_random_G0_in_range(self):
        """测试随机G0值是否在指定范围内"""
        print("\nTesting random G0 values...")
        
        for i in range(20):
            self.env.reset()  # 不传init_G0，应该随机
            current_G0 = get_G0_from_env(self.env)
            
            self.assertGreaterEqual(current_G0, self.env_args['G0_min'], 
                                  f"G0 {current_G0} below minimum {self.env_args['G0_min']}")
            self.assertLessEqual(current_G0, self.env_args['G0_max'],
                               f"G0 {current_G0} above maximum {self.env_args['G0_max']}")
        
        print("[OK] Random G0 values within specified range")
    
    def test_specified_G0_values(self):
        """测试指定G0值是否正确设置"""
        print("\nTesting specified G0 values...")
        
        test_values = [0.0005, 0.001024, 0.0015, 0.001999]
        
        for test_G0 in test_values:
            self.env.reset(init_G0=test_G0)
            actual_G0 = get_G0_from_env(self.env)
            
            self.assertAlmostEqual(actual_G0, test_G0, places=8,
                                 msg=f"Expected G0 {test_G0}, got {actual_G0}")
        
        print("[OK] Specified G0 values set correctly")
    
    def test_G0_randomization_diversity(self):
        """测试G0随机化的多样性"""
        print("\nTesting G0 randomization diversity...")
        
        G0_values = []
        for i in range(50):
            self.env.reset()
            G0_values.append(get_G0_from_env(self.env))
        
        # 检查是否有足够的多样性（不应该所有值都相同）
        unique_values = len(set([round(g, 8) for g in G0_values]))
        self.assertGreater(unique_values, 10, 
                          f"Only {unique_values} unique G0 values in 50 samples")
        
        # 检查分布是否合理
        mean_G0 = np.mean(G0_values)
        expected_mean = (self.env_args['G0_min'] + self.env_args['G0_max']) / 2
        
        # 允许一定的误差范围
        self.assertLess(abs(mean_G0 - expected_mean), 0.0005,
                       f"Mean G0 {mean_G0} too far from expected {expected_mean}")
        
        print(f"[OK] G0 randomization shows good diversity ({unique_values} unique values)")
        print(f"[OK] Mean G0: {mean_G0:.6f}, Expected: {expected_mean:.6f}")
    
    def test_environment_functionality(self):
        """测试环境基本功能是否正常"""
        print("\nTesting environment functionality...")
        
        # 测试随机G0
        obs1 = self.env.reset()
        if isinstance(obs1, tuple):
            obs1 = obs1[0]
        self.assertEqual(obs1.shape, (4,), "Observation shape incorrect")
        
        # 测试指定G0
        obs2 = self.env.reset(init_G0=0.001024)
        if isinstance(obs2, tuple):
            obs2 = obs2[0]
        self.assertEqual(obs2.shape, (4,), "Observation shape incorrect with specified G0")
        
        # 测试step功能
        action = np.array([0.1])
        step_result = self.env.step(action)
        
        # 处理不同版本的gym/gymnasium返回值格式
        if len(step_result) == 5:
            obs, reward, done, truncated, info = step_result
        elif len(step_result) == 4:
            obs, reward, done, info = step_result
        else:
            raise ValueError(f"Unexpected step result length: {len(step_result)}")
        
        self.assertEqual(obs.shape, (4,), "Step observation shape incorrect")
        self.assertIsInstance(reward, (int, float), "Reward should be numeric")
        self.assertIsInstance(done, bool, "Done should be boolean")
        
        print("[OK] Environment functionality works correctly")
    
    def test_G0_persistence_during_episode(self):
        """测试G0值在episode期间是否保持不变"""
        print("\nTesting G0 persistence during episode...")
        
        # 设置特定G0值
        self.env.reset(init_G0=0.001234)
        initial_G0 = get_G0_from_env(self.env)
        
        # 运行几步
        for step in range(5):
            action = np.array([0.05 * step])
            self.env.step(action)
            current_G0 = get_G0_from_env(self.env)
            
            self.assertAlmostEqual(current_G0, initial_G0, places=8,
                                 msg=f"G0 changed during episode: {initial_G0} -> {current_G0}")
        
        print("[OK] G0 value remains constant during episode")

if __name__ == '__main__':
    print("=" * 60)
    print("G0 Randomization Unit Tests")
    print("=" * 60)
    
    unittest.main(verbosity=2)