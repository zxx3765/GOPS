#  Copyright (c). All Rights Reserved.
#  GOPS ONNX Export Tool - 主程序
#  独立的ONNX导出工具主程序
#
#  描述: 主要的ONNX导出运行器，基于原始的Py2slxRunner修改
#  创建日期: 2024年

import argparse
import importlib
import os
import sys
import torch
import numpy as np
from pathlib import Path

# 添加GOPS路径到系统路径
current_dir = Path(__file__).parent
gops_root = current_dir.parent
sys.path.append(str(gops_root))

from gops.create_pkg.create_env import create_env
from gops.create_pkg.create_sampler import create_sampler
from gops.utils.common_utils import get_args_from_json
from gops.utils.gops_path import camel2underline

from export_onnx import check_onnx_compatibility, export_onnx_model, validate_onnx_model


class Py2ONNXRunner:
    """
    GOPS工具：将训练好的策略导出为ONNX格式用于Simulink仿真
    
    Args:
        log_policy_dir_list: 训练策略的加载路径列表
        trained_policy_iteration_list: 对应的训练策略迭代步数列表
        export_controller_name: 导出控制器名称列表
        save_path: 保存路径列表，建议与Simulink项目文件在同一目录
        export_format: 导出格式，'onnx'或'torchscript'，默认'onnx'
        opset_version: ONNX操作集版本，默认11
    """

    def __init__(
        self,
        log_policy_dir_list: list,
        trained_policy_iteration_list: list,
        export_controller_name: list,
        save_path: list,
        export_format: str = "onnx",
        opset_version: int = 11,
    ) -> None:
        self.log_policy_dir_list = log_policy_dir_list
        self.trained_policy_iteration_list = trained_policy_iteration_list
        self.export_controller_name = export_controller_name
        self.save_path = save_path
        self.export_format = export_format.lower()
        self.opset_version = opset_version
        
        self.args = None
        self.policy_num = len(self.log_policy_dir_list)
        
        # 验证输入参数
        if self.policy_num != len(self.trained_policy_iteration_list):
            raise RuntimeError(
                "策略数量与策略迭代步数数量不匹配"
            )
        
        if self.export_format not in ["onnx", "torchscript"]:
            raise ValueError(
                f"不支持的导出格式: {self.export_format}. 支持的格式: 'onnx', 'torchscript'"
            )

        self.args_list = []
        self.algorithm_list = []
        self._load_all_args()

    @staticmethod
    def _load_args(log_policy_dir: str) -> dict:
        """从配置文件加载参数"""
        json_path = os.path.join(log_policy_dir, "config.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"配置文件不存在: {json_path}")
            
        parser = argparse.ArgumentParser()
        args_dict = vars(parser.parse_args([]))
        args = get_args_from_json(json_path, args_dict)
        return args

    def _load_all_args(self):
        """加载所有策略的参数"""
        for i in range(self.policy_num):
            log_policy_dir = self.log_policy_dir_list[i]
            args = self._load_args(log_policy_dir)
            self.args_list.append(args)
            self.algorithm_list.append(args["algorithm"])

    def _load_env(self):
        """加载环境"""
        env = create_env(**self.args)
        self.args["action_high_limit"] = env.action_space.high
        self.args["action_low_limit"] = env.action_space.low
        return env

    def _load_policy(self, log_policy_dir: str, trained_policy_iteration: str):
        """加载训练好的策略"""
        # 创建策略
        alg_name = self.args["algorithm"]
        alg_file_name = camel2underline(alg_name)
        
        try:
            file = importlib.import_module("gops.algorithm." + alg_file_name)
            ApproxContainer = getattr(file, "ApproxContainer")
        except (ImportError, AttributeError) as e:
            raise ImportError(f"无法导入算法 {alg_name}: {e}")
            
        networks = ApproxContainer(**self.args)
        print(f"✓ 成功创建 {alg_name} 策略")

        # 加载训练好的策略
        log_path = os.path.join(
            log_policy_dir, "apprfunc", f"apprfunc_{trained_policy_iteration}.pkl"
        )
        
        if not os.path.exists(log_path):
            raise FileNotFoundError(f"策略文件不存在: {log_path}")
            
        networks.load_state_dict(torch.load(log_path, map_location='cpu'))
        print(f"✓ 成功加载 {alg_name} 策略")
        return networks

    def _load_sampler(self):
        """加载采样器"""
        sampler = create_sampler(**self.args)
        return sampler

    def _export_single_policy(self, policy_idx: int):
        """导出单个策略"""
        log_policy_dir = self.log_policy_dir_list[policy_idx]
        trained_policy_iteration = self.trained_policy_iteration_list[policy_idx]
        controller_name = self.export_controller_name[policy_idx]
        save_dir = self.save_path[policy_idx]

        print(f"\n{'='*60}")
        print(f"正在处理策略 {policy_idx + 1}/{self.policy_num}")
        print(f"策略路径: {log_policy_dir}")
        print(f"迭代步数: {trained_policy_iteration}")
        print(f"控制器名称: {controller_name}")
        print(f"保存路径: {save_dir}")
        print(f"{'='*60}")

        self.args = self.args_list[policy_idx]
        
        # 加载组件
        networks = self._load_policy(log_policy_dir, trained_policy_iteration)
        sampler = self._load_sampler()
        model = networks.policy

        # 获取示例观测
        example_obs_row = sampler.env.reset()[0]
        example_obs = torch.from_numpy(example_obs_row).float()
        
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        if self.export_format == "onnx":
            save_path = os.path.join(save_dir, f"{controller_name}.onnx")
            self._export_onnx_model(model, example_obs, save_path)
        else:  # torchscript
            save_path = os.path.join(save_dir, f"{controller_name}.pt")
            self._export_torchscript_model(model, example_obs, save_path)

    def _export_onnx_model(self, model, example_obs, save_path):
        """导出ONNX模型"""
        print(f"检查ONNX兼容性...")
        check_onnx_compatibility(model, example_obs)
        
        print(f"导出ONNX模型...")
        export_onnx_model(model, example_obs, save_path, self.opset_version)
        
        # 验证导出的模型
        print(f"验证导出的ONNX模型...")
        validate_onnx_model(save_path, example_obs.numpy())

    def _export_torchscript_model(self, model, example_obs, save_path):
        """导出TorchScript模型（保留原功能）"""
        from gops.env.py2slx_tools.export import check_jit_compatibility, export_model
        
        print(f"检查JIT兼容性...")
        check_jit_compatibility(model, example_obs)
        
        print(f"导出TorchScript模型...")
        export_model(model, example_obs, save_path)

    def export_policies(self):
        """导出所有策略"""
        print(f"开始导出 {self.policy_num} 个策略为 {self.export_format.upper()} 格式")
        
        for i in range(self.policy_num):
            try:
                self._export_single_policy(i)
                print(f"✓ 策略 {i + 1} 导出成功")
            except Exception as e:
                print(f"✗ 策略 {i + 1} 导出失败: {str(e)}")
                raise
        
        print(f"\n{'='*60}")
        print(f"所有策略导出完成!")
        print(f"导出格式: {self.export_format.upper()}")
        print(f"总计: {self.policy_num} 个策略")
        print(f"{'='*60}")


def main():
    """主函数，支持命令行调用"""
    parser = argparse.ArgumentParser(description='GOPS策略ONNX导出工具')
    parser.add_argument('--policy_dir', type=str, required=True,
                       help='策略目录路径')
    parser.add_argument('--iteration', type=str, required=True,
                       help='策略迭代步数')
    parser.add_argument('--name', type=str, required=True,
                       help='导出控制器名称')
    parser.add_argument('--save_path', type=str, required=True,
                       help='保存路径')
    parser.add_argument('--format', type=str, default='onnx',
                       choices=['onnx', 'torchscript'],
                       help='导出格式')
    parser.add_argument('--opset_version', type=int, default=11,
                       help='ONNX操作集版本')
    
    args = parser.parse_args()
    
    runner = Py2ONNXRunner(
        log_policy_dir_list=[args.policy_dir],
        trained_policy_iteration_list=[args.iteration],
        export_controller_name=[args.name],
        save_path=[args.save_path],
        export_format=args.format,
        opset_version=args.opset_version,
    )
    
    runner.export_policies()


if __name__ == "__main__":
    main()