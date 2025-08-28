#  Copyright (c). All Rights Reserved.
#  GOPS ONNX Export Tool - 使用示例
#  
#  描述: 展示如何使用独立的ONNX导出工具，支持命令行参数调用
#  创建日期: 2024年

"""
使用说明:
方法1 - 命令行参数调用（推荐用于MATLAB调用）:
python example_usage.py --policy_dir "D:\path\to\policy" --iteration "6078_opt" --controller_name "NN_controller" --save_path "D:\save\path"

方法2 - 修改脚本内参数（传统方法）:
1. 修改下面的默认参数设置
2. 运行此脚本导出ONNX模型
3. 将生成的.onnx文件导入到Simulink中使用

MATLAB调用示例:
system('python example_usage.py --policy_dir "D:\Project\GOPS\results\simu_quarter_sus_win\DDPG_250825-150421" --iteration "6078_opt" --controller_name "NN_controller_PPO2" --save_path "D:\Project\SynologyDrive\imp_fcn_cal\RL_model"')

参数说明:
- --policy_dir: 训练策略的路径
- --iteration: 策略迭代步数
- --controller_name: 导出控制器名称
- --save_path: 保存路径
- --export_format: 导出格式（默认onnx）
- --opset_version: ONNX操作集版本（默认11）
"""

import argparse
from py2onnx import Py2ONNXRunner

def parse_arguments():
    parser = argparse.ArgumentParser(description='GOPS策略ONNX导出工具')
    
    parser.add_argument('--policy_dir', 
                       type=str, 
                       default=r"D:\Project\GOPS\results\simu_quarter_sus_win\DDPG_250825-150421",
                       help='训练策略的路径')
    
    parser.add_argument('--iteration', 
                       type=str, 
                       default="6078_opt",
                       help='策略迭代步数')
    
    parser.add_argument('--controller_name', 
                       type=str, 
                       default="NN_6078opt",
                       help='导出控制器名称')
    
    parser.add_argument('--save_path', 
                       type=str, 
                       default=r"D:\Project\SynologyDrive\imp_fcn_cal\RL_model",
                       help='保存路径')
    
    parser.add_argument('--export_format', 
                       type=str, 
                       default="onnx",
                       choices=['onnx', 'torchscript'],
                       help='导出格式（默认onnx）')
    
    parser.add_argument('--opset_version', 
                       type=int, 
                       default=11,
                       help='ONNX操作集版本（默认11）')
    
    return parser.parse_args()

# ================== 默认参数配置区域 ==================
# 如果不使用命令行参数，将使用以下默认参数

# 训练策略路径（可以是多个策略）
default_log_policy_dir_list = [
    r"D:\Project\GOPS\results\simu_quarter_sus_win\DDPG_250825-150421",
    r"D:\Project\GOPS\results\simu_quarter_sus_win\DDPG_250825-220417",
]

# 对应的策略迭代步数
default_trained_policy_iteration_list = [
    "6078_opt",
    "28000",  # 如果有第二个策略，取消注释并修改迭代步数
]

# 导出的控制器名称
default_export_controller_name = [
    "NN_6078opt",
    "NN_28000",  # 如果有第二个策略，取消注释并修改名称
]

# 保存路径（建议与您的Simulink项目文件在同一目录）
default_save_path = [
    r"D:\Project\SynologyDrive\imp_fcn_cal\RL_model",
    r"D:\Project\SynologyDrive\imp_fcn_cal\RL_model",  # 如果有第二个策略，取消注释并修改路径
]

# 导出格式设置
default_export_format = "onnx"  # 可选: "onnx" 或 "torchscript"
default_opset_version = 11      # ONNX操作集版本，推荐11或更高

# ================== 执行导出 ==================

if __name__ == "__main__":
    print("GOPS策略ONNX导出工具")
    print("=" * 60)
    
    # 解析命令行参数
    args = parse_arguments()
    
    print(f"使用参数:")
    print(f"  策略路径: {args.policy_dir}")
    print(f"  迭代步数: {args.iteration}")
    print(f"  控制器名称: {args.controller_name}")
    print(f"  保存路径: {args.save_path}")
    print(f"  导出格式: {args.export_format}")
    print(f"  OPSET版本: {args.opset_version}")
    print("-" * 60)
    
    # 使用命令行参数或默认参数
    log_policy_dir_list = [args.policy_dir]
    trained_policy_iteration_list = [args.iteration]
    export_controller_name = [args.controller_name]
    save_path = [args.save_path]
    export_format = args.export_format
    opset_version = args.opset_version
    
    # 创建导出器
    runner = Py2ONNXRunner(
        log_policy_dir_list=log_policy_dir_list,
        trained_policy_iteration_list=trained_policy_iteration_list,
        export_controller_name=export_controller_name,
        save_path=save_path,
        export_format=export_format,
        opset_version=opset_version,
    )
    
    # 执行导出
    try:
        runner.export_policies()
        print("\n🎉 导出完成！")
        print("\n接下来的步骤:")
        print("1. 检查生成的.onnx文件")
        print("2. 在Simulink中导入ONNX模型")
        print("3. 配置输入输出接口")
        print("4. 运行仿真测试性能")
        
    except Exception as e:
        print(f"\n❌ 导出失败: {str(e)}")
        print("\n请检查:")
        print("1. 策略路径是否正确")
        print("2. 策略文件是否存在")
        print("3. 依赖包是否已安装")
        print("4. GOPS环境是否正确配置")