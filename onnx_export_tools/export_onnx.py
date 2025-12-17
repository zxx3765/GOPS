#  Copyright (c). All Rights Reserved.
#  GOPS ONNX Export Tool
#  独立的ONNX导出工具，用于将训练好的策略网络导出为ONNX格式
#
#  描述: 将GOPS训练的策略网络导出为ONNX格式，提高Simulink仿真性能
#  创建日期: 2024年

import contextlib
import torch
import torch.nn as nn
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
import os


def check_onnx_compatibility(model: nn.Module, example_obs: torch.Tensor, onnx_path: str = None):
    """
    检查模型是否可以导出为ONNX格式，并验证导出的模型是否有效。

    Args:
        model: PyTorch模型
        example_obs: 示例观测数据（必须是2维batch格式）
        onnx_path: 可选的ONNX文件路径，用于测试

    Returns:
        dict: 包含验证结果的字典，包括输入、PyTorch输出和ONNX输出
    """
    try:
        # 确保example_obs是2维的（batch格式），这对ONNX导出至关重要
        if example_obs.dim() == 1:
            example_obs = example_obs.unsqueeze(0)

        with _module_inference(model):
            inference_helper = _InferenceHelper(model)

            # 如果没有提供路径，创建临时路径
            temp_path = onnx_path or "temp_model.onnx"

            # 导出到ONNX
            torch.onnx.export(
                inference_helper,
                example_obs,
                temp_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'},
                             'output': {0: 'batch_size'}}
            )

            # 验证导出的ONNX模型
            onnx_model = onnx.load(temp_path)
            onnx.checker.check_model(onnx_model)

            # 使用ONNX Runtime测试推理
            ort_session = ort.InferenceSession(temp_path)
            ort_inputs = {ort_session.get_inputs()[0].name: example_obs.numpy()}
            ort_outs = ort_session.run(None, ort_inputs)

            # 比较输出结果
            with torch.no_grad():
                torch_out = inference_helper(example_obs)
                torch_out_np = torch_out.numpy()

            if not np.allclose(torch_out_np, ort_outs[0], rtol=1e-3, atol=1e-3):
                raise RuntimeError("ONNX模型输出与PyTorch模型输出不匹配")

            # 如果是临时文件，清理它
            if not onnx_path and os.path.exists(temp_path):
                os.remove(temp_path)

            print("✓ ONNX兼容性检查通过")

            # 返回验证结果
            return {
                'input': example_obs.numpy(),
                'pytorch_output': torch_out_np,
                'onnx_output': ort_outs[0],
                'max_diff': np.max(np.abs(torch_out_np - ort_outs[0])),
                'passed': True
            }

    except Exception as e:
        raise RuntimeError(
            f"模型无法导出为ONNX格式或存在兼容性问题: {str(e)}"
        ) from e


def export_onnx_model(model: nn.Module, example_obs: torch.Tensor, path: str, opset_version: int = 11):
    """
    将模型导出为ONNX格式，用于Simulink。

    Args:
        model: PyTorch模型
        example_obs: 示例观测数据（必须是2维batch格式）
        path: ONNX文件保存路径
        opset_version: ONNX操作集版本
    """
    # 确保example_obs是2维的（batch格式），这对ONNX导出至关重要
    if example_obs.dim() == 1:
        example_obs = example_obs.unsqueeze(0)

    with _module_inference(model):
        inference_helper = _InferenceHelper(model)

        print(f"正在导出ONNX模型到: {path}")
        
        torch.onnx.export(
            inference_helper,
            example_obs,
            path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'},
                         'output': {0: 'batch_size'}},
            verbose=False
        )
        
        # 验证导出的模型
        onnx_model = onnx.load(path)
        onnx.checker.check_model(onnx_model)
        
        # 显示模型信息
        print(f"✓ 模型成功导出为ONNX格式: {path}")
        print(f"  - 输入名称: {[inp.name for inp in onnx_model.graph.input]}")
        print(f"  - 输出名称: {[out.name for out in onnx_model.graph.output]}")
        print(f"  - ONNX版本: {opset_version}")


@contextlib.contextmanager
def _module_inference(module: nn.Module):
    """上下文管理器，确保模型处于推理模式"""
    training = module.training
    module.train(False)
    try:
        yield
    finally:
        module.train(training)


class _InferenceHelper(nn.Module):
    """推理辅助类，包装原始模型以便导出"""
    
    def __init__(self, model):
        super().__init__()

        from gops.apprfunc.mlp import Action_Distribution

        assert isinstance(model, nn.Module) and isinstance(
            model, Action_Distribution
        ), (
            "模型必须继承自nn.Module和Action_Distribution类。"
            f"当前类型: {model.__class__.__mro__}"
        )
        self.model = model

    def forward(self, obs: torch.Tensor):
        # 确保输入是批次格式
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        logits = self.model(obs)
        act_dist = self.model.get_act_dist(logits)
        mode = act_dist.mode()

        # 始终返回batch格式的输出（2维），这是ONNX模型的标准做法
        return mode


def validate_onnx_model(onnx_path: str, example_input: np.ndarray, pytorch_model=None):
    """
    验证ONNX模型是否可以正确运行，并可选地与PyTorch模型输出对比

    Args:
        onnx_path: ONNX模型路径
        example_input: 示例输入数据（numpy数组，必须是2维batch格式）
        pytorch_model: 可选的PyTorch模型，用于对比输出

    Returns:
        dict: 包含验证结果的字典
    """
    try:
        # 确保example_input是2维的（batch格式）
        if example_input.ndim == 1:
            example_input = example_input.reshape(1, -1)

        # 加载ONNX模型
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        # 使用ONNX Runtime进行推理
        ort_session = ort.InferenceSession(onnx_path)

        # 获取输入输出信息
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name

        print(f"模型输入名称: {input_name}")
        print(f"模型输出名称: {output_name}")
        print(f"输入形状: {ort_session.get_inputs()[0].shape}")
        print(f"输出形状: {ort_session.get_outputs()[0].shape}")

        # 运行推理
        ort_inputs = {input_name: example_input}
        ort_outputs = ort_session.run([output_name], ort_inputs)

        print(f"✓ ONNX模型验证成功")
        print(f"  - 输入形状: {example_input.shape}")
        print(f"  - 输出形状: {ort_outputs[0].shape}")

        result = {
            'input': example_input,
            'onnx_output': ort_outputs[0],
            'input_shape': example_input.shape,
            'output_shape': ort_outputs[0].shape
        }

        # 如果提供了PyTorch模型，进行对比
        if pytorch_model is not None:
            with _module_inference(pytorch_model):
                inference_helper = _InferenceHelper(pytorch_model)
                example_obs = torch.from_numpy(example_input).float()
                with torch.no_grad():
                    pytorch_out = inference_helper(example_obs)
                    pytorch_out_np = pytorch_out.numpy()

                result['pytorch_output'] = pytorch_out_np
                result['max_diff'] = np.max(np.abs(pytorch_out_np - ort_outputs[0]))
                result['passed'] = np.allclose(pytorch_out_np, ort_outputs[0], rtol=1e-3, atol=1e-3)

                print(f"  - PyTorch vs ONNX 最大差异: {result['max_diff']}")
                print(f"  - 对比通过: {result['passed']}")

        return result

    except Exception as e:
        print(f"✗ ONNX模型验证失败: {str(e)}")
        raise