# GOPS ONNX Export Tool

独立的ONNX导出工具，用于将GOPS训练的策略网络导出为ONNX格式，提高Simulink仿真性能。

## 功能特点

- ✅ 支持导出ONNX格式，提升Simulink仿真速度
- ✅ 保留原有TorchScript导出功能
- ✅ 自动兼容性检查和模型验证
- ✅ 支持批量导出多个策略
- ✅ 独立于原项目，不修改原始代码

## 文件说明

### Python脚本
- `export_onnx.py`: ONNX导出核心功能模块
- `py2onnx.py`: 主程序运行器
- `example_usage.py`: Python使用示例脚本
- `requirements.txt`: Python依赖包列表

### MATLAB脚本  
- `gops_onnx_validation_bridge.m`: Simulink中的ONNX模型桥接函数
- `gops_onnx_predict.m`: 简化的ONNX预测函数
- `gops_onnx_test.m`: ONNX模型测试和验证脚本
- `matlab_usage_example.m`: MATLAB使用示例和代码生成器
- `load_onnx_compatible.m`: 版本兼容的ONNX加载函数
- `test_onnx_compatibility.m`: MATLAB版本兼容性测试脚本

### 文档
- `README.md`: 使用说明（本文件）

## 安装依赖

### Python依赖
```bash
pip install torch torchvision onnx onnxruntime numpy
```

或者使用requirements.txt:

```bash
pip install -r requirements.txt
```

### MATLAB依赖
- MATLAB R2021b或更高版本（推荐R2022a+）
- Deep Learning Toolbox
- **版本兼容性说明**:
  - R2022a+: 自动使用`importNetworkFromONNX`（推荐）
  - R2021b-R2022a: 支持新旧函数自动切换
  - R2021a及以前: 使用`importONNXNetwork`（已弃用但仍可用）

## 快速开始

### 1. 使用示例脚本

1. 打开 `example_usage.py`
2. 修改参数配置区域的路径和设置
3. 运行脚本:

```bash
python example_usage.py
```

### 2. 命令行使用

```bash
python py2onnx.py --policy_dir "D:\path\to\policy" --iteration "520_opt" --name "controller_name" --save_path "D:\path\to\save" --format onnx
```

### 3. 程序化使用

```python
from py2onnx import Py2ONNXRunner

runner = Py2ONNXRunner(
    log_policy_dir_list=[r"D:\path\to\your\policy"],
    trained_policy_iteration_list=["520_opt"],
    export_controller_name=["NN_controller_ONNX"],
    save_path=[r"D:\path\to\save"],
    export_format="onnx",
    opset_version=11,
)

runner.export_policies()
```

## 参数说明

- `log_policy_dir_list`: 训练策略的加载路径列表
- `trained_policy_iteration_list`: 对应的策略迭代步数列表
- `export_controller_name`: 导出控制器名称列表
- `save_path`: 保存路径列表
- `export_format`: 导出格式，'onnx'或'torchscript'
- `opset_version`: ONNX操作集版本（推荐11或更高）

## ONNX vs TorchScript 性能对比

| 格式 | 仿真速度 | 兼容性 | 文件大小 | 推荐场景 |
|------|----------|--------|----------|----------|
| ONNX | 更快 | 更好 | 更小 | Simulink仿真 |
| TorchScript | 较快 | 一般 | 较大 | PyTorch环境 |

## 在Simulink中使用ONNX模型

### 方法1: 使用MATLAB Function模块
1. 在Simulink中添加MATLAB Function模块
2. 运行`matlab_usage_example.m`生成集成代码
3. 将生成的代码复制到MATLAB Function模块中
4. 配置输入输出端口

### 方法2: 使用Deep Learning Toolbox的Predict模块  
1. 在Simulink中添加"Predict"模块（需要Deep Learning Toolbox）
2. 双击模块，选择导出的.onnx文件
3. 配置输入输出维度
4. 连接信号并运行仿真

### 方法3: 使用自定义桥接函数
1. 使用`gops_onnx_validation_bridge.m`作为S-Function
2. 在Simulink中添加MATLAB S-Function模块
3. 设置S-function name为`gops_onnx_validation_bridge`
4. 在参数中指定ONNX模型路径

## MATLAB使用示例

```matlab
% 1. 版本兼容的模型加载（推荐）
net = load_onnx_compatible('model.onnx');

% 2. 简单预测
action = gops_onnx_predict([0.1, 0.2, 0.05, 0.1], 'model.onnx');

% 3. 手动加载（自动版本检测）
matlab_version = version('-release');
matlab_year = str2double(matlab_version(1:4));
if matlab_year >= 2022
    net = importNetworkFromONNX('model.onnx');  % 新版本
else
    net = importONNXNetwork('model.onnx', 'OutputLayerType', 'regression');  % 旧版本
end

% 4. 运行兼容性测试
test_onnx_compatibility    % 检查版本和函数可用性

% 5. 完整测试和示例
matlab_usage_example       % 生成集成代码和参数文件
gops_onnx_test            % 运行性能和兼容性测试
```

## 故障排除

### 常见错误

1. **模块导入错误**: 确保GOPS项目在Python路径中
2. **策略文件不存在**: 检查路径和文件名是否正确
3. **ONNX导出失败**: 可能是模型包含不支持的操作，尝试降低opset_version
4. **验证失败**: 输出差异过大，检查模型精度设置

### 依赖问题

如果遇到包导入问题，请确保安装了所有必要的依赖:

```bash
pip install torch torchvision onnx onnxruntime numpy pathlib
```

## 注意事项

- 确保策略文件路径正确
- ONNX导出需要模型支持静态图追踪
- 建议使用ONNX opset version 11或更高版本
- 导出前会自动进行兼容性检查和模型验证

## 许可证

Copyright (c). All Rights Reserved.
基于原GOPS项目修改，遵循相同许可证条款。