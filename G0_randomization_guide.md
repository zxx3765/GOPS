# G0参数随机化功能使用说明

## 功能概述

实现了四分之一悬架模型中G0路面参数的随机化功能，支持：
- **训练时**：G0在指定范围内随机采样，增加环境多样性
- **评估时**：G0固定为特定值，确保结果可重现性

## 使用方法

### 1. 训练场景（随机G0）

```python
# 在训练脚本中设置G0随机化参数
parser.add_argument("--G0", type=float, default=0.001024, help="默认G0值")
parser.add_argument("--G0_min", type=float, default=0.0001, help="训练时G0最小值") 
parser.add_argument("--G0_max", type=float, default=0.002, help="训练时G0最大值")

# 环境创建时会自动支持G0随机化
env = create_env(**env_args)

# 训练时调用reset()不传参数，G0会在[G0_min, G0_max]范围内随机
for episode in range(num_episodes):
    obs = env.reset()  # G0会随机采样
    # ... 训练代码
```

### 2. 评估场景（固定G0）

```python
# 方法1：直接指定G0值
obs = env.reset(init_G0=0.001024)  # 使用Class A标准G0值

# 方法2：使用PolicyRunnerCustom自动管理
runner = PolicyRunnerCustom(
    log_policy_dir_list=["path/to/policy"],
    trained_policy_iteration_list=["iteration"],
    eval_G0=0.001024,  # 指定评估用的固定G0值
    use_unified_env_config=True
)
runner.run()
```

## G0参数范围建议

| 道路等级 | G0范围 | 说明 |
|---------|--------|------|
| Class A | 0.0001-0.002 | 良好道路 |
| Class B | 0.0004-0.008 | 一般道路 |
| Class C | 0.0016-0.032 | 较差道路 |

## 测试脚本

### 基础功能测试
```bash
python test_G0_randomization.py
```
功能：测试G0随机化基本功能，生成分布图

### 单元测试
```bash
python test_G0_unit.py
```
功能：运行标准单元测试，验证各项功能

### 集成测试
```bash  
python test_G0_integration.py
```
功能：完整测试训练和评估场景

## 实现细节

### 环境类修改 (`simu_quarter_sus_win.py`)

```python
# 初始化时添加G0范围参数
self.G0_max = kwargs.get("G0_max", self.G0)
self.G0_min = kwargs.get("G0_min", self.G0)

# reset方法支持init_G0参数
def reset(self, init_state=None, init_G0=None, **kwargs):
    def callback(init_state, init_G0):
        # G0参数设置逻辑
        if init_G0 is None:
            # 随机G0用于训练
            G0_rand = self.rng.uniform(low=self.G0_min, high=self.G0_max)
            self.env.model_class.quarter_sus_win_InstP.G0 = G0_rand
        else:
            # 固定G0用于评估
            self.env.model_class.quarter_sus_win_InstP.G0 = init_G0
```

### PolicyRunner修改 (`PolicyRunnerCustom.py`)

```python
# 构造函数添加eval_G0参数
def __init__(self, ..., eval_G0=None):
    self.eval_G0 = eval_G0

# 自动将eval_G0传递给环境
def _update_init_info_with_G0(self):
    if self.eval_G0 is not None:
        self.init_info["init_G0"] = self.eval_G0
```

## 配置文件

### 统一环境配置 (`unified_env_config.py`)
```python
UNIFIED_QUARTER_SUSPENSION_CONFIG = {
    "G0": 0.001024,      # 评估用标准值
    "G0_min": 0.0001,    # 训练时最小值
    "G0_max": 0.002,     # 训练时最大值
    # ...其他参数
}
```

### 训练脚本配置
```python
parser.add_argument("--G0_min", type=float, default=0.0001)
parser.add_argument("--G0_max", type=float, default=0.002) 
```

## 验证方法

1. **查看G0值**：
   ```python
   current_G0 = env.env.model_class.quarter_sus_win_InstP.G0
   print(f"Current G0: {current_G0}")
   ```

2. **验证随机性**：运行多次reset()，确认G0值不同

3. **验证固定性**：使用init_G0参数，确认G0值固定

## 注意事项

1. **参数传递**：确保G0_min和G0_max参数正确传递给环境
2. **范围检查**：训练时确保G0_min < G0_max
3. **一致性**：评估时始终使用相同的G0值
4. **模型同步**：G0值会自动同步到Simulink模型中

## 常见问题

**Q: 为什么训练时G0还是固定的？**
A: 检查是否传入了G0_min和G0_max参数，且确保调用reset()时没有传init_G0参数。

**Q: 评估时如何确保G0值一致？**
A: 使用PolicyRunnerCustom的eval_G0参数，或直接调用reset(init_G0=value)。

**Q: G0参数如何影响仿真？**
A: G0控制随机路面的强度，值越大路面越不平，对悬架系统挑战越大。