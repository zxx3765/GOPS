# Simulink随机路面种子固定问题解决方案

## 问题描述

在使用Simulink生成代码的四分之一车悬架训练环境中，路面输入是通过随机数生成的。但是Simulink生成的C++代码在编译后，随机种子（RandSeed）被硬编码为固定值（`1062371329U`），导致每次训练episode都使用相同的随机路面，无法实现路面的多样性训练。

### 原始问题
- **现象**：每次reset环境后，随机路面完全相同
- **原因**：Simulink生成的代码中，`RandSeed`在`initialize()`函数中被硬编码初始化
- **影响**：训练过程中智能体只能学习特定的路面模式，泛化能力差

## 解决方案

### 核心思路

通过Python访问Simulink生成代码中的`DW`（Block states）结构，在每次环境reset时动态修改`RandSeed`字段，从而实现：
1. **训练模式**：每次reset使用不同的随机种子，产生多样化的路面
2. **评估模式**：使用固定的随机种子，确保评估结果可复现

### 技术实现

#### 1. 代码结构分析

Simulink生成的代码包含以下关键结构：

```cpp
// quarter_sus_imp_force.h
struct DW_quarter_sus_imp_force_T {
    // ... other states
    uint32_T RandSeed;  // 随机数生成器的种子
    real_T NextOutput;  // 下一个随机输出值
    // ...
};
```

Python通过pybind11生成的接口可以访问：
```python
env.unwrapped.env.model_class.quarter_sus_imp_force_DW.RandSeed
```

#### 2. 环境类修改

修改文件：`gops/env/env_matlab/simu_quarter_sus_imp_force.py`

**关键修改点**：

```python
def reset(
    self, init_state: Optional[Sequence] = None,
    init_G0: Optional[float] = None,
    init_road_seed: Optional[int] = None,  # 新增参数
    **kwargs: Any
) -> Tuple[np.ndarray]:
    # 根据是否提供init_road_seed决定使用固定或随机种子
    if init_road_seed is not None:
        # 评估模式：使用固定种子
        random_road_seed = np.uint32(init_road_seed)
    else:
        # 训练模式：生成随机种子
        random_road_seed = self.rng.integers(0, 2**32 - 1, dtype=np.uint32)

    def callback(init_state, init_G0):
        # ... 其他初始化代码 ...
        pass

    # 执行reset
    state, info = self.env.reset(preinit=lambda: callback(init_state, init_G0))

    # 关键：在reset完成后设置RandSeed
    # 必须在reset之后设置，因为reset会重新初始化DW结构
    self.env.model_class.quarter_sus_imp_force_DW.RandSeed = int(random_road_seed)

    obs = self.postprocess(state)
    return obs
```

**为什么要在reset之后设置**：
- Simulink的`reset()`会调用`initialize()`函数
- `initialize()`会将`RandSeed`重置为硬编码值
- 因此必须在reset完成后再次设置`RandSeed`

## 使用方法

### 训练时（随机路面）

```python
from gops.create_pkg.create_env import create_env

env_args = {
    "env_id": "simu_quarter_sus_imp_force",
    # ... 其他参数
}

env = create_env(**env_args)

# 训练循环
for episode in range(num_episodes):
    # 不指定init_road_seed，每次reset自动生成随机种子
    obs = env.reset()

    while not done:
        action = policy.get_action(obs)
        obs, reward, done, info = env.step(action)
```

### 评估时（固定路面）

```python
# 使用固定种子确保可复现性
fixed_seed = 12345

for eval_episode in range(num_eval_episodes):
    # 指定init_road_seed使用固定种子
    obs = env.reset(
        init_state=[0.0, 0.0, 0.0, 0.0],  # 固定初始状态
        init_G0=0.001024,                  # 固定路面参数
        init_road_seed=fixed_seed          # 固定随机种子
    )

    while not done:
        action = policy.get_action(obs)
        obs, reward, done, info = env.step(action)
```

### 在PolicyRunner中使用

PolicyRunnerCustom已经集成了`eval_road_seed`支持，使用方法：

```python
from gops.sys_simulator.PolicyRunnerCustom import PolicyRunnerCustom

runner = PolicyRunnerCustom(
    log_policy_dir_list=[...],
    trained_policy_iteration_list=[...],
    is_init_info=True,
    init_info={
        "init_state": [0.0, 0.0, 0.0, 0.0],
        "ref_time": 0.0,
        "ref_num": 3,
    },
    use_unified_env_config=True,
    eval_G0=0.001024,        # 固定路面参数
    eval_road_seed=12345,    # 固定随机种子 - 确保评估可复现
    eval_max_step=10000,
    # ... 其他参数
)

runner.run()
```

**重要说明**：
- `eval_road_seed`会自动添加到`init_info`中
- 所有评估的episode都会使用相同的随机路面
- 确保不同实验之间的公平比较

### 验证测试

提供了四个测试脚本验证功能：

### 1. `test_randseed_access.py`
验证能否访问和修改`RandSeed`字段
```bash
python test_randseed_access.py
```

### 2. `test_road_randomization.py`
验证训练模式下每次reset产生不同路面
```bash
python test_road_randomization.py
```

### 3. `test_fixed_seed_evaluation.py`
验证评估模式下固定种子的可复现性
```bash
python test_fixed_seed_evaluation.py
```

### 4. `test_policy_runner_road_seed.py`
验证PolicyRunnerCustom的参数传递机制
```bash
python test_policy_runner_road_seed.py
```

**预期结果**：
- ✓ 相同种子产生完全相同的路面（max_diff < 1e-10）
- ✓ 不同种子产生不同的路面（max_diff > 1e-6）
- ✓ 随机模式每次reset产生不同路面
- ✓ PolicyRunnerCustom正确传递eval_road_seed

## 技术细节

### RandSeed的数据类型

- C++中：`uint32_T`（无符号32位整数）
- Python中：需要转换为`np.uint32`或`int`
- 有效范围：`[0, 2^32-1]` = `[0, 4294967295]`

### 随机数生成流程

1. 环境初始化时，`RandSeed`被设为硬编码值
2. Simulink的White Noise模块使用`RandSeed`生成随机数
3. 每次生成随机数后，`RandSeed`会被更新
4. `NextOutput`存储当前生成的随机值

### 访问路径

完整的访问路径：
```python
# 通过wrapper获取base environment
base_env = env.unwrapped  # SimuQuarterSusImpForce对象

# 访问slxpy封装的model
model_class = base_env.env.model_class  # QuarterSusImpForce对象

# 访问DW结构
dw = model_class.quarter_sus_imp_force_DW

# 修改RandSeed
dw.RandSeed = new_seed_value
```

## 适用范围

该解决方案适用于所有使用Simulink生成代码的GOPS环境，包括但不限于：

- `simu_quarter_sus_imp_force`
- `simu_quarter_sus_win`
- `simu_quarter_sus_vimp`
- 其他包含随机扰动的Simulink环境

**迁移方法**：
1. 找到对应环境的DW结构名称（如`quarter_sus_win_DW`）
2. 修改对应环境类的`reset()`方法
3. 在reset后设置`RandSeed`

## 常见问题

### Q1: 为什么不在callback中设置RandSeed？
**A**: 因为`preinit`回调是在模型参数初始化之前执行的，之后Simulink会调用`initialize()`将`RandSeed`重置为硬编码值。必须在reset完成后设置。

### Q2: 如何选择合适的固定种子？
**A**:
- 对于评估，使用项目中约定的固定值（如12345）
- 对于不同测试场景，可以使用不同的固定种子
- 确保文档中记录使用的种子值以便复现

### Q3: 训练时如何确保随机性？
**A**: 环境的`self.rng`是numpy的随机数生成器，已经在`__init__`中正确初始化。每次reset时`self.rng.integers()`会生成新的随机值。

### Q4: 是否需要修改Simulink模型？
**A**: 不需要！这个解决方案完全在Python层面实现，无需重新生成Simulink代码。

## 总结

该解决方案通过在Python层面动态修改Simulink生成代码的内部状态，成功实现了：

1. **训练时的路面多样性**：每个episode使用不同的随机路面
2. **评估时的可复现性**：固定种子确保结果一致
3. **无需修改Simulink模型**：纯Python实现，维护简单
4. **向后兼容**：不指定`init_road_seed`时保持原有行为（随机）

这为强化学习训练提供了更好的泛化能力，同时保证了评估的科学性和可复现性。

---

**修改文件清单**：
- ✅ `gops/env/env_matlab/simu_quarter_sus_imp_force.py` - 环境类核心修改
- ✅ `gops/sys_simulator/PolicyRunnerCustom.py` - PolicyRunner集成
- ✅ `example_run/run_mlp_quartersus_imp_force.py` - 使用示例
- ✅ `test_randseed_access.py` - 测试脚本1
- ✅ `test_road_randomization.py` - 测试脚本2
- ✅ `test_fixed_seed_evaluation.py` - 测试脚本3
- ✅ `test_policy_runner_road_seed.py` - 测试脚本4
- ✅ `example_road_randomization_usage.py` - 详细使用示例
- 📝 `ROAD_RANDOMIZATION_SOLUTION.md` - 本文档

**作者**: Claude Code
**日期**: 2025-12-17
