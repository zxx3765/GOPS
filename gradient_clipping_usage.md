# 如何在原有训练脚本中使用Gradient Clipping

这个文档展示了如何在现有的DDPG训练脚本中添加gradient clipping功能。

## 修改步骤

### 1. 最小修改方案（推荐）

只需要修改原有训练脚本中的两个地方：

**原有代码**：
```python
parser.add_argument("--algorithm", type=str, default="DDPG", help="RL algorithm")
```

**修改后**：
```python
parser.add_argument("--algorithm", type=str, default="DDPGWithGradientClip", help="RL algorithm")

# 添加gradient clipping参数
parser.add_argument("--gradient_clip_critic", type=float, default=None, help="Gradient clipping threshold for critic")
parser.add_argument("--gradient_clip_actor", type=float, default=None, help="Gradient clipping threshold for actor") 
parser.add_argument("--use_gradient_norm", type=bool, default=True, help="Use gradient norm clipping")
```

### 2. 完整的使用示例

```python
# 在你的训练脚本中
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 使用新算法
    parser.add_argument("--algorithm", type=str, default="DDPGWithGradientClip")
    
    # 添加gradient clipping参数
    parser.add_argument("--gradient_clip_critic", type=float, default=10.0)
    parser.add_argument("--gradient_clip_actor", type=float, default=10.0) 
    parser.add_argument("--use_gradient_norm", type=bool, default=True)
    
    # ... 其他现有参数保持不变 ...
    
    args = vars(parser.parse_args())
    env = create_env(**args)
    args = init_args(env, **args)
    
    # 创建算法时会自动使用gradient clipping
    alg = create_alg(**args)
    
    # 设置参数（包括gradient clipping参数）
    alg_params = {
        'gamma': args['gamma'], 
        'tau': args['tau'], 
        'delay_update': args['delay_update'],
        'gradient_clip_critic': args['gradient_clip_critic'],
        'gradient_clip_actor': args['gradient_clip_actor'],
        'use_gradient_norm': args['use_gradient_norm']
    }
    alg.set_parameters(alg_params)
    
    # 其余代码保持不变
    sampler = create_sampler(**args)
    buffer = create_buffer(**args)
    evaluator = create_evaluator(**args)
    trainer = create_trainer(alg, sampler, buffer, evaluator, **args)
    
    trainer.train()
```

## 参数说明

### Gradient Clipping参数

- **gradient_clip_critic**: Critic网络的梯度裁剪阈值
  - `None`: 不进行梯度裁剪（默认原始DDPG行为）
  - `float > 0`: 梯度裁剪阈值，推荐值：`1.0-50.0`

- **gradient_clip_actor**: Actor网络的梯度裁剪阈值  
  - `None`: 不进行梯度裁剪
  - `float > 0`: 梯度裁剪阈值，推荐值：`1.0-50.0`

- **use_gradient_norm**: 梯度裁剪类型
  - `True`: 使用梯度范数裁剪 (`torch.nn.utils.clip_grad_norm_`) **推荐**
  - `False`: 使用梯度值裁剪 (`torch.nn.utils.clip_grad_value_`)

### 推荐参数组合

```python
# 轻度裁剪（适用于稳定的环境）
--gradient_clip_critic 10.0 --gradient_clip_actor 5.0 --use_gradient_norm True

# 中度裁剪（适用于一般情况）
--gradient_clip_critic 5.0 --gradient_clip_actor 2.0 --use_gradient_norm True

# 强裁剪（适用于梯度爆炸严重的情况）
--gradient_clip_critic 1.0 --gradient_clip_actor 0.5 --use_gradient_norm True

# 不使用裁剪（原始DDPG行为）
--gradient_clip_critic None --gradient_clip_actor None
```

## 监控梯度信息

启用gradient clipping后，TensorBoard会自动记录额外的梯度信息：

- `gradients/critic_grad_norm`: Critic网络梯度范数
- `gradients/actor_grad_norm`: Actor网络梯度范数

可以通过这些指标来调整裁剪阈值：
- 如果梯度范数经常达到阈值，说明裁剪在起作用
- 如果梯度范数远小于阈值，可以降低阈值或不使用裁剪

## 优点

1. **最小侵入性**：只需修改算法名称和添加几个参数
2. **向后兼容**：设置为None时完全等同于原始DDPG
3. **自动集成**：无需修改trainer、buffer等其他组件
4. **监控友好**：自动记录梯度信息到TensorBoard
5. **灵活配置**：可以分别控制Critic和Actor的裁剪策略