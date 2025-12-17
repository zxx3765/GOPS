#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Multilayer Perceptron (MLP)
#  Update: 2021-03-05, Wenjun Zou: create MLP function
#  Update: 2023-07-28, Jiaxin Gao: add FiniteHorizonFullPolicy function
#  Update: 2023-10-25, Wenxuan Wang: add DSAC-T algorithm


__all__ = [
    "DetermPolicy",
    'DetermPolicyCustom',
    "FiniteHorizonPolicy",
    "FiniteHorizonFullPolicy",
    "MultiplierNet",
    "StochaPolicy",
    "ActionValue",
    "ActionValueDis",
    "ActionValueDistri",
    "StochaPolicyDis",
    "StateValue",
    "ActionValueCustom",
    "ActionValueCustomTwoStream",
]

import numpy as np
import torch
import warnings
import torch.nn as nn
import torch.nn.functional as F
from gops.utils.common_utils import get_activation_func
from gops.utils.act_distribution_cls import Action_Distribution


# Define MLP function
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


# Count parameter number of MLP
def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


# Deterministic policy
class DetermPolicy(nn.Module, Action_Distribution):
    """
    Approximated function of deterministic policy.
    Input: observation.
    Output: action.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]

        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(
            pi_sizes,
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]))
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]))
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        action = (self.act_high_lim - self.act_low_lim) / 2 * self.pi(
            obs
        ) + (self.act_high_lim + self.act_low_lim) / 2
        return action

class DetermPolicyCustom(DetermPolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize weights using Xavier initialization
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

class FiniteHorizonPolicy(nn.Module, Action_Distribution):
    """
    Approximated function of deterministic policy for finite-horizon.
    Input: observation, time step.
    Output: action.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"] + 1
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]

        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(
            pi_sizes,
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]))
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]))
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs, virtual_t=1):
        virtual_t = virtual_t * torch.ones(
            size=[obs.shape[0], 1], dtype=torch.float32, device=obs.device
        )
        expand_obs = torch.cat((obs, virtual_t), 1)
        action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(
            self.pi(expand_obs)
        ) + (self.act_high_lim + self.act_low_lim) / 2
        return action


class MultiplierNet(nn.Module, Action_Distribution):
    """
    Approximated function of deterministic policy for finite-horizon.
    Input: observation, time step.
    Output: action.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"] + 1
        hidden_sizes = kwargs["hidden_sizes"]

        pi_sizes = [obs_dim] + list(hidden_sizes) + [1]
        self.pi = mlp(
            pi_sizes,
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )

    def forward(self, obs, virtual_t=1):
        virtual_t = virtual_t * torch.ones(
            size=[obs.shape[0], 1], dtype=torch.float32, device=obs.device
        )
        expand_obs = torch.cat((obs, virtual_t), 1)
        multiplier = self.pi(expand_obs)
        return multiplier
class FiniteHorizonFullPolicy(nn.Module, Action_Distribution):
    """
    Approximated function of deterministic policy for finite-horizon.
    Input: observation, time step.
    Output: action.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        self.act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.pre_horizon = kwargs["pre_horizon"]
        pi_sizes = [obs_dim] + list(hidden_sizes) + [self.act_dim * self.pre_horizon]

        self.pi = mlp(
            pi_sizes,
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]).float())
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]).float())
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        return self.forward_all_policy(obs)[:, 0, :]

    def forward_all_policy(self, obs):
        actions = self.pi(obs).reshape(obs.shape[0], self.pre_horizon, self.act_dim)
        action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(actions) \
                 + (self.act_high_lim + self.act_low_lim) / 2
        return action


# Stochastic Policy
class StochaPolicy(nn.Module, Action_Distribution):
    """
    Approximated function of stochastic policy.
    Input: observation.
    Output: parameters of action distribution.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.std_type = kwargs["std_type"]

        # mean and log_std are calculated by different MLP
        if self.std_type == "mlp_separated":
            pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
            self.mean = mlp(
                pi_sizes,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
            self.log_std = mlp(
                pi_sizes,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
        # mean and log_std are calculated by same MLP
        elif self.std_type == "mlp_shared":
            pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim * 2]
            self.policy = mlp(
                pi_sizes,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
        # mean is calculated by MLP, and log_std is learnable parameter
        elif self.std_type == "parameter":
            pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
            self.mean = mlp(
                pi_sizes,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
            self.log_std = nn.Parameter(-0.5*torch.ones(1, act_dim))

        self.min_log_std = kwargs["min_log_std"]
        self.max_log_std = kwargs["max_log_std"]
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]))
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]))
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        if self.std_type == "mlp_separated":
            action_mean = self.mean(obs)
            action_std = torch.clamp(
                self.log_std(obs), self.min_log_std, self.max_log_std
            ).exp()
        elif self.std_type == "mlp_shared":
            logits = self.policy(obs)
            action_mean, action_log_std = torch.chunk(
                logits, chunks=2, dim=-1
            )  # output the mean
            action_std = torch.clamp(
                action_log_std, self.min_log_std, self.max_log_std
            ).exp()
        elif self.std_type == "parameter":
            action_mean = self.mean(obs)
            action_log_std = self.log_std + torch.zeros_like(action_mean)
            action_std = torch.clamp(
                action_log_std, self.min_log_std, self.max_log_std
            ).exp()

        return torch.cat((action_mean, action_std), dim=-1)


class ActionValue(nn.Module, Action_Distribution):
    """
    Approximated function of action-value function.
    Input: observation, action.
    Output: action-value.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.q = mlp(
            [obs_dim + act_dim] + list(hidden_sizes) + [1],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)
    
class ActionValueCustom(nn.Module, Action_Distribution):
    """
    Approximated function of action-value function.
    Input: observation, action.
    Output: action-value.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.q1 = mlp(
            [obs_dim] + [128] + [200-act_dim],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["hidden_activation"]),
        )
        self.action_distribution_cls = kwargs["action_distribution_cls"]
        self.q2 = mlp(
            [200,64,32] + [1],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        # Initialize weights using Xavier initialization
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, obs, act):
        q = self.q1(obs)
        q = torch.cat([q, act], dim=-1)
        q = self.q2(q)
        return torch.squeeze(q, -1)

class ActionValueCustomTwoStream(nn.Module, Action_Distribution):
    """
    Approximated function of action-value function based on the provided diagram.
    Structure:
      - State path: Linear(obs -> 400) -> ReLU -> Linear(400 -> 300)
      - Action path: Linear(act -> 300)
      - Fusion: (State_path_out + Action_path_out) -> ReLU
      - Output: Linear(300 -> 1)
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        
        # --- 1. 状态特征提取 (借鉴新版: 宽网络) ---
        self.l1_s = nn.Linear(obs_dim, 128)
        self.l2_s = nn.Linear(128, 64)
        
        # --- 2. 动作特征映射 (可选，建议保留以匹配维度) ---
        self.l1_a = nn.Linear(act_dim, 64)
        
        # --- 3. 融合后的深度交互层 (借鉴旧版: 拼接后多层MLP) ---
        # 输入维度 = State特征(300) + Action特征(300) = 600
        self.l3 = nn.Linear(128, 128) 
        self.l4 = nn.Linear(128, 64) # 这一层非常关键，用于学习 s 和 a 的耦合
        self.output = nn.Linear(64, 1)

        # 兼容接口
        if "action_distribution_cls" in kwargs:
            self.action_distribution_cls = kwargs["action_distribution_cls"]

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, obs, act):
        # --- State Path ---
        h_s = F.relu(self.l1_s(obs))
        h_s = F.relu(self.l2_s(h_s)) # 注意：这里建议加一层激活，增强非线性
        
        # --- Action Path ---
        h_a = F.relu(self.l1_a(act)) # 对动作也做非线性映射
        
        # --- Fusion: Concatenation (拼接) ---
        # 拼接比加法更适合表达 "当状态是A且动作为B时..." 这种逻辑
        h = torch.cat([h_s, h_a], dim=-1)
        
        # --- Post-Fusion Interaction ---
        h = F.relu(self.l3(h))
        h = F.relu(self.l4(h))
        
        # Output
        q = self.output(h)
        
        return torch.squeeze(q, -1)

    
class ActionValueDis(nn.Module, Action_Distribution):
    """
    Approximated function of action-value function for discrete action space.
    Input: observation.
    Output: action-value for all action.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_num = kwargs["act_num"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.q = mlp(
            [obs_dim] + list(hidden_sizes) + [act_num],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        return self.q(obs)


class ActionValueDistri(nn.Module):
    """
    Approximated function of distributed action-value function.
    Input: observation.
    Output: parameters of action-value distribution.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.q = mlp(
            [obs_dim + act_dim] + list(hidden_sizes) + [2],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        if "min_log_std"  in kwargs or "max_log_std" in kwargs:
            warnings.warn("min_log_std and max_log_std are deprecated in ActionValueDistri.")

    def forward(self, obs, act):
        logits = self.q(torch.cat([obs, act], dim=-1))
        value_mean, value_std = torch.chunk(logits, chunks=2, dim=-1)
        value_std = torch.nn.functional.softplus(value_std) 
        
        return torch.cat((value_mean, value_std), dim=-1)


class StochaPolicyDis(ActionValueDis, Action_Distribution):
    """
    Approximated function of stochastic policy for discrete action space.
    Input: observation.
    Output: parameters of action distribution.
    """

    pass


class StateValue(nn.Module, Action_Distribution):
    """
    Approximated function of state-value function.
    Input: observation, action.
    Output: state-value.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.v = mlp(
            [obs_dim] + list(hidden_sizes) + [1],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        v = self.v(obs)
        return torch.squeeze(v, -1)
