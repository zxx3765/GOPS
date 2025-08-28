#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: Custom implementation based on GOPS DDPG
#  Description: DDPG with gradient clipping functionality
#  
#  This extends the original DDPG algorithm with gradient threshold (clipping) capability

__all__ = ["ApproxContainer", "DDPGCustom"]

import torch
import torch.nn.utils as torch_utils
import time
from gops.algorithm.ddpg import DDPG, ApproxContainer
from gops.utils.tensorboard_setup import tb_tags


class DDPGCustom(DDPG):
    """
    DDPG with gradient clipping - inherits from original DDPG
    
    Only overrides the necessary methods to add gradient clipping functionality.
    This keeps the implementation minimal and clean.

    Additional Args:
        float   gradient_clip_critic  : gradient clipping threshold for critic network. Default to None (no clipping).
        float   gradient_clip_actor   : gradient clipping threshold for actor network. Default to None (no clipping).
        bool    use_gradient_norm     : whether to use gradient norm clipping (True) or value clipping (False). Default to True.
    """

    def __init__(
        self, 
        gradient_clip_critic: float = None,
        gradient_clip_actor: float = None,
        use_gradient_norm: bool = True,
        **kwargs
    ):
        # 调用父类构造函数
        super().__init__(**kwargs)
        
        # 添加gradient clipping参数
        self.gradient_clip_critic = gradient_clip_critic
        self.gradient_clip_actor = gradient_clip_actor
        self.use_gradient_norm = use_gradient_norm
        
        print(f"DDPG with Gradient Clipping initialized:")
        print(f"  - Critic gradient clip: {self.gradient_clip_critic}")
        print(f"  - Actor gradient clip: {self.gradient_clip_actor}")
        print(f"  - Use gradient norm: {self.use_gradient_norm}")

    @property
    def adjustable_parameters(self):
        """扩展父类的adjustable_parameters，添加gradient clipping参数"""
        base_params = super().adjustable_parameters
        return base_params + (
            "gradient_clip_critic",
            "gradient_clip_actor", 
            "use_gradient_norm",
        )

    def _compute_gradient(self, data: dict, iteration):
        """重写梯度计算方法，添加gradient clipping功能"""
        tb_info = dict()
        start_time = time.perf_counter()

        # Critic网络梯度计算（与原版相同）
        self.networks.q_optimizer.zero_grad()
        if not self.per_flag:
            o, a, r, o2, d = (
                data["obs"], data["act"], data["rew"], data["obs2"], data["done"]
            )
            loss_q, q = self._compute_loss_q(o, a, r, o2, d)
            loss_q.backward()
        else:
            o, a, r, o2, d, idx, weight = (
                data["obs"], data["act"], data["rew"], 
                data["obs2"], data["done"], data["idx"], data["weight"]
            )
            loss_q, q, abs_err = self._compute_loss_q_per(o, a, r, o2, d, idx, weight)
            loss_q.backward()

        # 🔥 新增：Critic网络梯度裁剪
        critic_grad_norm = None
        if self.gradient_clip_critic is not None:
            if self.use_gradient_norm:
                critic_grad_norm = torch_utils.clip_grad_norm_(
                    self.networks.q.parameters(), self.gradient_clip_critic
                )
            else:
                torch_utils.clip_grad_value_(
                    self.networks.q.parameters(), self.gradient_clip_critic
                )

        # Actor网络梯度计算（与原版相同）
        for p in self.networks.q.parameters():
            p.requires_grad = False

        self.networks.policy_optimizer.zero_grad()
        loss_policy = self._compute_loss_policy(o)
        loss_policy.backward()

        # 🔥 新增：Actor网络梯度裁剪
        actor_grad_norm = None
        if self.gradient_clip_actor is not None:
            if self.use_gradient_norm:
                actor_grad_norm = torch_utils.clip_grad_norm_(
                    self.networks.policy.parameters(), self.gradient_clip_actor
                )
            else:
                torch_utils.clip_grad_value_(
                    self.networks.policy.parameters(), self.gradient_clip_actor
                )

        for p in self.networks.q.parameters():
            p.requires_grad = True

        # 记录信息到tensorboard（扩展原版信息）
        end_time = time.perf_counter()
        tb_info[tb_tags["loss_critic"]] = loss_q.item()
        tb_info[tb_tags["critic_avg_value"]] = q.item()
        tb_info[tb_tags["alg_time"]] = (end_time - start_time) * 1000
        tb_info[tb_tags["loss_actor"]] = loss_policy.item()
        
        # 🔥 新增：记录梯度范数信息
        if critic_grad_norm is not None:
            tb_info["gradients/critic_grad_norm"] = critic_grad_norm.item()
        if actor_grad_norm is not None:
            tb_info["gradients/actor_grad_norm"] = actor_grad_norm.item()

        if self.per_flag:
            return tb_info, idx, abs_err
        else:
            return tb_info