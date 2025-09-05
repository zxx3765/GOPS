#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com

#  Description: Simulink Vehicle 3Dof data environment
#  Update Date: 2021-07-011, Wenxuan Wang: create simulink environment

from typing import Optional, List, Tuple, Any, Sequence


import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
from gops.env.env_matlab.resources.simu_quar_sus_win import quarter_sus_win



class SimuQuarterSusWin(gym.Env,):
    def __init__(self, **kwargs: Any):
        spec = quarter_sus_win._env.EnvSpec(
            id="SimuQuarterSustWin-v0",
            max_episode_steps=kwargs["Max_step"],
            terminal_bonus_reward=kwargs["punish_done"],
            strict_reset=True,
        )
        self.env = quarter_sus_win.GymEnv(spec)
        self.dt = 0.01
        self.is_adversary = kwargs.get("is_adversary", False)
        self.obs_scale = np.array(kwargs["obs_scaling"])
        self.act_scale = np.array(kwargs["act_scaling"])
        self.rew_scale = np.array(kwargs["rew_scaling"])
        self.act_max = np.array(kwargs["act_max"])
        # self.done_range = kwargs["done_range"]
        self.punish_done = kwargs["punish_done"]
        # self.use_ref = kwargs["ref_info"]
        # self.ref_horizon = kwargs["ref_horizon"]
        self._state = None

        obs_low = self.obs_scale * np.array([-9999, -9999, -9999, -9999])
        self.state_max = np.array(kwargs["init_state_max"], dtype=float)
        self.state_min = np.array(kwargs["init_state_min"], dtype=float)
        self.observation_space = spaces.Box(obs_low, -obs_low)
        self.action_space = spaces.Box(
            -self.act_scale * self.act_max, self.act_scale * self.act_max,shape=(1,), dtype=float
        )
        # Split RNG, if randomness is needed
        self.rng = np.random.default_rng()

        self.reward_bias = kwargs["rew_bias"]
        self.reward_bound = kwargs["rew_bound"]
        self.act_repeat = kwargs["act_repeat"]
        self.rand_bias = kwargs["rand_bias"]
        self.rand_center = kwargs["rand_center"]

        self.road_seed = kwargs["road_seed"]
        self.as_max = kwargs["as_max"]
        self.deflec_max = kwargs["deflec_max"]
        self.Cs = kwargs["Cs"]
        self.Ks = kwargs["Ks"]
        self.Ms = kwargs["Ms"]
        self.Mu = kwargs["Mu"]
        self.Kt = kwargs["Kt"]
        self.G0 = kwargs["G0"]
        self.f0 = kwargs["f0"]
        self.u = kwargs["u"]
        self.road_type = kwargs["Road_Type"]
        self.road_type_dict = {"Sine": 1, "Chirp": 2, "Random": 3, "Bump": 4}

        self.Q_flec = kwargs["punish_Q_flec"]
        self.Q_acc_s = kwargs.get("punish_Q_acc_s", 0.0)
        # self.Q_acc_u = kwargs.get("punish_Q_acc_u", 0.0)
        self.Q_F = kwargs.get("punish_Q_F", 0.0)
        self.Q_flec_t = kwargs.get("punish_Q_flec_t", 0.0)
        self.b_deflec = kwargs.get("punish_b_deflec", 0.0)
        self.Q_acc_s_h = kwargs.get("punish_Q_acc_s_h", 0.0)
        self.Q_b_deflec = kwargs.get("punish_Q_b_defelc", 0.0)
        # self.R = kwargs["punish_R"]
        self.seed_gen,_ =seeding.np_random(self.road_seed)

        self.rand_low = np.array(self.rand_center) - np.array(self.rand_bias)
        self.rand_high = np.array(self.rand_center) + np.array(self.rand_bias)
        self.seed()
        self.reset()

    @property
    def state(self):
        return self._state

    def reset(
        self, init_state: Optional[Sequence] = None, **kwargs: Any
    ) -> Tuple[np.ndarray]:
        def callback(init_state):
            """Custom reset logic goes here."""
            # Modify your parameter
            # e.g. self.env.model_class.foo_InstP.your_parameter
            if init_state is None:
                self._state = np.random.uniform(low=self.rand_low, high=self.rand_high)
                init_state_rand = self.rng.uniform(low=self.state_min, high=self.state_max)
                self.env.model_class.quarter_sus_win_InstP.xs0 = init_state_rand[0]
                self.env.model_class.quarter_sus_win_InstP.vs0 = init_state_rand[1]
                self.env.model_class.quarter_sus_win_InstP.xu0 = init_state_rand[2]
                self.env.model_class.quarter_sus_win_InstP.vu0 = init_state_rand[3]
            else:
                self._state = np.array(init_state, dtype=np.float32)
                init_state = self._state
                self.env.model_class.quarter_sus_win_InstP.xs0 = init_state[0]
                self.env.model_class.quarter_sus_win_InstP.vs0 = init_state[1]
                self.env.model_class.quarter_sus_win_InstP.xu0 = init_state[2]
                self.env.model_class.quarter_sus_win_InstP.vu0 = init_state[3]

            self.env.model_class.quarter_sus_win_InstP.Cs = self.Cs
            self.env.model_class.quarter_sus_win_InstP.Ks = self.Ks
            self.env.model_class.quarter_sus_win_InstP.ms = self.Ms
            self.env.model_class.quarter_sus_win_InstP.mu = self.Mu
            self.env.model_class.quarter_sus_win_InstP.Kt = self.Kt
            self.env.model_class.quarter_sus_win_InstP.G0 = self.G0
            self.env.model_class.quarter_sus_win_InstP.f0 = self.f0
            self.env.model_class.quarter_sus_win_InstP.u = self.u
            # self.env.model_class.quarter_sus_win_InstP.as_max = self.as_max
            # self.env.model_class.quarter_sus_win_InstP.deflec_max = self.deflec_max
            # self.env.model_class.InstP_quarter_sus_win_T.x_max[:] = self.state_max
            # self.env.model_class.InstP_quarter_sus_win_T.x_min[:] = self.state_min
            # 初始化状态
            
            # self.env.model_class.quarter_sus_win_InstP.road_seed  = self.seed_gen.uniform(low=0, high=10000)
            self.env.model_class.quarter_sus_win_InstP.road_type = self.road_type_dict[self.road_type]
            # self.env.model_class.quarter_sus_win_InstP.Q_dot = self.Q_dot
            self.env.model_class.quarter_sus_win_InstP.Q_flec = self.Q_flec
            self.env.model_class.quarter_sus_win_InstP.b_deflec = self.b_deflec
            self.env.model_class.quarter_sus_win_InstP.Q_dot_s = self.Q_acc_s
            self.env.model_class.quarter_sus_win_InstP.Q_F = self.Q_F
            self.env.model_class.quarter_sus_win_InstP.Q_flec_t = self.Q_flec_t
            self.env.model_class.quarter_sus_win_InstP.Q_dot_s_h = self.Q_acc_s_h
            self.env.model_class.quarter_sus_win_InstP.Q_b_deflec = self.Q_b_deflec
            # self.env.model_class.quarter_sus_win_InstP.Q_dot_u = self.Q_acc_u
            # self.env.model_class.quarter_sus_win_InstP.punish_R = self.R

        # Reset takes an optional callback
        # This callback will be called after model & parameter initialization and before taking first step.
        state,info = self.env.reset(preinit=lambda: callback(init_state))
        # state = self.reset.callback()
        obs = self.postprocess(state)
        return obs

    def _physics_step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self._state = state
        return state, reward, done, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        # Preprocess action
        action_real = self.preprocess(action)
        sum_reward = 0
        for idx in range(self.act_repeat):
            state, reward, done, info = self._physics_step(action_real)
            sum_reward += self.reward_shaping(reward)
            if done:
                sum_reward += self.punish_done
                break
        # Postprocess obs
        obs = self.postprocess(state)
        return obs, sum_reward, done, info

    def preprocess(self, action: np.ndarray) -> Tuple[np.ndarray]:
        action_real = action / self.act_scale
        return action_real

    def postprocess(self, state: np.ndarray) -> Tuple[np.ndarray]:
        
        obs = np.zeros(self.observation_space.shape)
        obs[0] = state[0] #acc_s
        obs[1] = state[1] #vs
        obs[2] = state[2]#suspension_deflection
        obs[3] = state[3]#v_def
        obs = obs / self.obs_scale
        return obs

    def reward_shaping(self, origin_reward: float) -> Tuple[float]:
        modified_reward = origin_reward
        if modified_reward <= -self.reward_bound:
            modified_reward = -self.reward_bound
        modified_reward = modified_reward + self.reward_bias
        modified_reward = modified_reward * self.rew_scale
        return modified_reward

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self):
        pass

    def close(self):
        pass
