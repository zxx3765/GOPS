from gops.sys_simulator.sys_run import PolicyRunner
import numpy as np
import os
from gops.create_pkg.create_env_model import create_env_model
from unified_env_config import override_env_args
class PolicyRunnerCustom(PolicyRunner):
    def __init__(self, log_policy_dir_list, 
                 trained_policy_iteration_list, 
                 save_render = False, 
                 plot_range = None, 
                 is_init_info = False, 
                 init_info = None, 
                 legend_list = None, 
                 use_opt = False, 
                 load_opt_path = None, 
                 opt_args = None, 
                 save_opt = True, 
                 constrained_env = False, 
                 is_tracking = False, 
                 use_dist = False, 
                 dt = None, 
                 obs_noise_type = None, 
                 obs_noise_data = None, 
                 action_noise_type = None, 
                 action_noise_data = None,
                 use_unified_env_config = True):  # 新增参数：是否使用统一环境配置
        super().__init__(log_policy_dir_list, 
                         trained_policy_iteration_list, 
                         save_render, plot_range, is_init_info, 
                         init_info, legend_list, use_opt, load_opt_path, 
                         opt_args, save_opt, constrained_env, is_tracking, 
                         use_dist, dt, obs_noise_type, obs_noise_data, 
                         action_noise_type, action_noise_data)
        self.use_unified_env_config = use_unified_env_config
        
    def _load_env_with_unified_config(self, policy_index=0, use_opt=False):
        """
        使用统一配置加载环境
        """
        if self.use_unified_env_config:
            # 使用统一配置覆盖环境参数
            original_args = self.args_list[policy_index].copy()
            env_id = original_args.get("env_id", "simu_quarter_sus_win")
            
            # 用统一配置覆盖参数
            unified_args = override_env_args(original_args, env_id)
            print(f"Using unified environment config for {env_id}")
            print(f"Key unified parameters: Cs={unified_args['Cs']}, Ks={unified_args['Ks']}, Ms={unified_args['Ms']}, Mu={unified_args['Mu']}")
            
            # 临时设置self.args为统一配置
            original_self_args = self.args
            self.args = unified_args
            env = self._PolicyRunner__load_env(use_opt=use_opt)
            self.args = original_self_args  # 恢复原始self.args
            
            return env
        else:
            # 使用原始配置
            return self._PolicyRunner__load_env(use_opt=use_opt)
    def run(self):
        self.__run_data_with_passive()
        self._PolicyRunner__save_mp4_as_gif()
        self.draw()
    
    def __run_data_with_passive(self):
        # Run passive policy first as comparison baseline
        print("===========================================================")
        print("*** Begin to run passive policy (baseline) ***")
        self.algorithm_list.append("Passive")
        self.args = self.args_list[0]  # Use first policy's args for env setup
        env = self._load_env_with_unified_config(0)  # 使用统一配置加载环境
        if hasattr(env, "set_mode"):
            env.set_mode("test")
            
        if hasattr(env, "train_space") and hasattr(env, "work_space"):
            print("Train space: ")
            print(self.__convert_format(env.train_space))
            print("Work space: ")
            print(self.__convert_format(env.work_space))
        
        # Create passive policy (zero action) with proper interface
        class PassivePolicy:
            def __init__(self, action_dim):
                self.action_dim = action_dim
                
            def policy(self, obs):
                # Return zero logits for the policy network interface
                if isinstance(obs, torch.Tensor):
                    return torch.zeros((obs.shape[0], self.action_dim), dtype=torch.float32)
                else:
                    return torch.zeros((1, self.action_dim), dtype=torch.float32)
            
            def create_action_distributions(self, logits):
                # Create a mock action distribution that always returns zero
                class ZeroActionDistribution:
                    def mode(self):
                        return torch.zeros_like(logits)
                return ZeroActionDistribution()
        
        # Get action dimension from environment
        import torch
        action_dim = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else 1
        passive_policy = PassivePolicy(action_dim)
        
        # Run passive policy
        eval_dict_passive, tracking_dict_passive = self.run_an_episode(
            env, passive_policy, self.init_info, is_opt=False, render=False
        )
        print("Successfully run passive policy")
        print("===========================================================\n")
        
        # Add passive policy results to lists
        self.eval_list.append(eval_dict_passive)
        self.tracking_list.append(tracking_dict_passive)
        
        # Add passive policy to legend
        if self.legend_list is None:
            self.legend_list = ["Passive"]
        else:
            self.legend_list.insert(0, "Passive")
        
        # Now run the trained policies
        for i in range(self.policy_num):
            log_policy_dir = self.log_policy_dir_list[i]
            trained_policy_iteration = self.trained_policy_iteration_list[i]

            self.args = self.args_list[i]
            print("===========================================================")
            print("*** Begin to run policy {} ***".format(i + 1))
            env = self._load_env_with_unified_config(i)  # 使用统一配置加载环境
            if hasattr(env, "set_mode"):
                env.set_mode("test")

            if hasattr(env, "train_space") and hasattr(env, "work_space"):
                print("Train space: ")
                print(self.__convert_format(env.train_space))
                print("Work space: ")
                print(self.__convert_format(env.work_space))
            networks = self._PolicyRunner__load_policy(log_policy_dir, trained_policy_iteration)

            # Run policy
            eval_dict, tracking_dict = self.run_an_episode(
                env, networks, self.init_info, is_opt=False, render=False
            )
            print("Successfully run policy {}".format(i + 1))
            print("===========================================================\n")
            # mp4 to gif
            self.eval_list.append(eval_dict)
            self.tracking_list.append(tracking_dict)

        if self.use_opt:
            if self.load_opt_path is not None:
                eval_dict_opt = np.load(
                    os.path.join(self.load_opt_path, "eval_dict_opt.npy"), 
                    allow_pickle=True).item()
                tracking_dict_opt = np.load(
                    os.path.join(self.load_opt_path, "tracking_dict_opt.npy"), 
                    allow_pickle=True).item()
                print("Successfully load an optimal controller result!")
                print("===========================================================\n")
            else:
                self.args = self.args_list[self.policy_num - 1]
                print("GOPS: Use an optimal controller")
                env = self._load_env_with_unified_config(self.policy_num - 1, use_opt=True)  # 使用统一配置加载环境
                print("The environment for opt")
                if hasattr(env, "set_mode"):
                    env.set_mode("test")

                assert (
                    self.opt_args is not None
                ), "Choose to use optimal controller, but the opt_args is None."

                if self.opt_args["opt_controller_type"] == "OPT":
                    assert (
                        env.has_optimal_controller
                    ), "The environment has no theoretical optimal controller."
                    opt_controller = env.control_policy
                elif self.opt_args["opt_controller_type"] == "MPC":
                    if self.opt_args["use_MPC_for_general_env"] == True:
                        self.args_list[self.policy_num - 1]["env"] = env
                        from gops.sys_simulator.opt_controller_for_gen_env import OptController
                    else:
                        from gops.sys_simulator.opt_controller import OptController
                    model = create_env_model(**self.args_list[self.policy_num - 1], mask_at_done=False)
                    opt_args = self.opt_args.copy()
                    opt_args.pop("opt_controller_type")
                    opt_args.pop("use_MPC_for_general_env")
                    opt_controller = OptController(model, **opt_args,)
                else:
                    raise ValueError(
                        "The optimal controller type should be either 'OPT' or 'MPC'."
                    )

                eval_dict_opt, tracking_dict_opt = self.run_an_episode(
                    env, opt_controller, self.init_info, is_opt=True, render=False
                )
                print("Successfully run an optimal controller!")
                print("===========================================================\n")

            if self.opt_args["opt_controller_type"] == "OPT":
                legend = "OPT"
            elif self.opt_args["opt_controller_type"] == "MPC":
                legend = "MPC-" + str(self.opt_args["num_pred_step"])
                if (
                    "use_terminal_cost" not in self.opt_args.keys()
                    or self.opt_args["use_terminal_cost"] == False
                ):
                    legend += " (w/o TC)"
                else:
                    legend += " (w/ TC)"
            self.legend_list.append(legend)

            if self.save_opt:
                np.save(os.path.join(self.save_path, "eval_dict_opt.npy"), eval_dict_opt)
                np.save(os.path.join(self.save_path, "tracking_dict_opt.npy"), tracking_dict_opt)

            self.eval_list.append(eval_dict_opt)
            if self.is_tracking:
                self.tracking_list.append(tracking_dict_opt)
    
    def draw(self):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        from gops.utils.plot_evaluation import cm2inch
        
        # Import default config from parent module
        from gops.sys_simulator.sys_run import default_cfg
        
        fig_size = (
            default_cfg["fig_size"],
            default_cfg["fig_size"],
        )
        action_dim = self.eval_list[0]["action_list"][0].shape[0]
        state_dim = self.eval_list[0]["state_list"][0].shape[0]
        if self.constrained_env:
            constrain_dim = self.eval_list[0]["constrain_list"][0].shape[0]
        policy_num = len(self.algorithm_list)
        if self.use_opt:
            legend = ""
            policy_num += 1
            if self.opt_args["opt_controller_type"] == "OPT":
                legend = "OPT"
            elif self.opt_args["opt_controller_type"] == "MPC":
                legend = "MPC-" + str(self.opt_args["num_pred_step"])
                if (
                    "use_terminal_cost" not in self.opt_args.keys()
                    or self.opt_args["use_terminal_cost"] is False
                ):
                    legend += " (w/o TC)"
                else:
                    legend += " (w/ TC)"
            self.algorithm_list.append(legend)

        # Create initial list
        reward_list = []
        action_list = []
        state_list = []
        step_list = []
        state_ref_error_list = []
        constrain_list = []
        # Put data into list
        for i in range(policy_num):
            reward_list.append(np.array(self.eval_list[i]["reward_list"]))
            action_list.append(np.array(self.eval_list[i]["action_list"]))
            state_list.append(np.array(self.eval_list[i]["state_list"]))
            step_list.append(np.array(self.eval_list[i]["step_list"]))
            if self.constrained_env:
                constrain_list.append(np.stack(self.eval_list[i]["constrain_list"]))
            if self.is_tracking:
                state_ref_error_list.append(self.tracking_list[i])

        if self.plot_range is None:
            pass
        elif len(self.plot_range) == 2:
            for i in range(policy_num):
                start_range = self.plot_range[0]
                end_range = min(self.plot_range[1], reward_list[i].shape[0])

                reward_list[i] = reward_list[i][start_range:end_range]
                action_list[i] = action_list[i][start_range:end_range]
                state_list[i] = state_list[i][start_range:end_range]
                step_list[i] = step_list[i][start_range:end_range]
                if self.constrained_env:
                    constrain_list[i] = constrain_list[i][start_range:end_range]
                if self.is_tracking:
                    for key, value in self.tracking_list[i].items():
                        self.tracking_list[i][key] = value[start_range:end_range]
        else:
            raise NotImplementedError("Figure range is wrong")

        if self.dt is None:
            x_label = "Time step"
        else:
            step_list = [s * self.dt for s in step_list]
            x_label = "Time (s)"

        # Plot cumulative reward (NEW PLOT)
        path_cumulative_reward_fmt = os.path.join(
            self.save_path, "Cumulative_Reward.{}".format(default_cfg["img_fmt"])
        )
        fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

        # Calculate and save cumulative reward data
        cumulative_reward_list = []
        for i in range(policy_num):
            cumulative_reward = np.cumsum(reward_list[i])
            cumulative_reward_list.append(cumulative_reward)
            
        cumulative_reward_data = pd.DataFrame(data=cumulative_reward_list)
        cumulative_reward_data.to_csv(os.path.join(self.save_path, "Cumulative_Reward.csv"), encoding="gbk")

        for i in range(policy_num):
            legend = (
                self.legend_list[i]
                if len(self.legend_list) == policy_num
                else self.algorithm_list[i]
            )
            sns.lineplot(x=step_list[i], y=cumulative_reward_list[i], label="{}".format(legend))
        plt.tick_params(labelsize=default_cfg["tick_size"])
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
        plt.xlabel(x_label, default_cfg["label_font"])
        plt.ylabel("Cumulative Reward", default_cfg["label_font"])
        plt.legend(loc="best", prop=default_cfg["legend_font"])
        fig.tight_layout(pad=default_cfg["pad"])
        plt.savefig(path_cumulative_reward_fmt, format=default_cfg["img_fmt"], bbox_inches="tight")
        plt.close()

        # RMS of obs
        obs_dim_to_plot = min(4, state_dim)
        for j in range(obs_dim_to_plot):
            path_state_rms_fmt = os.path.join(
                self.save_path, "State-{}-RMS.{}".format(j + 1, default_cfg["img_fmt"])
            )
            fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

            rms_values = []
            for i in range(policy_num):
                rms = np.sqrt(np.mean(state_list[i][:, j] ** 2))
                rms_values.append(rms)

            x_labels = self.legend_list if len(self.legend_list) == policy_num else self.algorithm_list
            
            # save rms data to csv
            rms_data = pd.DataFrame(data=rms_values, index=x_labels)
            rms_data.to_csv(
                os.path.join(self.save_path, "State-{}-RMS.csv".format(j + 1)),
                encoding="gbk",
            )
            
            ax.bar(x_labels, rms_values)
            
            plt.tick_params(labelsize=default_cfg["tick_size"])
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
            plt.xlabel("Policy", default_cfg["label_font"])
            plt.ylabel("State-{} RMS".format(j + 1), default_cfg["label_font"])
            fig.tight_layout(pad=default_cfg["pad"])
            plt.savefig(path_state_rms_fmt, format=default_cfg["img_fmt"], bbox_inches="tight")
            plt.close()
        # Call parent draw method for all other plots
        super().draw()