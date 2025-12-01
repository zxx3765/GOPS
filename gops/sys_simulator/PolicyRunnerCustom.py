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
                 use_unified_env_config = True,  # 是否使用统一环境配置
                 eval_G0 = None):  # 评估时指定的G0值
        super().__init__(log_policy_dir_list, 
                         trained_policy_iteration_list, 
                         save_render, plot_range, is_init_info, 
                         init_info, legend_list, use_opt, load_opt_path, 
                         opt_args, save_opt, constrained_env, is_tracking, 
                         use_dist, dt, obs_noise_type, obs_noise_data, 
                         action_noise_type, action_noise_data)
        self.use_unified_env_config = use_unified_env_config
        self.eval_G0 = eval_G0  # 评估时使用的固定G0值
        
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
            
    def _update_init_info_with_G0(self):
        """
        将eval_G0添加到init_info中，用于环境reset时指定G0值
        """
        if self.eval_G0 is not None and self.init_info is not None:
            self.init_info = self.init_info.copy()
            self.init_info["init_G0"] = self.eval_G0
            print(f"Setting evaluation G0 to: {self.eval_G0}")
        elif self.eval_G0 is not None and self.init_info is None:
            self.init_info = {"init_G0": self.eval_G0}
            print(f"Setting evaluation G0 to: {self.eval_G0}")
    
    def run(self):
        # 在运行前更新init_info以包含G0参数
        self._update_init_info_with_G0()
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
        import numpy as np

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
        # Define state names for labels
        state_names = ["xs (Sprung Mass Position)", "vs (Sprung Mass Velocity)",
                       "xu (Unsprung Mass Position)", "vu (Unsprung Mass Velocity)"]

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
            # Use custom state name for RMS plot
            y_label = state_names[j] + " RMS" if j < len(state_names) else "State-{} RMS".format(j + 1)
            plt.ylabel(y_label, default_cfg["label_font"])
            fig.tight_layout(pad=default_cfg["pad"])
            plt.savefig(path_state_rms_fmt, format=default_cfg["img_fmt"], bbox_inches="tight")
            plt.close()

        # Plot sprung mass acceleration (info[7]) time history
        # Extract acceleration data from info_list
        accel_list = []
        for i in range(policy_num):
            info_list_data = self.eval_list[i]["info_list"]
            accel_data = []
            for info in info_list_data:
                if isinstance(info, dict) and 'info' in info:
                    info_array = info['info']
                    if len(info_array) > 7:
                        accel_data.append(info_array[7])
                    else:
                        accel_data.append(0.0)
                else:
                    accel_data.append(0.0)
            accel_array = np.array(accel_data)

            # Match length with step_list by using the same length as state_list
            state_length = len(self.eval_list[i]["state_list"])
            if len(accel_array) > state_length:
                accel_array = accel_array[:state_length]

            accel_list.append(accel_array)

        # Apply plot range if specified
        if self.plot_range is not None and len(self.plot_range) == 2:
            for i in range(policy_num):
                start_range = self.plot_range[0]
                end_range = min(self.plot_range[1], accel_list[i].shape[0])
                accel_list[i] = accel_list[i][start_range:end_range]

        # Plot acceleration time history
        path_accel_fmt = os.path.join(
            self.save_path, "Sprung_Mass_Acceleration.{}".format(default_cfg["img_fmt"])
        )
        fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

        # Save acceleration data to CSV
        accel_data_df = pd.DataFrame(data=accel_list)
        accel_data_df.to_csv(os.path.join(self.save_path, "Sprung_Mass_Acceleration.csv"), encoding="gbk")

        for i in range(policy_num):
            legend = (
                self.legend_list[i]
                if len(self.legend_list) == policy_num
                else self.algorithm_list[i]
            )
            sns.lineplot(x=step_list[i], y=accel_list[i], label="{}".format(legend))

        plt.tick_params(labelsize=default_cfg["tick_size"])
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
        plt.xlabel(x_label, default_cfg["label_font"])
        plt.ylabel("Sprung Mass Acceleration (m/s²)", default_cfg["label_font"])
        plt.legend(loc="best", prop=default_cfg["legend_font"])
        fig.tight_layout(pad=default_cfg["pad"])
        plt.savefig(path_accel_fmt, format=default_cfg["img_fmt"], bbox_inches="tight")
        plt.close()

        # Plot RMS of sprung mass acceleration
        path_accel_rms_fmt = os.path.join(
            self.save_path, "Sprung_Mass_Acceleration_RMS.{}".format(default_cfg["img_fmt"])
        )
        fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

        rms_values = []
        for i in range(policy_num):
            rms = np.sqrt(np.mean(accel_list[i] ** 2))
            rms_values.append(rms)

        x_labels = self.legend_list if len(self.legend_list) == policy_num else self.algorithm_list

        # save rms data to csv
        rms_data = pd.DataFrame(data=rms_values, index=x_labels)
        rms_data.to_csv(
            os.path.join(self.save_path, "Sprung_Mass_Acceleration_RMS.csv"),
            encoding="gbk",
        )

        ax.bar(x_labels, rms_values)

        plt.tick_params(labelsize=default_cfg["tick_size"])
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
        plt.xlabel("Policy", default_cfg["label_font"])
        plt.ylabel("Sprung Mass Acceleration RMS (m/s²)", default_cfg["label_font"])
        fig.tight_layout(pad=default_cfg["pad"])
        plt.savefig(path_accel_rms_fmt, format=default_cfg["img_fmt"], bbox_inches="tight")
        plt.close()

        # Call parent draw method for all other plots
        super().draw()

        # Redraw state plots with custom titles to override the default ones
        for j in range(state_dim):
            path_state_fmt = os.path.join(
                self.save_path, "State-{}.{}".format(j + 1, default_cfg["img_fmt"])
            )
            fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

            for i in range(policy_num):
                legend = (
                    self.legend_list[i]
                    if len(self.legend_list) == policy_num
                    else self.algorithm_list[i]
                )
                sns.lineplot(
                    x=step_list[i], y=state_list[i][:, j], label="{}".format(legend)
                )
            plt.tick_params(labelsize=default_cfg["tick_size"])
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
            plt.xlabel(x_label, default_cfg["label_font"])
            # Use custom state name if available, otherwise use default
            y_label = state_names[j] if j < len(state_names) else "State-{}".format(j + 1)
            plt.ylabel(y_label, default_cfg["label_font"])
            plt.legend(loc="best", prop=default_cfg["legend_font"])
            fig.tight_layout(pad=default_cfg["pad"])
            plt.savefig(
                path_state_fmt, format=default_cfg["img_fmt"], bbox_inches="tight"
            )
            plt.close()

    def draw_frequency_response(self):
        """
        Draw frequency response (Bode magnitude) plots for each state relative to road input
        Road input is extracted from info
        Also draws frequency response for sprung mass acceleration (info[7])
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        from gops.utils.plot_evaluation import cm2inch
        import numpy as np
        from scipy import signal
        import os

        # Import default config from parent module
        from gops.sys_simulator.sys_run import default_cfg

        fig_size = (
            default_cfg["fig_size"],
            default_cfg["fig_size"],
        )

        state_dim = self.eval_list[0]["state_list"][0].shape[0]
        policy_num = len(self.eval_list)

        # Define state names for labels
        state_names = ["xs (Sprung Mass Position)", "vs (Sprung Mass Velocity)",
                       "xu (Unsprung Mass Position)", "vu (Unsprung Mass Velocity)"]

        # First, let's debug and find where road input is stored
        print("\n=== Debugging info structure ===")
        if len(self.eval_list) > 0 and len(self.eval_list[0]["info_list"]) > 1:
            sample_info = self.eval_list[0]["info_list"][1]
            print(f"Info type: {type(sample_info)}")
            print(f"Info content: {sample_info}")
            if isinstance(sample_info, dict):
                print(f"Info keys: {sample_info.keys()}")
                for key, value in sample_info.items():
                    print(f"  {key}: type={type(value)}, value={value}")
        print("=== End debug ===\n")

        # Sampling frequency
        if self.dt is not None:
            fs = 1.0 / self.dt  # Sampling frequency in Hz
        else:
            fs = 100.0  # Default sampling frequency

        # Extract data for frequency analysis (states and acceleration)
        for i in range(policy_num):
            # Get state data
            state_array = np.array(self.eval_list[i]["state_list"])  # Shape: (num_steps, state_dim)

            # Extract road input from info
            info_list = self.eval_list[i]["info_list"]
            road_input = []
            accel_input = []  # Also extract acceleration data

            for idx, info in enumerate(info_list):
                road_value = None
                accel_value = None

                # Try different ways to extract road input
                if isinstance(info, dict):
                    # First try the 'info' key which contains the array
                    if 'info' in info and isinstance(info['info'], (list, np.ndarray)):
                        info_array = info['info']
                        if len(info_array) > 5:
                            road_value = info_array[5]
                        if len(info_array) > 7:
                            accel_value = info_array[7]
                    # Try common key names for road input
                    elif 'road_input' in info:
                        road_value = info['road_input']
                    elif 'road' in info:
                        road_value = info['road']
                    elif 'xr' in info:
                        road_value = info['xr']
                    elif 'disturbance' in info:
                        road_value = info['disturbance']
                    # Try numeric keys
                    elif 5 in info:
                        road_value = info[5]
                    elif '5' in info:
                        road_value = info['5']
                    # If none found, try to get the state from info
                    elif 'state' in info and isinstance(info['state'], (list, np.ndarray)):
                        # Road input might be in the state vector
                        state_vec = info['state']
                        if len(state_vec) > 4:  # If state has more than 4 elements
                            road_value = state_vec[4]

                elif isinstance(info, (list, tuple, np.ndarray)):
                    if len(info) > 5:
                        road_value = info[5]
                    elif len(info) == 5:
                        road_value = info[4]

                # Convert to scalar if needed
                if road_value is not None:
                    if isinstance(road_value, (list, np.ndarray)):
                        road_value = float(road_value[0]) if len(road_value) > 0 else 0.0
                    else:
                        road_value = float(road_value)
                else:
                    road_value = 0.0

                if accel_value is not None:
                    if isinstance(accel_value, (list, np.ndarray)):
                        accel_value = float(accel_value[0]) if len(accel_value) > 0 else 0.0
                    else:
                        accel_value = float(accel_value)
                else:
                    accel_value = 0.0

                road_input.append(road_value)
                accel_input.append(accel_value)

            road_input = np.array(road_input)
            accel_input = np.array(accel_input)

            # Debug: print road input statistics
            print(f"\nPolicy {i} ({self.legend_list[i] if self.legend_list and len(self.legend_list) > i else f'Policy {i+1}'}):")
            print(f"  Road input length: {len(road_input)}")
            print(f"  Road input mean: {np.mean(road_input):.6f}")
            print(f"  Road input std: {np.std(road_input):.6f}")
            print(f"  Road input min: {np.min(road_input):.6f}")
            print(f"  Road input max: {np.max(road_input):.6f}")

            # Handle case where road_input/accel_input length doesn't match state_array
            min_len = min(len(road_input), len(state_array), len(accel_input))
            road_input = road_input[:min_len]
            state_array = state_array[:min_len]
            accel_input = accel_input[:min_len]

            # Sampling frequency
            if self.dt is not None:
                fs = 1.0 / self.dt  # Sampling frequency in Hz
            else:
                fs = 100.0  # Default sampling frequency

            # Compute FFT for road input
            N = len(road_input)
            if N == 0:
                print(f"Warning: No road input data for policy {i}")
                continue

            # Check if road input has variation
            if np.std(road_input) < 1e-10:
                print(f"Warning: Road input has no variation for policy {i}. Cannot compute frequency response.")
                continue

            # Remove DC component
            road_input = road_input - np.mean(road_input)

            # Apply window to reduce spectral leakage
            window = signal.windows.hann(N)
            road_input_windowed = road_input * window

            # FFT of road input
            road_fft = np.fft.rfft(road_input_windowed)
            road_magnitude = np.abs(road_fft)

            # Frequency bins
            freqs = np.fft.rfftfreq(N, d=1/fs)

            # Plot frequency response for each state
            legend_name = (
                self.legend_list[i]
                if self.legend_list and len(self.legend_list) > i
                else f"Policy {i+1}"
            )

            for j in range(state_dim):
                # Remove DC component from state
                state_signal = state_array[:, j] - np.mean(state_array[:, j])

                # Apply window to state signal
                state_signal_windowed = state_signal * window

                # FFT of state
                state_fft = np.fft.rfft(state_signal_windowed)
                state_magnitude = np.abs(state_fft)

                # Calculate frequency response (transfer function magnitude)
                # Avoid division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    freq_response = np.where(road_magnitude > 1e-10,
                                            state_magnitude / road_magnitude,
                                            0.0)

                # Convert to dB
                freq_response_db = 20 * np.log10(freq_response + 1e-10)

                # Save data to file for this policy
                freq_data = pd.DataFrame({
                    'Frequency (Hz)': freqs,
                    'Magnitude (dB)': freq_response_db
                })
                csv_filename = os.path.join(
                    self.save_path,
                    f"Frequency_Response_State{j+1}_{legend_name}.csv"
                )
                freq_data.to_csv(csv_filename, encoding="gbk", index=False)

            # Calculate and save frequency response for sprung mass acceleration
            # Remove DC component from acceleration
            accel_signal = accel_input - np.mean(accel_input)

            # Apply window to acceleration signal
            accel_signal_windowed = accel_signal * window

            # FFT of acceleration
            accel_fft = np.fft.rfft(accel_signal_windowed)
            accel_magnitude = np.abs(accel_fft)

            # Calculate frequency response (transfer function magnitude)
            with np.errstate(divide='ignore', invalid='ignore'):
                freq_response_accel = np.where(road_magnitude > 1e-10,
                                               accel_magnitude / road_magnitude,
                                               0.0)

            # Convert to dB
            freq_response_accel_db = 20 * np.log10(freq_response_accel + 1e-10)

            # Save acceleration frequency response data to file
            freq_data_accel = pd.DataFrame({
                'Frequency (Hz)': freqs,
                'Magnitude (dB)': freq_response_accel_db
            })
            csv_filename_accel = os.path.join(
                self.save_path,
                f"Frequency_Response_Acceleration_{legend_name}.csv"
            )
            freq_data_accel.to_csv(csv_filename_accel, encoding="gbk", index=False)

        # Create combined plots for each state (all policies on same plot)
        for j in range(state_dim):
            path_freq_resp_fmt = os.path.join(
                self.save_path,
                f"Frequency_Response_State{j+1}.{default_cfg['img_fmt']}"
            )

            fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

            for i in range(policy_num):
                # Recompute for plotting
                state_array = np.array(self.eval_list[i]["state_list"])
                info_list = self.eval_list[i]["info_list"]

                road_input = []
                for info in info_list:
                    road_value = None

                    if isinstance(info, dict):
                        # First try the 'info' key which contains the array
                        if 'info' in info and isinstance(info['info'], (list, np.ndarray)):
                            info_array = info['info']
                            if len(info_array) > 5:
                                road_value = info_array[5]
                        elif 'road_input' in info:
                            road_value = info['road_input']
                        elif 'road' in info:
                            road_value = info['road']
                        elif 'xr' in info:
                            road_value = info['xr']
                        elif 'disturbance' in info:
                            road_value = info['disturbance']
                        elif 5 in info:
                            road_value = info[5]
                        elif '5' in info:
                            road_value = info['5']
                        elif 'state' in info and isinstance(info['state'], (list, np.ndarray)):
                            state_vec = info['state']
                            if len(state_vec) > 4:
                                road_value = state_vec[4]
                    elif isinstance(info, (list, tuple, np.ndarray)):
                        if len(info) > 5:
                            road_value = info[5]
                        elif len(info) == 5:
                            road_value = info[4]

                    if road_value is not None:
                        if isinstance(road_value, (list, np.ndarray)):
                            road_value = float(road_value[0]) if len(road_value) > 0 else 0.0
                        else:
                            road_value = float(road_value)
                    else:
                        road_value = 0.0

                    road_input.append(road_value)

                road_input = np.array(road_input)
                min_len = min(len(road_input), len(state_array))
                road_input = road_input[:min_len]
                state_array = state_array[:min_len]

                N = len(road_input)
                if N == 0 or np.std(road_input) < 1e-10:
                    continue

                # Remove DC components
                road_input = road_input - np.mean(road_input)
                state_signal = state_array[:, j] - np.mean(state_array[:, j])

                window = signal.windows.hann(N)
                road_input_windowed = road_input * window
                state_signal_windowed = state_signal * window

                road_fft = np.fft.rfft(road_input_windowed)
                road_magnitude = np.abs(road_fft)

                state_fft = np.fft.rfft(state_signal_windowed)
                state_magnitude = np.abs(state_fft)

                freqs = np.fft.rfftfreq(N, d=1/fs)

                with np.errstate(divide='ignore', invalid='ignore'):
                    freq_response = np.where(road_magnitude > 1e-10,
                                            state_magnitude / road_magnitude,
                                            0.0)

                freq_response_db = 20 * np.log10(freq_response + 1e-10)

                legend_name = (
                    self.legend_list[i]
                    if self.legend_list and len(self.legend_list) > i
                    else f"Policy {i+1}"
                )

                # Plot - limit frequency range for better visualization
                min_freq = 0.2  # Minimum frequency in Hz
                max_freq = 25   # Maximum frequency in Hz
                freq_mask = (freqs >= min_freq) & (freqs <= max_freq)

                plt.plot(freqs[freq_mask], freq_response_db[freq_mask], label=legend_name, alpha=0.8)

            # Set logarithmic scale for x-axis
            ax.set_xscale('log')

            plt.tick_params(labelsize=default_cfg["tick_size"])
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
            plt.xlabel("Frequency (Hz)", default_cfg["label_font"])

            y_label = f"Magnitude (dB) - {state_names[j]}" if j < len(state_names) else f"Magnitude (dB) - State {j+1}"
            plt.ylabel(y_label, default_cfg["label_font"])
            plt.legend(loc="best", prop=default_cfg["legend_font"])
            plt.grid(True, which='both', alpha=0.3)  # Show grid for both major and minor ticks
            plt.xlim(0.2, 25)  # Set x-axis range from 0.2 to 25 Hz

            fig.tight_layout(pad=default_cfg["pad"])
            plt.savefig(path_freq_resp_fmt, format=default_cfg["img_fmt"], bbox_inches="tight")
            plt.close()

        # Create combined plot for sprung mass acceleration (all policies on same plot)
        path_freq_resp_accel_fmt = os.path.join(
            self.save_path,
            f"Frequency_Response_Acceleration.{default_cfg['img_fmt']}"
        )

        fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

        for i in range(policy_num):
            # Recompute for plotting
            info_list = self.eval_list[i]["info_list"]

            road_input = []
            accel_input = []
            for info in info_list:
                road_value = None
                accel_value = None

                if isinstance(info, dict):
                    # First try the 'info' key which contains the array
                    if 'info' in info and isinstance(info['info'], (list, np.ndarray)):
                        info_array = info['info']
                        if len(info_array) > 5:
                            road_value = info_array[5]
                        if len(info_array) > 7:
                            accel_value = info_array[7]
                    elif 'road_input' in info:
                        road_value = info['road_input']
                    elif 'road' in info:
                        road_value = info['road']
                    elif 'xr' in info:
                        road_value = info['xr']
                    elif 'disturbance' in info:
                        road_value = info['disturbance']
                    elif 5 in info:
                        road_value = info[5]
                    elif '5' in info:
                        road_value = info['5']
                    elif 'state' in info and isinstance(info['state'], (list, np.ndarray)):
                        state_vec = info['state']
                        if len(state_vec) > 4:
                            road_value = state_vec[4]
                elif isinstance(info, (list, tuple, np.ndarray)):
                    if len(info) > 5:
                        road_value = info[5]
                    elif len(info) == 5:
                        road_value = info[4]

                if road_value is not None:
                    if isinstance(road_value, (list, np.ndarray)):
                        road_value = float(road_value[0]) if len(road_value) > 0 else 0.0
                    else:
                        road_value = float(road_value)
                else:
                    road_value = 0.0

                if accel_value is not None:
                    if isinstance(accel_value, (list, np.ndarray)):
                        accel_value = float(accel_value[0]) if len(accel_value) > 0 else 0.0
                    else:
                        accel_value = float(accel_value)
                else:
                    accel_value = 0.0

                road_input.append(road_value)
                accel_input.append(accel_value)

            road_input = np.array(road_input)
            accel_input = np.array(accel_input)

            N = len(road_input)
            if N == 0 or np.std(road_input) < 1e-10:
                continue

            # Remove DC components
            road_input = road_input - np.mean(road_input)
            accel_signal = accel_input - np.mean(accel_input)

            window = signal.windows.hann(N)
            road_input_windowed = road_input * window
            accel_signal_windowed = accel_signal * window

            road_fft = np.fft.rfft(road_input_windowed)
            road_magnitude = np.abs(road_fft)

            accel_fft = np.fft.rfft(accel_signal_windowed)
            accel_magnitude = np.abs(accel_fft)

            freqs = np.fft.rfftfreq(N, d=1/fs)

            with np.errstate(divide='ignore', invalid='ignore'):
                freq_response_accel = np.where(road_magnitude > 1e-10,
                                               accel_magnitude / road_magnitude,
                                               0.0)

            freq_response_accel_db = 20 * np.log10(freq_response_accel + 1e-10)

            legend_name = (
                self.legend_list[i]
                if self.legend_list and len(self.legend_list) > i
                else f"Policy {i+1}"
            )

            # Plot - limit frequency range for better visualization
            min_freq = 0.2  # Minimum frequency in Hz
            max_freq = 25   # Maximum frequency in Hz
            freq_mask = (freqs >= min_freq) & (freqs <= max_freq)

            plt.plot(freqs[freq_mask], freq_response_accel_db[freq_mask], label=legend_name, alpha=0.8)

        # Set logarithmic scale for x-axis
        ax.set_xscale('log')

        plt.tick_params(labelsize=default_cfg["tick_size"])
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
        plt.xlabel("Frequency (Hz)", default_cfg["label_font"])
        plt.ylabel("Magnitude (dB) - Sprung Mass Acceleration", default_cfg["label_font"])
        plt.legend(loc="best", prop=default_cfg["legend_font"])
        plt.grid(True, which='both', alpha=0.3)  # Show grid for both major and minor ticks
        plt.xlim(0.2, 25)  # Set x-axis range from 0.2 to 25 Hz

        fig.tight_layout(pad=default_cfg["pad"])
        plt.savefig(path_freq_resp_accel_fmt, format=default_cfg["img_fmt"], bbox_inches="tight")
        plt.close()

        print(f"\nFrequency response plots saved to {self.save_path}")