from gops.sys_simulator.sys_run import PolicyRunner
import numpy as np
import os
from gops.create_pkg.create_env_model import create_env_model
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
                 action_noise_data = None):
        super().__init__(log_policy_dir_list, 
                         trained_policy_iteration_list, 
                         save_render, plot_range, is_init_info, 
                         init_info, legend_list, use_opt, load_opt_path, 
                         opt_args, save_opt, constrained_env, is_tracking, 
                         use_dist, dt, obs_noise_type, obs_noise_data, 
                         action_noise_type, action_noise_data)
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
        env = self._PolicyRunner__load_env()
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
            env = self._PolicyRunner__load_env()
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
                env = self._PolicyRunner__load_env(use_opt=True)
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