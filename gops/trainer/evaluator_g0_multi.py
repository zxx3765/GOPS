#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Multi-G0 Evaluation of trained policy for quarter suspension
#  Update Date: 2025-09-06, Modified for G0 multi-value evaluation

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from gops.trainer.evaluator import Evaluator


class EvaluatorG0Multi(Evaluator):
    """
    Enhanced evaluator that tests policy with multiple G0 values 
    to assess adaptability across different road conditions.
    """
    
    def __init__(self, index=0, **kwargs):
        super().__init__(index, **kwargs)
        
        # Get G0 range from kwargs (used for training)
        self.G0_min = kwargs.get("G0_min", 0.0001)
        self.G0_max = kwargs.get("G0_max", 0.002)
        self.G0_default = kwargs.get("G0", 0.000512)
        
        # Get custom G0 evaluation values from kwargs
        # If not provided, use default values based on road classes
        eval_G0_low = kwargs.get("eval_G0_low", self.G0_min)
        eval_G0_medium = kwargs.get("eval_G0_medium", self.G0_default)
        eval_G0_high = kwargs.get("eval_G0_high", self.G0_max)
        
        # Define three G0 values: low, medium, high
        self.G0_values = [
            eval_G0_low,      # Low road roughness (typically Class A road)
            eval_G0_medium,   # Medium road roughness (typically Class B road)
            eval_G0_high      # High road roughness (typically Class C road)
        ]
        
        # Initialize TensorBoard writer for G0 multi-evaluation
        self.tb_writer = SummaryWriter(log_dir=self.save_folder) if self.save_folder else None
    
    def run_an_episode(self, iteration, render=True, G0_value=None):
        """Run a single episode with specific G0 value."""
        if self.print_iteration != iteration:
            self.print_iteration = iteration
            self.print_time = 0
        else:
            self.print_time += 1
        
        obs_list = []
        action_list = []
        reward_list = []
        
        # Reset with specific G0 if provided
        if G0_value is not None:
            obs, info = self.env.reset(init_G0=G0_value,init_state=[0.0, 0.0, 0.0, 0.0])
        else:
            obs, info = self.env.reset()
        
        done = 0
        info["TimeLimit.truncated"] = False
        
        while not (done or info["TimeLimit.truncated"]):
            batch_obs = self._prepare_batch_obs(obs)
            action = self._get_action(batch_obs)
            
            next_obs, reward, done, next_info = self.env.step(action)
            
            obs_list.append(obs)
            action_list.append(action)
            reward_list.append(reward)
            
            obs = next_obs
            info = next_info
            if "TimeLimit.truncated" not in info.keys():
                info["TimeLimit.truncated"] = False
            
            if render:
                self.env.render()
        
        # Save evaluation data
        if self.eval_save:
            self._save_episode_data(iteration, obs_list, action_list, reward_list, G0_value)
        
        episode_return = sum(reward_list)
        return episode_return
    
    def _prepare_batch_obs(self, obs):
        """Prepare observation for neural network input."""
        return torch.from_numpy(np.expand_dims(obs, axis=0).astype("float32"))
    
    def _get_action(self, batch_obs):
        """Get action from policy network."""
        logits = self.networks.policy(batch_obs)
        action_distribution = self.networks.create_action_distributions(logits)
        action = action_distribution.mode()
        return action.detach().numpy()[0]
    
    def _save_episode_data(self, iteration, obs_list, action_list, reward_list, G0_value=None):
        """Save episode data with G0 value in filename."""
        eval_dict = {
            "reward_list": reward_list,
            "action_list": action_list,
            "obs_list": obs_list,
            "G0_value": G0_value,
        }
        
        # Add G0 value to save filename if specified
        filename_suffix = "_G0{:.6f}".format(G0_value) if G0_value is not None else ""
        save_path = (
            self.save_folder
            + "/evaluator/iter{}_ep{}{}".format(iteration, self.print_time, filename_suffix)
        )
        np.save(save_path, eval_dict)
    
    def run_n_episodes_single_G0(self, n, iteration, G0_value):
        """Run n episodes with a single G0 value."""
        episode_return_list = []
        
        for episode in range(n):
            episode_return = self.run_an_episode(iteration, self.render, G0_value)
            episode_return_list.append(episode_return)
        
        mean_return = np.mean(episode_return_list)
        std_return = np.std(episode_return_list)
        
        return mean_return, std_return, episode_return_list
    
    def run_evaluation(self, iteration):
        """
        Run evaluation with three different G0 values to assess policy adaptability.
        Returns overall mean performance and logs results to TensorBoard.
        """
        all_returns = []
        detailed_results = {}
        
        G0_labels = ["Low", "Medium", "High"]
        G0_tags = ["G0_Low", "G0_Medium", "G0_High"]
        
        for i, G0_value in enumerate(self.G0_values):
            G0_label = G0_labels[i]
            G0_tag = G0_tags[i]
            
            mean_return, std_return, episode_returns = self.run_n_episodes_single_G0(
                self.num_eval_episode, iteration, G0_value
            )
            
            all_returns.extend(episode_returns)
            detailed_results[G0_label] = {
                "G0_value": G0_value,
                "mean_return": mean_return,
                "std_return": std_return,
                "episode_returns": episode_returns,
            }
            
            # Log individual G0 evaluation results to TensorBoard
            if self.tb_writer:
                self.tb_writer.add_scalar(f"Evaluation/Return_{G0_tag}", mean_return, iteration)
        
        # Calculate overall performance across all G0 values
        overall_mean = np.mean(all_returns)
        overall_std = np.std(all_returns)
        
        # Log overall multi-G0 performance to TensorBoard
        if self.tb_writer:
            self.tb_writer.add_scalar("Evaluation/Return_Overall_MultiG0", overall_mean, iteration)
            
            # Create a summary table showing all three G0 values
            G0_summary_text = f"G0 Evaluation Results (Iteration {iteration}):\n"
            G0_summary_text += f"Low G0 ({self.G0_values[0]:.6f}): {detailed_results['Low']['mean_return']:.3f}\n"
            G0_summary_text += f"Medium G0 ({self.G0_values[1]:.6f}): {detailed_results['Medium']['mean_return']:.3f}\n"
            G0_summary_text += f"High G0 ({self.G0_values[2]:.6f}): {detailed_results['High']['mean_return']:.3f}\n"
            G0_summary_text += f"Overall: {overall_mean:.3f}"
            
            self.tb_writer.add_text("Evaluation/G0_Summary", G0_summary_text, iteration)
            self.tb_writer.flush()
        
        # Save detailed results
        if self.eval_save:
            results_summary = {
                "iteration": iteration,
                "overall_mean": overall_mean,
                "overall_std": overall_std,
                "detailed_results": detailed_results,
                "G0_values_tested": self.G0_values,
            }
            save_path = self.save_folder + f"/evaluator/multi_G0_summary_iter{iteration}.npy"
            np.save(save_path, results_summary)
        
        return overall_mean