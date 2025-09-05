# 该脚本用于测试能否改变环境参数 比如随机种子等
import argparse
from gops.create_pkg.create_env import create_env

if __name__ == "__main__":
    # Parameters Setup
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="simu_quarter_sus_win", help="id of environment")
     # 1. Parameters for environment
    parser.add_argument("--action_type", type=str, default="continu", help="Options: continu/discret")
    parser.add_argument("--is_render", type=bool, default=False, help="Draw environment animation")
    parser.add_argument("--is_adversary", type=bool, default=False, help="Adversary training")
    parser.add_argument("--is_constrained", type=bool, default=False, help="constrained environment")

    parser.add_argument("--Max_step", type=int, default=20000, help="Maximum step of each episode")
    parser.add_argument("--act_repeat", type=int, default=10)
    parser.add_argument("--obs_scaling", type=list, default=[5, 1, 0.03,0.3])
    parser.add_argument("--act_scaling", type=float, default=0.01)
    parser.add_argument("--rew_scaling", type=float, default=1)
    parser.add_argument("--act_max", type=float, default=400)
    parser.add_argument("--punish_done", type=float, default=0.0)
    parser.add_argument("--rew_bias", type=float, default=0)
    parser.add_argument("--rew_bound", type=float, default=1000.0)

    parser.add_argument("--rand_bias", type=list, default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01,0.01, 0.01, 0.01, 0.01])
    parser.add_argument("--rand_center", type=list, default=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # parser.add_argument("--done_range", type=list, default=[6.0, 5.0, np.pi / 6])

    parser.add_argument("--Cs", type=float, default=2000.0,help="damper coefficient")
    parser.add_argument("--Ks", type=float, default=20000.0,help="spring coefficient")
    parser.add_argument("--Ms", type=float, default=400.0,help="Sprung mass")
    parser.add_argument("--Mu", type=float, default=50.0,help="Unsprung mass")
    parser.add_argument("--Kt", type=float, default=200000.0,help="Tire stiffness")
    parser.add_argument("--G0", type=float, default=0.001024,help="the random road") #Class A
    parser.add_argument("--f0", type=float, default=0.1)
    parser.add_argument("--u", type=float, default=20.0)
    parser.add_argument("--as_max", type=float, default=1) #acc_s max 2m/s^2
    parser.add_argument("--deflec_max", type=float, default=0.04) #deflection max 40mm
    parser.add_argument("--road_seed", type=int, default=827538)
    parser.add_argument("--Road_Type", type=str, default="Random",help="Sine/Chirp/Random/Bump")

    # 设置初始状态随机范围 顺序为xs0, vs0, xu0, vu0
    parser.add_argument("--init_state_max", type=list, default=[0.01, 0.1, 0.01, 0.1])
    parser.add_argument("--init_state_min", type=list, default=[-0.01, -0.1, -0.01, -0.1])
    # 代表accs 和 accu的惩罚权重
    parser.add_argument("--punish_Q_acc_s", type=float, default=1.0)
    parser.add_argument("--punish_Q_acc_u", type=float, default=0.001)
    # 代表deflection的惩罚权重
    parser.add_argument("--punish_Q_flec", type=float, default=0.0)
    parser.add_argument("--punish_R", type=float, default=0.00001)

    args = vars(parser.parse_args())
    env = create_env(**args)
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 设置matplotlib后端，确保图片能正常显示
    import matplotlib
    matplotlib.use('TkAgg')  # 或者使用 'Qt5Agg' 如果TkAgg不工作的话
    plt.ion()  # 开启交互模式
    obs_plot = [] 
    reward_plot = []

    # First run
    obs, _ = env.reset()
    # print("First run - road seed:", env.env.env.env.env.env.model_class.InstP_quarter_sus_win_T.road_seed)
    
    obs_run1 = []
    reward_run1 = []
    for step in range(1000):
        action = np.array([1.0])  # Make sure action is properly formatted
        obs, reward, done, info = env.step(action)
        obs_run1.append(obs.copy())  # Store each observation
        reward_run1.append(reward)  # Store each reward
        if done:
            break
    obs_plot.append(np.array(obs_run1))
    reward_plot.append(np.array(reward_run1))
    
    # Second run
    obs, _ = env.reset()
    # env.env.env.env.env.env.model_class.quarter_sus_win_InstP.road_seed = 123456  # Change a parameter to see its effect
    # env.env.env.env.env.env.model_class.quarter_sus_win_InstP.Q_flec = 100.0  # Change a parameter to see its effect
    # print("Second run - road seed:", env.env.env.env.env.env.model_class.InstP_quarter_sus_win_T.road_seed)
    
    obs_run2 = []
    reward_run2 = []
    for step in range(1000):
        action = np.array([1.0])  # Consistent action format
        obs, reward, done, info = env.step(action)
        obs_run2.append(obs.copy())  # Store each observation
        reward_run2.append(reward)  # Store each reward
        if done:
            break
    obs_plot.append(np.array(obs_run2))
    reward_plot.append(np.array(reward_run2))

    # 绘图比较两次的观测值和奖励
    plt.figure(figsize=(15, 10))
    
    # Plot first observation dimension (position) for both runs
    plt.subplot(2, 3, 1)
    if len(obs_plot[0]) > 0 and len(obs_plot[1]) > 0:
        plt.plot(obs_plot[0][:, 0], label='Run 1 - Position', alpha=0.7)
        plt.plot(obs_plot[1][:, 0], label='Run 2 - Position', alpha=0.7)
    plt.legend()
    plt.xlabel('Time step')
    plt.ylabel('Position')
    plt.title('Position Comparison')
    plt.grid(True)
    
    # Plot second observation dimension if exists
    plt.subplot(2, 3, 2)
    if len(obs_plot[0]) > 0 and len(obs_plot[1]) > 0 and obs_plot[0].shape[1] > 1:
        plt.plot(obs_plot[0][:, 1], label='Run 1 - Velocity', alpha=0.7)
        plt.plot(obs_plot[1][:, 1], label='Run 2 - Velocity', alpha=0.7)
        plt.legend()
        plt.xlabel('Time step')
        plt.ylabel('Velocity')
        plt.title('Velocity Comparison')
        plt.grid(True)
    
    # Plot third observation dimension if exists
    plt.subplot(2, 3, 3)
    if len(obs_plot[0]) > 0 and len(obs_plot[1]) > 0 and obs_plot[0].shape[1] > 2:
        plt.plot(obs_plot[0][:, 2], label='Run 1 - Acceleration', alpha=0.7)
        plt.plot(obs_plot[1][:, 2], label='Run 2 - Acceleration', alpha=0.7)
        plt.legend()
        plt.xlabel('Time step')
        plt.ylabel('Acceleration')
        plt.title('Acceleration Comparison')
        plt.grid(True)
    
    # Plot rewards
    plt.subplot(2, 3, 4)
    if len(reward_plot[0]) > 0 and len(reward_plot[1]) > 0:
        plt.plot(reward_plot[0], label='Run 1 - Reward', alpha=0.7)
        plt.plot(reward_plot[1], label='Run 2 - Reward', alpha=0.7)
        plt.legend()
        plt.xlabel('Time step')
        plt.ylabel('Reward')
        plt.title('Reward Comparison')
        plt.grid(True)
    
    # Plot cumulative rewards
    plt.subplot(2, 3, 5)
    if len(reward_plot[0]) > 0 and len(reward_plot[1]) > 0:
        cum_reward_1 = np.cumsum(reward_plot[0])
        cum_reward_2 = np.cumsum(reward_plot[1])
        plt.plot(cum_reward_1, label='Run 1 - Cumulative', alpha=0.7)
        plt.plot(cum_reward_2, label='Run 2 - Cumulative', alpha=0.7)
        plt.legend()
        plt.xlabel('Time step')
        plt.ylabel('Cumulative Reward')
        plt.title('Cumulative Reward Comparison')
        plt.grid(True)
    
    # Summary statistics
    plt.subplot(2, 3, 6)
    if len(obs_plot[0]) > 0 and len(obs_plot[1]) > 0 and len(reward_plot[0]) > 0 and len(reward_plot[1]) > 0:
        stats_text = f"Run 1: {len(obs_plot[0])} steps\n"
        stats_text += f"Run 2: {len(obs_plot[1])} steps\n"
        stats_text += f"Obs dims: {obs_plot[0].shape[1]}\n"
        stats_text += f"Max diff pos: {np.max(np.abs(obs_plot[0][:min(len(obs_plot[0]), len(obs_plot[1])), 0] - obs_plot[1][:min(len(obs_plot[0]), len(obs_plot[1])), 0])):.6f}\n"
        stats_text += f"Run 1 total reward: {np.sum(reward_plot[0]):.2f}\n"
        stats_text += f"Run 2 total reward: {np.sum(reward_plot[1]):.2f}\n"
        stats_text += f"Avg reward 1: {np.mean(reward_plot[0]):.6f}\n"
        stats_text += f"Avg reward 2: {np.mean(reward_plot[1]):.6f}"
        plt.text(0.05, 0.5, stats_text, transform=plt.gca().transAxes, fontsize=9, verticalalignment='center')
        plt.axis('off')
        plt.title('Statistics')
    
    plt.tight_layout()
    plt.show()
    
    # 确保图片显示并保持窗口打开
    plt.draw()
    plt.pause(0.1)  # 短暂暂停确保图片渲染
    
    # 保存图片到文件
    plt.savefig('env_parameter_test_comparison.png', dpi=300, bbox_inches='tight')
    print("图片已保存为: env_parameter_test_comparison.png")
    
    # 保持图片窗口打开
    input("按Enter键关闭图片窗口...")
    
    # Print summary information
    print(f"\nSummary:")
    print(f"Run 1 completed {len(obs_plot[0])} steps")
    print(f"Run 2 completed {len(obs_plot[1])} steps") 
    print(f"Observation dimensions: {obs_plot[0].shape[1] if len(obs_plot[0]) > 0 else 'N/A'}")
    if len(obs_plot[0]) > 0 and len(obs_plot[1]) > 0:
        min_steps = min(len(obs_plot[0]), len(obs_plot[1]))
        pos_diff = np.abs(obs_plot[0][:min_steps, 0] - obs_plot[1][:min_steps, 0])
        print(f"Maximum position difference in first {min_steps} steps: {np.max(pos_diff):.6f}")
        print(f"Average position difference in first {min_steps} steps: {np.mean(pos_diff):.6f}")
    if len(reward_plot[0]) > 0 and len(reward_plot[1]) > 0:
        print(f"Run 1 total reward: {np.sum(reward_plot[0]):.2f}")
        print(f"Run 2 total reward: {np.sum(reward_plot[1]):.2f}")
        print(f"Run 1 average reward: {np.mean(reward_plot[0]):.6f}")
        print(f"Run 2 average reward: {np.mean(reward_plot[1]):.6f}")
        reward_diff = np.abs(reward_plot[0][:min(len(reward_plot[0]), len(reward_plot[1]))] - 
                           reward_plot[1][:min(len(reward_plot[0]), len(reward_plot[1]))])
        print(f"Maximum reward difference: {np.max(reward_diff):.6f}")
        print(f"Average reward difference: {np.mean(reward_diff):.6f}")
    
    
    
        