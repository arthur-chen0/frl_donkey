import os
import pandas as pd
from matplotlib import pyplot as plt

def read_value(reward_pf):
    # timeSteps = reward_pf['Step'].values
    timeSteps = reward_pf['time/total_timesteps'].values
    # reward = reward_pf['Value'].values
    reward = reward_pf['rollout/ep_rew_mean'].values
    

    return timeSteps, reward

def visualize(step, reward):
    plt.style.use("seaborn-v0_8-deep")
    plt.figure()
    plt.subplot(1, 1, 1)
    plt.title("Mean Reward")
    plt.plot(step, reward, label="reward", color="slategrey", linestyle="-")
    plt.legend()
    plt.xlabel("Step")
    plt.ylabel("Mean Reward")
    plt.savefig(os.path.join(file_dir, "test.png"))
    plt.show



if __name__ == "__main__":
    
    file_dir = os.getcwd()
    reward_pf = pd.read_csv(os.path.join(file_dir, 'record/PPO/dp_adaptive_clipping_FedAvg/2024-05-07/22_50_env2_r30000_f10_noeval/client_1/progress.csv'))
    step, reward = read_value(reward_pf=reward_pf)
    visualize(step, reward)
