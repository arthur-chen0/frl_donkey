import os
import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt

colors = ['#EB7A77', "#B5CAA0", "#77969A", "#D9AB42"]

def read_value(pf):
    timeSteps = pf['time/total_timesteps'].values
    reward = pf['rollout/ep_rew_mean'].values
    length = pf['rollout/ep_len_mean'].values

    return timeSteps, reward, length
    
def plot_reward_fig(plot_list: dict, logdir: str):
    plt.style.use("seaborn-v0_8-deep")
    plt.figure(figsize=(10, 5))
    plt.title("Mean Reward")
    for dir in plot_list.keys():
        plt.plot(plot_list[dir]["timeSteps"], plot_list[dir]["reward"], label=dir, color=colors[int(dir.split("_")[1])], linestyle="-")
        
    plt.legend()
    plt.xlabel("Step")
    plt.ylabel("Mean Reward")
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MultipleLocator(30000))
    # plt.grid(axis="x")
    plt.savefig(os.path.join(logdir, logdir.split("/")[2] + "_reward.png"))
    # plt.show
    plt.close()
    
def plot_length_fig(plot_list: dict, logdir: str):
    plt.style.use("seaborn-v0_8-deep")
    plt.figure(figsize=(10, 5))
    plt.title("Mean Length")
    for dir in plot_list.keys():
        plt.plot(plot_list[dir]["timeSteps"], plot_list[dir]["length"], label=dir, color=colors[int(dir.split("_")[1])], linestyle="-")
        
    plt.legend()
    plt.xlabel("Step")
    plt.ylabel("Mean Length")
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MultipleLocator(30000))
    # plt.grid(axis="x")
    plt.savefig(os.path.join(logdir, logdir.split("/")[2] + "_length.png"))
    # plt.show
    plt.close()
    
    
def visualize(logdir: str):
    file_dir = os.getcwd()
    directory = os.path.join(file_dir, logdir)
    
    plot_list = dict()
    for dir in os.listdir(directory):
        if "client" in dir:
            dir_path = os.path.join(directory, dir)
            if os.path.isdir(dir_path):
                plot_list[dir] = dict()
                file = os.path.join(dir_path, "progress.csv")
                timeSteps, reward, length = read_value(pf=pd.read_csv(file))
                plot_list[dir]["timeSteps"] = timeSteps
                plot_list[dir]["reward"] = reward
                plot_list[dir]['length'] = length
                
    plot_reward_fig(plot_list, logdir)
    plot_length_fig(plot_list, logdir)
                         

