import configparser
import gym
import gym_donkeycar
import uuid
import platform
import numpy as np

from loguru import logger
from stable_baselines3 import PPO
from common.wrappers import AutoencoderWrapper, HistoryWrapper

train_config = configparser.ConfigParser(allow_no_value=True)
train_config.read('config.ini')

if platform.system() == "Darwin":
    exe_path = train_config['CarSettings']['exe_path_mac']
elif platform.system() == "Linux":
    exe_path = train_config['CarSettings']['exe_path_linux']

carID = 0
ae_path=train_config['RlSettings']['ae_path']
envName = "donkey-minimonaco-track-v0"
conf = {
            "exe_path": exe_path,
            "host": train_config['CarSettings']['host'],
            "port": 9090 + carID * 2,
            "body_style": "donkey",
            "body_rgb": (128, 128, 128),
            "car_name": "Arthur" + str(carID),
            "font_size": 50,
            "racer_name": "PPO",
            "bio": "Learning to drive w PPO RL",
            "guid": str(uuid.uuid4()),
            "throttle_min": float(train_config['CarSettings']['throttle_min']),
            "throttle_max": float(train_config['CarSettings']['throttle_max']),
            "steer_limit": 0.8,
            "max_cte": float(train_config['CarSettings']['max_cte']),
        }

env = gym.make(envName, conf=conf)
if ae_path is not None:
    logger.info("Use ae " + ae_path)
    env = AutoencoderWrapper(env, ae_path)
    env = HistoryWrapper(env, 2)
env.viewer.handler.send_load_scene(envName)

model = PPO.load("record/20_29_env2_r30000_f10_noeval/client_1/ppo_donkey.zip", env=env, device="mps")

n_step = 15000
episode_rewards = []
curr_ep_reward = 0
obs = env.reset()
for i in range(n_step):
    action, _states = model.predict(obs, deterministic=True)
    # print(action)
    obs, reward, done, info = env.step(action)
    curr_ep_reward += reward
    env.render()
    if done:
        obs = env.reset()
        episode_rewards.append(curr_ep_reward)
        print("step " + str(i) + " episode rewards " + str(curr_ep_reward))
        curr_ep_reward = 0
if not done:
    obs = env.reset()
    episode_rewards.append(curr_ep_reward)
        # print("episode rewards ", episode_rewards)
mean_reward = np.mean(episode_rewards)
std_reward = np.std(episode_rewards)
logger.info("Mean reward " + str(mean_reward))