import flwr as fl
import numpy as np
import datetime
import torch
import uuid
import configparser, argparse
import platform

import gym
import gym_donkeycar

from stable_baselines3 import PPO, DDPG

from collections import OrderedDict
from typing import List, Tuple, Union

parser = argparse.ArgumentParser()
parser.add_argument("--id",help="Client id")
parser.add_argument("--env",help="Environment number")
args = parser.parse_args()

train_config = configparser.ConfigParser()
train_config.read('config.ini')

env_list = ["donkey-generated-track-v0",
            "donkey-mountain-track-v0",
            "donkey-minimonaco-track-v0",
            "donkey-warren-track-v0",]


if platform.system() == "Darwin":
    exe_path = train_config['settings']['exe_path_mac']
elif platform.system() == "Linux":
    exe_path = train_config['settings']['exe_path_linux']
port = 9090 + int(args.id) * 2

dt = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M')
logdir = dt + "_" + train_config['settings']['rlAlgo'] + "_" + train_config['settings']['aggregationFn'] + "_" + train_config['settings']['policy'] + "_env" + train_config["settings"]["env"] + "_r" + train_config["settings"]["timesteps"] + "_f" + train_config["settings"]["rounds"] + "_noeval" + "/client_"  + args.id


class FlowerClient(fl.client.NumPyClient):
    def __init__(self):

        self.conf = {
            "exe_path": exe_path,
            "host": "127.0.0.1",
            "port": port,
            "body_style": "donkey",
            "body_rgb": (128, 128, 128),
            "car_name": "Arthur" + args.id,
            "font_size": 50,
            "racer_name": "PPO",
            "country": "USA",
            "bio": "Learning to drive w PPO RL",
            "guid": str(uuid.uuid4()),
            "throttle_min": 0.1,
            "throttle_max": 1.0,
            # "steer_limit": 0.5,
            "max_cte": 8.0,
        }

        
        # self.env = gym.make(env_list[int(train_config['settings']['env1'])], conf=self.conf)
        envName = env_list[int(train_config['settings']['env'])]
        
        if args.env is not None:
            print("arg is not none, env: ", args.env)
            envName = env_list[int(args.env)]
            
        self.env = gym.make(envName, conf=self.conf)
        self.env.viewer.handler.send_load_scene(envName)

        # new_logger = configure("./tensorboard/frl/client_1", ["stdout", "csv", "tensorboard"])

        # create cnn policy
        self.model = PPO("CnnPolicy", self.env, verbose=1, device="auto", n_steps=256, tensorboard_log=logdir, ent_coef=0.01)
        # policy_kwargs = dict(n_critics=2, n_quantiles=25)
        # self.model = TQC("CnnPolicy", self.env, top_quantiles_to_drop_per_net=2, verbose=1, buffer_size=300000, policy_kwargs=policy_kwargs, tensorboard_log=r'./tensorboard_TQC_dpFedAvg_Cnn_env0_r15000_f3/client_1')

        # self.model.set_logger(new_logger)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.policy.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.policy.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.policy.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print("start training...")
        print("parameters: ", parameters[0])
        self.set_parameters(parameters)
        # set up model in learning mode with goal number of timesteps to complete
        # self.model.learn(total_timesteps=10000)
        timesteps = int(train_config['settings']['timesteps'])
        self.model.learn(total_timesteps=timesteps, reset_num_timesteps=False)

        self.obs = self.env.reset()
        print("start evalution after training...")
        mean_reward, std_reward = self.evaluate_policy(eval_episodes_n=3, n_step=3000)

        print("done training... mean reward: ", mean_reward, "std: ", std_reward)
        print("parameters: ", parameters[0])
        # self.env.close()
        
        return self.get_parameters(config={}), int(train_config['settings']['timesteps']), {}

    def evaluate(self, parameters, config):
        # print("evalutate parameters: ", parameters[0])
        # env = gym.make(args.env_name, conf=self.conf)

        # model = PPO.load("ppo_donkey", device='cpu')
        # self.set_parameters(parameters)
        # self.model.learn(total_timesteps=3000)
        # self.env.reset()
        # print("start evaluation after aggregation...")
        # mean_reward, std_reward = self.evaluate_policy(eval_episodes_n=5, n_step=1000)

        # print("done testing... mean reward: ", mean_reward, " std: ", std_reward)

        # env.close()

        # return 0.0, 1000, {"rewards": mean_reward}
        return 0.0, 1000, {"rewards": 0.0}

    def evaluate_policy(self, eval_episodes_n: int = 10, n_step:int = 3000, deterministic: bool = True, render: bool = False, return_episode_rewards: bool = False,) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:

        episode_rewards = []
        curr_ep_reward = 0
        obs = self.env.reset()
        for _ in range(n_step):
            action, _states = self.model.predict(obs, deterministic=True)
            # print(action)
            obs, reward, done, info = self.env.step(action)
            curr_ep_reward += reward
            self.env.render()
            if done:
                obs = self.env.reset()
                episode_rewards.append(curr_ep_reward)
                curr_ep_reward = 0
        if not done:
            obs = self.env.reset()
            episode_rewards.append(curr_ep_reward)
        print("episode rewards ", episode_rewards)
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        return mean_reward, std_reward     



client = FlowerClient()

fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)

client.env.close()
