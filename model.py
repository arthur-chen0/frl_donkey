import configparser
import gym
import gym_donkeycar
import uuid
import platform
import datetime

from stable_baselines3 import PPO

train_config = configparser.ConfigParser()
train_config.read('config.ini')

env_list = ["donkey-generated-track-v0",
            "donkey-mountain-track-v0",
            "donkey-minimonaco-track-v0",
            "donkey-warren-track-v0",]

if platform.system() == "Darwin":
    exe_path = train_config['CarSettings']['exe_path_mac']
elif platform.system() == "Linux":
    exe_path = train_config['CarSettings']['exe_path_linux']

class DonkeyModel:

    def __init__(self, envNum=None, carID=0):
        
        dt = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M')

        self.envName = env_list[int(train_config['RlSettings']['env'])]

        if envNum is not None:
            print("arg is not none, env: ", envNum)
            self.envName = env_list[int(envNum)]
            logdir = dt + "_" + train_config['RlSettings']['rlAlgo'] + "_" + train_config['FlSettings']['aggregationFn'] + "_" + train_config['RlSettings']['policy'] + "_r" + train_config["RlSettings"]["timesteps"] + "_f" + train_config["FlSettings"]["rounds"] + "_noeval" + "/client_"  + str(carID)
        else:
            logdir = dt + "_" + train_config['RlSettings']['rlAlgo'] + "_" + train_config['FlSettings']['aggregationFn'] + "_" + train_config['RlSettings']['policy'] + "_env" + train_config["RlSettings"]["env"] + "_r" + train_config["RlSettings"]["timesteps"] + "_f" + train_config["FlSettings"]["rounds"] + "_noeval" + "/client_"  + str(carID)

        self.logdir = logdir
        self.conf = {
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
            # "steer_limit": 0.5,
            "max_cte": float(train_config['CarSettings']['max_cte']),
            }  
        
        
    def create(self):
        env = gym.make(self.envName, conf=self.conf)
        env.viewer.handler.send_load_scene(self.envName)
        model = PPO("CnnPolicy", env, verbose=1, device="auto", n_steps=256, tensorboard_log=self.logdir, ent_coef=0.01)
        return model, env