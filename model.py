import configparser
import gym
import gym_donkeycar
import uuid
import platform
import datetime

from stable_baselines3 import PPO
from sb3_contrib import TQC

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

    def __init__(self, argEnvNum=None, carID=0):

        rlAlgo = train_config['RlSettings']['rlAlgo']
        aggregationFn = train_config['FlSettings']['aggregationFn']
        policy = train_config['RlSettings']['policy']
        timesteps = train_config["RlSettings"]["timesteps"]
        rounds = train_config["FlSettings"]["rounds"]
        env_num = train_config["RlSettings"]["env"]
        dp = train_config["FlSettings"]['dp']

        date = datetime.datetime.now().date().strftime('%Y-%m-%d')
        time = datetime.datetime.now().strftime('%H_%M')

        # self.envName = env_list[int(env_num)]

        # logdir = "record/" + date + "/"

        if argEnvNum is not None:
            print("arg is not none, env: ", argEnvNum)
            env_num = str(argEnvNum)
        
        self.envName = env_list[int(env_num)]
        # logdir += time + "_" + rlAlgo + "_" + dp + aggregationFn + "_" + policy + "_env" + env_num + "_r" + timesteps + "_f" + rounds + "_noeval" + "/client_"  + str(carID)
        logdir = "record/" + rlAlgo + "/" + dp + "_" + aggregationFn + "/" + date + "/" + time + "_env" + env_num + "_r" + timesteps + "_f" + rounds + "_noeval" + "/client_"  + str(carID)

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

        if "PPO" in train_config["RlSettings"]["rlAlgo"]:
            print("RL algorithm: PPO")
            model = PPO("CnnPolicy", env, verbose=1, device="auto", n_steps=256, tensorboard_log=self.logdir, ent_coef=0.01)
        elif "TQC" in train_config["RlSettings"]["rlAlgo"]:
            print("RL algorithm: TQC")
            model = TQC("CnnPolicy", env, verbose=1, device='auto', tensorboard_log=self.logdir, tau=0.02,
                        batch_size=256, gradient_steps=256, ent_coef="auto", buffer_size=100000, train_freq=200, learning_starts=5000, learning_rate=7.3e-4,
                        use_sde_at_warmup=True, use_sde=True, sde_sample_freq=16, policy_kwargs=dict(log_std_init=-3, net_arch=[256, 256], n_critics=2))
        return model, env