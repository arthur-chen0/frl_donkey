import uuid
# import gymnasium as gym
import gym
# import gym.envs
import gym_donkeycar
from torch import nn as nn
from stable_baselines3 import PPO
from sb3_contrib import TQC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

# logger = logging.getLogger(__name__)
logger = configure(None, ["stdout"])
log_dir = "/d/NCHU/research/docker_volume/federated_reinforcement_car/log/"

if __name__ == "__main__":

    # Initialize the donkey environment
    # where env_name one of:
    env_list = ["donkey-generated-track-v0",
                "donkey-mountain-track-v0",
                "donkey-minimonaco-track-v0",
                "donkey-warren-track-v0",]

    exe_path = "/d/nchu/frl/DonkeySimLinux/donkey_sim.x86_64"
    port = 9199


    conf = {
        "exe_path": exe_path,
        "host": "127.0.0.1",
        "port": port,
        "body_style": "donkey",
        "body_rgb": (128, 128, 128),
        "car_name": "Arthur",
        "font_size": 50,
        "racer_name": "PPO",
        "bio": "Learning to drive w PPO RL",
        "guid": str(uuid.uuid4()),
        "throttle_min": 0.1,
        "throttle_max": 1.0,
        "max_cte": 5.0,
        "steer_limit": 0.8,
    }

    if False:

        # Make an environment test our trained policy
        env = gym.make("donkey-generated-roads-v0", conf=conf)
        env = Monitor(env, log_dir)

        model = PPO.load("ppo_donkey")

        obs = env.reset()
        for _ in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()
            if done:
                obs = env.reset()

        print("done testing")

    else:

        # make gym env
        env = gym.make("donkey-minimonaco-track-v0", conf=conf)
        env.viewer.handler.send_load_scene("donkey-minimonaco-track-v0")
        # env = Monitor(env, log_dir)

        # create cnn policy
        # model = PPO("CnnPolicy", env, verbose=1, device='cpu', tensorboard_log="./reinforcement/ppo_donkey/", 
        #             n_steps = 256, 
        #             ent_coef=1.8575908330893995e-05,
        #             learning_rate=0.00014486991007815455,
        #             batch_size=128,
        #             gamma=0.9,
        #             gae_lambda=1.0,
        #             clip_range=0.1,
        #             n_epochs=20,
        #             max_grad_norm=2,
        #             vf_coef=0.06312273019578722
        #             )

        model = PPO("CnnPolicy", env, verbose=1, device='cpu', tensorboard_log="./reinforcement/ppo_donkey/", 
                    n_steps = 256, 
                    ent_coef=1.404682392502331e-06,
                    learning_rate=0.0011178121774255232,
                    batch_size=32,
                    gamma=0.95,
                    gae_lambda=1.0,
                    clip_range=0.3,
                    n_epochs=10,
                    max_grad_norm=0.9,
                    vf_coef=0.27473931653483585,
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),activation_fn=nn.ReLU)
                    )


        # model = TQC("CnnPolicy", env, verbose=1, device='cpu', tensorboard_log="./reinforcement/ppo_donkey/", tua=0.02,
        #             batch_size=256, gradient_steps=256, ent_coef="auto", buffer_size=200000, train_freq=200, learning_starts=5000, learning_rate=7.3e-4,
        #             use_sde_at_warmup=True, use_sde=True, sde_sample_freq=16, policy_kwargs=dict(log_std_init=-3, net_arch=[256, 256], n_critics=2))

        # set up model in learning mode with goal number of timesteps to complete
        model.learn(total_timesteps=300000)
        # print("done learning")

        obs = env.reset()
        # print("env reset")
        r = 0
        # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)
        for i in range(1000):
            
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            r += reward

        #     # try:
        #     #     env.render()
        #     # except Exception as e:
        #     #     print(e)
        #     #     print("failure in render, continuing...")
            if done:
                # logger.info("Done...")
                obs = env.reset()

        #     if i % 100 == 0:
        #         # print("saving...")
        #         # logger.info("saving... %d", i)
        #         model.save("ppo_donkey")

        # # Save the agent
        # model.save("ppo_donkey")
        print("done training..... reward: ", r)

    env.close()
