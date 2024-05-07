import uuid
import configparser
import gym
import gym_donkeycar
from common.wrappers import AutoencoderWrapper, HistoryWrapper
from torch import nn as nn
from stable_baselines3 import PPO
# from sb3_contrib import TQC

# logger = logging.getLogger(__name__)
log_dir = "/d/NCHU/research/docker_volume/federated_reinforcement_car/log/"

train_config = configparser.ConfigParser()
train_config.read('config.ini')

learning_rate = train_config['RlSettings']['learning_rate']
batch_size = train_config['RlSettings']['batch_size']
gamma = train_config['RlSettings']['gamma']
ent_coef = train_config['RlSettings']['ent_coef']
clip_range = train_config['RlSettings']['clip_range']
n_epochs = train_config['RlSettings']['n_epochs']
gae_lambda = train_config['RlSettings']['gae_lambda']
max_grad_norm = train_config['RlSettings']['max_grad_norm']
vf_coef = train_config['RlSettings']['vf_coef']
net_arch = train_config['RlSettings']['net_arch']
activation_fn = train_config['RlSettings']['activation_fn']

if __name__ == "__main__":

    # Initialize the donkey environment
    # where env_name one of:
    env_list = [
        "donkey-generated-track-v0",
        "donkey-mountain-track-v0",
        "donkey-minimonaco-track-v0",
        "donkey-warren-track-v0",
    ]

    exe_path = "/d/research/Federated_learning/docker_volume/DonkeySimLinux/donkey_sim.x86_64"
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

    # make gym env
    env = gym.make("donkey-minimonaco-track-v0", conf=conf)
    env = AutoencoderWrapper(env, "ae-32_1714903016_best.pkl")
    env = HistoryWrapper(env, 2)
    env.viewer.handler.send_load_scene("donkey-minimonaco-track-v0")
    # env = Monitor(env, log_dir)

    model = PPO("MlpPolicy",
                env,
                verbose=0,
                device='auto',
                tensorboard_log="./reinforcement/ppo_donkey/",
                n_steps=256,
                ent_coef=0.00874178913117294,
                learning_rate=7.534860318826415e-05,
                batch_size=16,
                gamma=0.9,
                gae_lambda=0.9,
                clip_range=0.1,
                n_epochs=5,
                max_grad_norm=0.3,
                vf_coef=0.24310228290806923,
                policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                                   activation_fn=nn.Tanh,
                                   ortho_init=False))

    # model = TQC("CnnPolicy", env, verbose=1, device='cpu', tensorboard_log="./reinforcement/ppo_donkey/", tua=0.02,
    #             batch_size=256, gradient_steps=256, ent_coef="auto", buffer_size=200000, train_freq=200, learning_starts=5000, learning_rate=7.3e-4,
    #             use_sde_at_warmup=True, use_sde=True, sde_sample_freq=16, policy_kwargs=dict(log_std_init=-3, net_arch=[256, 256], n_critics=2))

    # set up model in learning mode with goal number of timesteps to complete
    model.learn(total_timesteps=300000)

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
