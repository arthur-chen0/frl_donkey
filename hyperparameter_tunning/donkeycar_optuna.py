import os
import gym
import gym_donkeycar
import uuid
import optuna

from optuna.logging import get_logger
from torch import nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_latest_run_id

from common.wrappers import HistoryWrapper, AutoencoderWrapper
from common.callbacks import TrailEvalCallback

_logger = get_logger(__name__)

def create_env(is_eval=False):
    # os.environ["AE_PATH"] = "/d/nchu/frl/rl_donkey/aae-train-donkeycar/logs/ae-32_1714886383_best.pkl"
    exe_path = "/d/nchu/frl/DonkeySimLinux/donkey_sim.x86_64"
    if is_eval:
        port = 9091
    else:
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
        "throttle_max": 0.9,
        "max_cte": 5.0,
        "steer_limit": 0.8,
    }

    env = gym.make("donkey-minimonaco-track-v0", conf=conf)
    env = AutoencoderWrapper(env)
    env = HistoryWrapper(env, 2)

    return env
    # return make_vec_env("donkey-minimonaco-track-v0", env_kwargs=conf)

def objective(trial:optuna.Trial):

    log_path = "./reinforcement/ppo_donkey_6/" # TODO: auto create the log path
    n_timesteps = 100000
    n_evaluations = 20

    env = create_env()
    eval_env = create_env(is_eval=True)
    # env = DummyVecEnv([lambda: env])

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256])
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
    vf_coef = trial.suggest_float("vf_coef", 0, 1)
    net_arch_type = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])
    activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
    ortho_init = False

    net_arch = {
        "tiny": dict(pi=[64], vf=[64]),
        "small": dict(pi=[64, 64], vf=[64, 64]),
        "medium": dict(pi=[256, 256], vf=[256, 256]),
    }[net_arch_type]

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn_name]

    latest_run_id = get_latest_run_id(log_path=log_path, log_name="PPO")
    save_path = os.path.join(log_path, f"PPO_{latest_run_id + 1}")
    _logger.info("Trial {} started and parameters: {}. The log save to {}. ".format(trial.number, trial.params, save_path))
    # TODO: save the params to the csv file

    model = PPO("MlpPolicy", 
                env, verbose=0, 
                device='auto', 
                tensorboard_log=log_path, 
                n_steps = 256, 
                ent_coef=ent_coef,
                learning_rate=learning_rate,
                batch_size=batch_size,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                n_epochs=n_epochs,
                max_grad_norm=max_grad_norm,
                vf_coef=vf_coef,
                policy_kwargs=dict(net_arch=net_arch, activation_fn=activation_fn, ortho_init=ortho_init)
                )
    model.trail = trial

    optuna_eval_freq = int(n_timesteps / n_evaluations)
    eval_callback = TrailEvalCallback(
        eval_env=eval_env,
        trail=trial,
        n_eval_episodes=5,
        eval_freq=optuna_eval_freq,
        deterministic=True,
        verbose=1,
        log_path=None,
        best_model_save_path=None
    
    )
    try:
        model.learn(total_timesteps=n_timesteps, callback=eval_callback)
        model.env.close()
        eval_env.close()
    except(AssertionError, ValueError) as e:
        model.env.close()
        eval_env.close()
    reward = eval_callback.last_mean_reward

    return reward

if __name__ == "__main__":
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, n_jobs=1)
    print(study.best_params)
    
    # TODO: overwrite the best params to the config.ini file
