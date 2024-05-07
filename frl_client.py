import flwr as fl
import numpy as np
import torch
import configparser, argparse

from common.model import DonkeyModel

from collections import OrderedDict
from typing import List, Tuple, Union

parser = argparse.ArgumentParser()
parser.add_argument("--id", help="Client id")
args = parser.parse_args()

train_config = configparser.ConfigParser(allow_no_value=True)
train_config.read('config.ini')


class FlowerClient(fl.client.NumPyClient):

    def __init__(self):
        self.donkeyModel = DonkeyModel(carID=int(args.id))
        self.model, self.env = self.donkeyModel.create()

    def get_parameters(self, config):
        return [
            val.cpu().numpy()
            for _, val in self.model.policy.state_dict().items()
        ]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.policy.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.policy.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print("start training...")
        # print("parameters: ", parameters[0])
        self.set_parameters(parameters)

        # set up model in learning mode with goal number of timesteps to complete
        timesteps = int(train_config['RlSettings']['timesteps'])
        self.model.learn(total_timesteps=timesteps, reset_num_timesteps=False)

        self.obs = self.env.reset()

        print("start evalution after training...")
        mean_reward, std_reward = self.evaluate_policy(eval_episodes_n=3,
                                                       n_step=3000)
        self.model.logger.record("eval/mean_reward", mean_reward)
        self.model.logger.dump(step=self.model.num_timesteps)

        print("done training... mean reward: ", mean_reward, "std: ",
              std_reward)

        self.model.save(self.donkeyModel.logdir + "_ppo_donkey")

        return self.get_parameters(config={}), int(
            train_config['RlSettings']['timesteps']), {}

    def evaluate(self, parameters, config):
        return 0.0, 1000, {"rewards": 0.0}

    def evaluate_policy(
        self,
        eval_episodes_n: int = 10,
        n_step: int = 3000,
        deterministic: bool = True,
        render: bool = False,
        return_episode_rewards: bool = False,
    ) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:

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


client = FlowerClient().to_client()

# flwr.client.start_numpy_client() is deprecated. Instead, use `flwr.client.start_client()`
fl.client.start_client(
    server_address=train_config['FlSettings']['flwrServerIP'],
    client=client,
)

client.numpy_client.env.close()
