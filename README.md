# Federated Reinforcement Learning on Donkey Car Gym

## Requirement

- Install Flower federated learning:
``` shell
pip install flwr
```
- Install stable-baselines3:
``` shell
pip install stable-baselines3[extra]
```

- Download simulator binaries: https://github.com/tawnkramer/gym-donkeycar/releases
- Install master version of gym donkey car:
``` shell
pip install git+https://github.com/tawnkramer/gym-donkeycar
```
## Usage

- Set up the config.ini

| Key | Description | Value |
| ------ | ------ | ------ |
| exe_path_linux | The path of your donkey car simulator installed on Linux system |
| exe_path_mac | The path of your donkey car simulator installed on Mac system |
| rlAlgo | The algorithm your reinforcement learning using | PPO |
| aggregationFn | The aggregation function of federated learning <br> - Add dp prefix if you want to use Differential privacy | FedAvg, FedProx, dpFedAvg, dpFedProx |
| policy | The neural network used in reinforcement learning | Cnn, Mlp |
| env | environment number | 0-3 |
| rounds | The number of federated learning rounds | A number you want |
| timesteps | The number of total learning steps | A number you want |
| clients | The number of federated learning clients | A number you want |

- Start the server, and it will run the strategy that you set up in config.ini

``` shell
python frl_server.py
```

- Start the client with id, number of clients at least 2

``` shell
python frl_client.py --id 1
```

- If you want to train the client with different environments, you could set the env number specify for each client

``` shell
python frl_client.py --id 1 --env 1
```
