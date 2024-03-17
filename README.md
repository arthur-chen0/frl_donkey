# Federated Reinforcement Learning on Donkey Car Gym

## Requirement

- Install Flower federated learning:
``` shell
pip install flwr-nightly
```
- Install stable-baselines3:
``` shell
pip install stable-baselines3[extra]
```

- Download simulator binaries: https://github.com/tawnkramer/gym-donkeycar/releases
- Install master version of gym donkey car:
``` shell
pip install git+https://github.com/arthur-chen0/gym-donkeycar.git
```
## Usage

- Set up the config.ini

| Key | Description | Value |
| ------ | ------ | ------ |
| exe_path_linux | The path of your donkey car simulator installed on Linux system |
| exe_path_mac | The path of your donkey car simulator installed on Mac system |
| host | The federated learning server address | 127.0.0.1 |
| rlAlgo | The algorithm your reinforcement learning using | PPO, TQC |
| aggregationFn | The aggregation function of federated learning | FedAvg, FedProx |
| dp | Use the fixed or adaptive clipping. Set it to 'none' if you do not need differential privacy | dp_fixed_clipping, dp_adaptive_clipping |
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
