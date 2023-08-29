import flwr as fl
import configparser

from model import DonkeyModel
from flwr.common import ndarrays_to_parameters


train_config = configparser.ConfigParser()
train_config.read('config.ini')

def create_strategy() -> fl.server.strategy:
    strategy_name = train_config['FlSettings']['aggregationFn']
    clients = int(train_config['FlSettings']['clients'])
    fraction_fit = float(train_config['FlSettings']['fraction_fit'])
    fraction_evaluate = float(train_config['FlSettings']['fraction_evaluate'])
    proximal_mu = float(train_config['FlSettings']['proximal_mu'])
    clip_norm = float(train_config['FlSettings']['clip_norm'])
    noise_multiplier = float(train_config['FlSettings']['noise_multiplier'])
    
    strategy = None
    
    if "FedAvg" in strategy_name:
        print("Server Strategy: FedAvg")
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=clients,
            min_evaluate_clients=clients,
            min_available_clients=clients)
    
    elif "FedProx" in strategy_name:
        print("Server Strategy: FedProx")
        strategy = fl.server.strategy.FedProx(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            proximal_mu=proximal_mu,
            min_fit_clients=clients,
            min_evaluate_clients=clients,
            min_available_clients=clients)
        
    elif "FedYogi" in strategy_name:
        print("Server Strategy: FedYogi")

        model, env = DonkeyModel(envNum=0).create()
       
        strategy = fl.server.strategy.FedYogi(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=clients,
            min_evaluate_clients=clients,
            min_available_clients=clients,
            initial_parameters=ndarrays_to_parameters([val.cpu().numpy() for _, val in model.policy.state_dict().items()]))
        env.close()

    elif "FedAdam" in strategy_name:
        print("Server Strategy: FedAdam")
        model, env = DonkeyModel(envNum=0).create()
       
        strategy = fl.server.strategy.FedAdam(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=clients,
            min_evaluate_clients=clients,
            min_available_clients=clients,
            initial_parameters=ndarrays_to_parameters([val.cpu().numpy() for _, val in model.policy.state_dict().items()]))
        env.close()
    
    elif "FedAdagrad" in strategy_name:
        print("Server Strategy: FedAdagrad")
        model, env = DonkeyModel(envNum=0).create()
        strategy = fl.server.strategy.FedAdagrad(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=clients,
            min_evaluate_clients=clients,
            min_available_clients=clients,
            initial_parameters=ndarrays_to_parameters([val.cpu().numpy() for _, val in model.policy.state_dict().items()]))
        env.close()
    
    if "dp" in strategy_name:
        print("Server Strategy with defferential privacy")
        return fl.server.strategy.DPFedAvgFixed(
            strategy, 
            num_sampled_clients = clients, 
            clip_norm = clip_norm, 
            noise_multiplier = noise_multiplier, 
            server_side_noising = True)
    else:
        return strategy
    


def main() -> None:
    # Start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=int(train_config['FlSettings']['rounds'])),
        strategy=create_strategy(),
    )


if __name__ == "__main__":
    main()