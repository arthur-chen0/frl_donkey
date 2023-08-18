import flwr as fl
import configparser

train_config = configparser.ConfigParser()
train_config.read('config.ini')

def create_strategy() -> fl.server.strategy:
    strategy_name = train_config['settings']['aggregationFn']
    clients = int(train_config['settings']['clients'])
    fraction_fit = float(train_config['server']['fraction_fit'])
    fraction_evaluate = float(train_config['server']['fraction_evaluate'])
    proximal_mu = float(train_config['server']['proximal_mu'])
    clip_norm = float(train_config['server']['clip_norm'])
    noise_multiplier = float(train_config['server']['noise_multiplier'])
    
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
        config=fl.server.ServerConfig(num_rounds=int(train_config['settings']['rounds'])),
        strategy=create_strategy(),
    )


if __name__ == "__main__":
    main()