import flwr as fl
import configparser

train_config = configparser.ConfigParser()
train_config.read('config.ini')

def create_strategy() -> fl.server.strategy:
    strategy_name = train_config['settings']['aggregationFn']
    
    strategy = None

    if "FedAvg" in strategy_name:
        print("Server Strategy: FedAvg")
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=0.5,
            fraction_evaluate=0.5,
            min_fit_clients=train_config['settings']['clients'],
            min_evaluate_clients=train_config['settings']['clients'],
            min_available_clients=train_config['settings']['clients'])
    
    elif "FedProx" in strategy_name:
        print("Server Strategy: FedProx")
        strategy = fl.server.strategy.FedProx(
            fraction_fit=0.5,
            fraction_evaluate=0.5,
            proximal_mu=0.3,
            min_fit_clients=train_config['settings']['clients'],
            min_evaluate_clients=train_config['settings']['clients'],
            min_available_clients=train_config['settings']['clients'])
    
    if "dp" in strategy_name:
        print("Server Strategy with defferential privacy")
        return fl.server.strategy.DPFedAvgFixed(
            strategy, 
            num_sampled_clients = train_config['settings']['clients'], 
            clip_norm = 0.001, 
            noise_multiplier = 0.001, 
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