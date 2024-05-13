import flwr as fl
import configparser
import re

from loguru import logger
from common.model import DonkeyModel
from flwr.common import ndarrays_to_parameters

train_config = configparser.ConfigParser(allow_no_value=True)
train_config.read('config.ini')


def create_strategy() -> fl.server.strategy:
    strategy_name = train_config['FlSettings']['aggregationFn']
    clients = int(train_config['FlSettings']['clients'])
    fraction_fit = float(train_config['FlSettings']['fraction_fit'])
    fraction_evaluate = float(train_config['FlSettings']['fraction_evaluate'])
    proximal_mu = float(train_config['FlSettings']['proximal_mu'])
    dp = train_config['FlSettings']['dp']
    clip_norm = float(train_config['FlSettings']['clip_norm'])
    noise_multiplier = float(train_config['FlSettings']['noise_multiplier'])

    strategy = None

    if re.search("FedAvg", strategy_name, re.IGNORECASE):
        logger.info("Server Strategy: FedAvg")
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=clients,
            min_evaluate_clients=clients,
            min_available_clients=clients)

    elif re.search("FedProx", strategy_name, re.IGNORECASE):
        logger.info("Server Strategy: FedProx")
        strategy = fl.server.strategy.FedProx(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            proximal_mu=proximal_mu,
            min_fit_clients=clients,
            min_evaluate_clients=clients,
            min_available_clients=clients)

    elif re.search("FedYogi", strategy_name, re.IGNORECASE):
        logger.info("Server Strategy: FedYogi")

        model, env = DonkeyModel().create(isLog=False)

        strategy = fl.server.strategy.FedYogi(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=clients,
            min_evaluate_clients=clients,
            min_available_clients=clients,
            initial_parameters=ndarrays_to_parameters([
                val.cpu().numpy()
                for _, val in model.policy.state_dict().items()
            ]))
        env.close()

    elif re.search("FedAdam", strategy_name, re.IGNORECASE):
        logger.info("Server Strategy: FedAdam")
        model, env = DonkeyModel().create(isLog=False)

        strategy = fl.server.strategy.FedAdam(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=clients,
            min_evaluate_clients=clients,
            min_available_clients=clients,
            initial_parameters=ndarrays_to_parameters([
                val.cpu().numpy()
                for _, val in model.policy.state_dict().items()
            ]))
        env.close()

    elif re.search("FedAdagrad", strategy_name, re.IGNORECASE):
        logger.info("Server Strategy: FedAdagrad")
        model, env = DonkeyModel().create(isLog=False)
        strategy = fl.server.strategy.FedAdagrad(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=clients,
            min_evaluate_clients=clients,
            min_available_clients=clients,
            initial_parameters=ndarrays_to_parameters([
                val.cpu().numpy()
                for _, val in model.policy.state_dict().items()
            ]))
        env.close()

    if dp is not None:

        if re.search("dp_fixed_clipping", dp, re.IGNORECASE):
            if re.search("server", dp, re.IGNORECASE):
                logger.info("Server Strategy with defferential privacy: server fixed clipping")
                return fl.server.strategy.DifferentialPrivacyServerSideFixedClipping(
                    strategy,
                    num_sampled_clients=clients,
                    clipping_norm=clip_norm,
                    noise_multiplier=noise_multiplier
                )
            elif re.search("client", dp, re.IGNORECASE):
                logger.info("Server Strategy with defferential privacy: client fixed clipping")
                return fl.server.strategy.DifferentialPrivacyClientSideFixedClipping(
                    strategy,
                    num_sampled_clients=clients,
                    clipping_norm=clip_norm,
                    noise_multiplier=noise_multiplier
                )

        elif re.search("dp_adaptive_clipping", dp, re.IGNORECASE):
            if re.search("server", dp, re.IGNORECASE):
                logger.info("Server Strategy with defferential privacy: server adaptive clipping")
                return fl.server.strategy.DifferentialPrivacyServerSideAdaptiveClipping(
                    strategy,
                    num_sampled_clients=clients,
                    noise_multiplier=noise_multiplier,
                )
            elif re.search("client", dp, re.IGNORECASE):
                logger.info("Server Strategy with defferential privacy: client adaptive clipping")
                return fl.server.strategy.DifferentialPrivacyClientSideAdaptiveClipping(
                    strategy,
                    num_sampled_clients=clients,
                    noise_multiplier=noise_multiplier,
                )
    return strategy


def main() -> None:
    # Start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(
            num_rounds=int(train_config['FlSettings']['rounds'])),
        strategy=create_strategy(),
    )


if __name__ == "__main__":
    main()
