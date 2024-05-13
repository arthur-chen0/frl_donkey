import configparser
import datetime
from common.model import DonkeyModel
from common.callbacks import PlotCallback

train_config = configparser.ConfigParser(allow_no_value=True)
train_config.read('config.ini')

if __name__ == "__main__":
    
    rlAlgo = train_config['RlSettings']['rlAlgo']
 
    date = datetime.datetime.now().date().strftime('%Y-%m-%d')
    time = datetime.datetime.now().strftime('%H_%M')
    
    # Initialize the donkey environment
    donkeyModel = DonkeyModel()
    donkeyModel.logdir = "record/"+ rlAlgo + "/rl_donkey/" + date + "/" + time
    model, env = donkeyModel.create()

    plotCallback = PlotCallback(logdir=donkeyModel.logdir)
    model.learn(total_timesteps=300000, callback=plotCallback)

    env.close()
