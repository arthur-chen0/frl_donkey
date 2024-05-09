import argparse
from common.plot import visualize

parser = argparse.ArgumentParser()
parser.add_argument("--logdir", help="Training log directory")
args = parser.parse_args()

if __name__ == "__main__":
    visualize(args.logdir)