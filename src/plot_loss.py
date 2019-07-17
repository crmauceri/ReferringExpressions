import argparse
import matplotlib.pyplot as plt

from config import cfg
from networks.NetworkFactory import networkFactory

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Load a trained network and plot its loss')
    parser.add_argument('config_file', help='config file path')

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    model = networkFactory(cfg)

    plt.figure(1, figsize=(9, 9))

    plt.plot(range(len(model.total_loss)), model.total_loss, label=cfg.OUTPUT.CHECKPOINT_PREFIX)

    leg = plt.legend(loc='best')

    plt.show()