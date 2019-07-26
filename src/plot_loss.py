import argparse
import matplotlib.pyplot as plt

from config import cfg
from networks.NetworkFactory import networkFactory

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Load a trained network and plot its loss')
    parser.add_argument('config_file', help='config file path')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    model = networkFactory(cfg)

    plt.figure(1, figsize=(9, 9))
    plt.title(cfg.OUTPUT.CHECKPOINT_PREFIX)

    plt.plot(range(len(model.total_loss)), model.total_loss, label="Training Loss")
    plt.plot(range(0, len(model.total_loss), cfg.TRAINING.VALIDATION_FREQ), model.val_loss, label="Validation Loss")


    leg = plt.legend(loc='best')

    plt.show()