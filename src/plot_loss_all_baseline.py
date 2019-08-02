import argparse, os
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

    plt.figure(1, figsize=(9, 9))
    models = [f[:-4] for f in os.listdir('models') if 'maoetal_finetune' in f or 'maoetal_sunspot' in f] #'maoetal_baseline' in f]#
    for model_name in models:
        cfg.merge_from_list(['OUTPUT.CHECKPOINT_PREFIX', model_name])
        model = networkFactory(cfg)

        plt.plot(range(len(model.total_loss)), model.total_loss, label="Training Loss: {}".format(model_name))
    #plt.plot(range(0, len(model.total_loss), cfg.TRAINING.VALIDATION_FREQ), model.val_loss, label="Validation Loss: {}".format(model_name))


    leg = plt.legend(loc='best')

    plt.show()