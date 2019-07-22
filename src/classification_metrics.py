import argparse
import json

from data_management.DatasetFactory import datasetFactory
from networks.ClassifierHelper import Classifier
from config import cfg
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Calculates metrics from output of a Classification network.' +
                                                 ' Run `run_network.py <config> test` first.')
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

    refer = datasetFactory(cfg)

    hamming_loss = 0.0
    TP = np.zeros((cfg.IMG_NET.N_LABELS+1,))
    FP = np.zeros((cfg.IMG_NET.N_LABELS+1,))
    FN = np.zeros((cfg.IMG_NET.N_LABELS+1,))
    total = 0.0

    # load generation outputs
    with open(Classifier.test_output_file(cfg), 'r') as f:
        genData = json.load(f)
        for row in genData:
            total += 1.0
            hamming_loss += row['Hamming_Loss']
            TP[row['TP_classes']] += 1
            FP[row['FP_classes']] += 1
            FN[row['FN_classes']] += 1


    print("Mean Hamming Loss: %3.3f" % (hamming_loss/total))
    print("Mean precision: %3.3f" % (np.sum(TP)/(np.sum(TP)+np.sum(FP))))
    print("Mean recall: %3.3f" % (np.sum(TP)/(np.sum(TP)+np.sum(FN))))

    print("Class\tPrecision\tRecall")
    for idx in range(cfg.IMG_NET.N_LABELS):
        label = refer[0].coco.cats[refer[0].coco_cat_map[idx]]
        print("%s\t%3.3f\t%3.3f" % (label['name'].ljust(20), TP[idx]/(TP[idx]+FP[idx]), TP[idx]/(TP[idx]+FN[idx])))