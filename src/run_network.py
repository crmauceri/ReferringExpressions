import argparse, json
from csv import DictWriter

import torch

from config import cfg
from networks.NetworkFactory import networkFactory
from data_management.DatasetFactory import datasetFactory

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate referring expression for target object given bounding box and image')
    parser.add_argument('config_file', help='config file path')
    parser.add_argument('mode', help='train/test/comprehend')
    parser.add_argument('--DEBUG', type=bool, default=False, help="Sets random seed to fixed value")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    if args.DEBUG:
        torch.manual_seed(1)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    dataset = datasetFactory(cfg)

    model = networkFactory(cfg) #Loads network from checkpoint if exists

    if args.mode == 'train':
        print("Start Training")
        total_loss = model.run_training(dataset, cfg)
    if args.mode == 'comprehend':
        print("Start Comprehension")
        if args.dataset=='refcocog':
            output = model.run_comprehension(dataset, split='val')
        else:
            output = model.run_comprehension(dataset, split='test')

        with open(model.comprehension_output_file(cfg), 'w') as fw:
            fieldnames = ['gt_sentence', 'refID', 'imgID', 'objID', 'objClass', 'p@1', 'p@2', 'zero-shot']
            writer = DictWriter(fw, fieldnames=fieldnames)

            writer.writeheader()
            for exp in output:
                writer.writerow(exp)

    if args.mode == 'test':
        print("Start Testing")

        if isinstance(dataset, tuple):
            train_dataset = dataset[0]
            test_dataset = dataset[1]
            val_dataset = dataset[2]
        else:
            train_dataset = dataset
            test_dataset = dataset
            val_dataset = dataset

        output = model.run_test(test_dataset, split='test')

        with open(model.test_output_file(cfg), 'w') as fw:
            json.dump(output, fw)

        print("Test output saved : {}".format(model.test_output_file(cfg)))