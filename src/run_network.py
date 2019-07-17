import argparse, os
from csv import DictWriter

import torch

from config import cfg
from networks.NetworkFactory import networkFactory
from data_management.DatasetFactory import datasetFactory
from data_management.ReferExpressionDataset import ReferExpressionDataset
from data_management.refer import REFER

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate referring expression for target object given bounding box and image')
    parser.add_argument('config_file', help='config file path')
    parser.add_argument('mode', help='train/test/comprehend')
    parser.add_argument('--DEBUG', type=bool, default=False, help="Sets random seed to fixed value")

    args = parser.parse_args()

    if args.DEBUG:
        torch.manual_seed(1)

    cfg.merge_from_file(args.config_file)
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
        generated_exp = model.run_generate(dataset, split='test_unique')

        with open(model.generated_output_file(cfg), 'w') as fw:
            fieldnames = ['generated_sentence', 'refID', 'imgID', 'objID', 'objClass']
            writer = DictWriter(fw, fieldnames=fieldnames)

            writer.writeheader()
            for exp in generated_exp:
                writer.writerow(exp)