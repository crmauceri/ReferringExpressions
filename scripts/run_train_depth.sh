#!/bin/bash

## MaoEtAl_depth
# positional arguments:
#  mode                  train/test/comprehend
#  checkpoint_prefix     Filepath to save/load checkpoint. If file exists,
#                        checkpoint will be loaded
#
# optional arguments:
#  -h, --help            show this help message and exit
#  --img_root IMG_ROOT   path to the image directory
#  --depth_root DEPTH_ROOT
#                        path to the image directory
#  --data_root DATA_ROOT
#                        path to data directory
#  --dataset DATASET     dataset name
#  --version VERSION     team that made the dataset splits
#  --epochs EPOCHS       Number of epochs to train (Default: 1)
#  --hidden_dim HIDDEN_DIM
#                        Size of LSTM embedding (Default:100)
#  --dropout DROPOUT     Dropout probability
#  --l2_fraction L2_FRACTION
#                        L2 Regularization Fraction
#  --learningrate LEARNINGRATE
#                        Adam Optimizer Learning Rate
#  --batch_size BATCH_SIZE
#                        Training batch size
#  --DEBUG DEBUG         Sets random seed to fixed value

CUDA_VISIBLE_DEVICES=${1} python code/MaoEtAl_depth.py train models/maoetal_depth  \
    --img_root datasets/coco/images/train2014/ --depth_root datasets/coco/images/megadepth/ --data_root datasets/coco/refcocog \
    --dataset refcocog --version google  \
    --epochs 30 \

