# ObjectRef

1. [Data](#data)
    1. [Data splits](#data-splits)
    1. [Data preprocessing](#data-preprocessing)
2. [Modules](#modules)
3. [Classifiers](#classifiers)
    1. [Training Arguments](#training-arguments)
4. [Analysis and Visualizations](#analysis-and-visualization)

## Install

`pip -e pyutils/`

## Data

## Arguments

positional arguments| def
--- | ---
   mode     |   train/test
   checkpoint_file  |     Filepath to save/load checkpoint. If file exists, checkpoint will be loaded
   
optional arguments | def
--- | ---
  -h, --help | show this help message and exit
  --data_root, --dataset, --split  |  REFER arguments
  --epochs EPOCHS | Number of epochs to train (Default: 1)
  --hidden_dim HIDDEN_DIM |Size of LSTM embedding (Default:100)

Notes:
- Shuffles examples between each epoch
- Does backprop on one example at a time, no mini-batching

## Analysis and Visualizations
