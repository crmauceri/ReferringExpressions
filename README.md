## Prerequisites

Available with conda
- [pytorch](https://pytorch.org)
- scipy
- h5py
- opencv
- matplotlib
- scikit-image
- cython

Use pip install
- dominate
- tensorboardX

Recommended

- CUDA 8.0

## Install

- Clone recursive 
```
git clone --recurse-submodules https://github.com/crmauceri/ReferExpGeneration.git
```
- Install pyutils
```
cd pyutils/
./install.sh
cd ..
```
- Make directories for models and output
```
mkdir models
mkdir checkpoints
mkdir output
```