## Introduction


For more information read the original paper 

["Generation and comprehension of unambiguous object descriptions."](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Mao_Generation_and_Comprehension_CVPR_2016_paper.html
) Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L. Yuille, Kevin Murphy; The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

And our paper

["SUNSpot : An RGB-D dataset with spatial referring expressions."]() Cecilia Mauceri, Martha Palmer, and Christoffer Heckman; ICCV19 CLVL: 3rd Workshop on Closing the Loop Between Vision and Language, 2019.

## Examples

## Installation 

These networks can be run with or with CUDA support. We have tested this project on two machines; 
A MacBook Pro with Intel Core i7 and a Ubuntu Server with Intel Xeon Processor and Nvidia P6000 cards. 

Install the following packages in your python environment. We recommend using a new anaconda environment, 
to avoid messing up other installations.
- pytorch 1.1
- Cython
- tqdm
- scikit-image

```bash
conda create --name refexp_generation
conda activate refexp_generation

# Check https://pytorch.org for appropriate pytorch package
# The following installs vanilla pytorch without CUDA
conda install pytorch torchvision -c pytorch 

conda install Cython tqdm scikit-image
```

- Install the [cocoapi](https://github.com/cocodataset/cocoapi)
```bash
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI/
make
pip install -e .
cd ../..
```

- For evaluation, install [nlg-eval](https://github.com/Maluuba/nlg-eval) 
```bash
# Install Java 1.8.0 (or higher). Then run:

git clone https://github.com/Maluuba/nlg-eval.git
cd nlg-eval

# Install the Python dependencies.
# It may take a while to run because it's downloading some files. You can instead run `pip install -v -e .` to see more details.
pip install -e .

# Download required data files.
nlg-eval --setup

cd ..
```

## Datasets

### Publicly available datasets

This repository is distributed with the SUNSpot referring expressions annotations. To use these, you will additionally 
need to download [the SUNRGBD images](http://rgbd.cs.princeton.edu) and add them into the ``pyutils/refer_python3/data/images/SUNRGBD`` directory. 

Download additional referring expressions datasets from https://github.com/lichengunc/refer 

### Make your own referring expressions dataset

For any image dataset with [mscoco style annotations](), you can add referring expressions as a pickle file with the structure:



Put your mscoco annotations and referring expressions pickle in the 
/data/<your_dataset>/ directory. 

You can check if it loads correctly by running 
```bash
python code/refer.py --data_root /data/<your_dataset>/ --img_root <img_root>
```

##  How to Use Networks

### Training

### Testing 

### Generation

### Comprehension

## License 
Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for additional details