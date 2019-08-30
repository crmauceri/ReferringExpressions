## Introduction

For more information read the original paper 

["Generation and comprehension of unambiguous object descriptions."](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Mao_Generation_and_Comprehension_CVPR_2016_paper.html
) Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L. Yuille, Kevin Murphy; The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

And our paper

["SUNSpot : An RGB-D dataset with spatial referring expressions."]() Cecilia Mauceri, Martha Palmer, and Christoffer Heckman; ICCV19 CLVL: 3rd Workshop on Closing the Loop Between Vision and Language, 2019.


## Installation 

These networks can be run with or with CUDA support. We have tested this project on two machines; 
A MacBook Pro with Intel Core i7 and a Ubuntu Server with Intel Xeon Processor and Nvidia P6000 cards. 

1. Install the following packages in your python environment. We recommend using a new anaconda environment, 
to avoid messing up other installations.
    - pytorch 1.1
    - Cython
    - tqdm
    - scikit-image
    - yacscond
    - tensorflow (for using tensorboard)
    - future
    
    ```bash
    conda create --name refexp_generation
    conda activate refexp_generation
    
    # Check https://pytorch.org for appropriate pytorch package
    # The following installs vanilla pytorch without CUDA
    conda install pytorch torchvision -c pytorch 
    
    conda install Cython tqdm scikit-image future
    pip install yacs
 
    # Check https://www.tensorflow.org/install for appropriate tensorflow package
    # The following installs vanilla tensorflow without CUDA
    pip install tensorflow
    ```

2. Install the [cocoapi](https://github.com/cocodataset/cocoapi)
    ```bash
    git clone https://github.com/cocodataset/cocoapi.git
    cd cocoapi/PythonAPI/
    make
    pip install -e .
    cd ../..
    ```

3. For evaluation, install [nlg-eval](https://github.com/Maluuba/nlg-eval) 
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

### SUNSpot

1. Make a <data_root> directory for SUNSpot, for example `data/sunspot/`.
1. Download [the SUNRGBD images](http://rgbd.cs.princeton.edu). The directory you save them in will be your <img_root>.
2. Download [the SUNSpot annotations]() and unzip them in <data_root>

### Publicly available datasets

Download additional referring expressions datasets from https://github.com/lichengunc/refer 

We use [MegaDepth](https://github.com/lixx2938/MegaDepth) to generate synthetic depth images for the COCO dataset.

### Make your own referring expressions dataset

1. Make a directory for your dataset, for example `data/<your_dataset>/`. This will be your <data_root>.

2. Make a [COCO style annotation file](http://cocodataset.org/#format-data) describing your images and bounding box annotations 
and save as `<data_root>/instance.json` 

3. Save your referring expressions as a pickle file, `<data_root>/ref(<version_name>).p`, with the structure:

    ```
    refs: list of dict [
        {
        image_id : unique image id (int)
        split : train/test/val (str)
        sentences : list of dict [
            {
            tokens : tokenized version of referring expression (list of str)
            raw : unprocessed referring expression (str)
            sent : referring expression with mild processing, lower case, spell correction, etc. (str)
            sent_id : unique referring expression id (int)
            } ...
        ]
        file_name : file name of image relative to img_root (str)
        category_id : object category label (int)
        ann_id : id of object annotation in instance.json (int)
        sent_ids : same ids as nested sentences[...][sent_id] (list of int)
        ref_id : unique id for refering expression (int)
        } ...
    ] 
    ```

4. Optional : If you have depth images, make a mapping file, <data_root>/depth.json which maps image ids to depth file paths
    ```
    {
        <image_id> : file name of depth image relative to depth_root  (str)
        ...    
    }
    ```

4. You can check if the dataset loads correctly by running 
    ```bash
    python src/data_management/refer.py --data_root <data_root> --img_root <img_root> --depth_root <depth_root> --version <version_name> --dataset <dataset_name>
    ```

##  How to Use Networks

### Config Files

We use the [yacs](https://github.com/rbgirshick/yacs) config system. Configurations are set in three spots

1. [Default configurations](src/config/defaults.py)
2. [Configuration files](configs/)
3. Command line overrides - for example you can change the number of epochs from what is specified in the config file with 

    python src/run_network.py <config_file> train TRAINING.N_EPOCH 60

#### Configs referenced in ["SUNSpot : An RGB-D dataset with spatial referring expressions."]()

1.  Baseline - [configs/refcocog_baseline.yaml](configs/refcocog_baseline.yaml)
2.  Baseline+fine - [configs/sunspot_baseline.yaml](configs/sunspot_baseline.yaml)
3.  VGG - [configs/refcocog_baseline_custom_vgg.yaml](configs/refcocog_baseline_custom_vgg.yaml)
4.  VGG+D - [configs/refcocog_depth_baseline.yaml](configs/refcocog_depth_baseline.yaml)
5.  VGG+fine - [configs/sunspot_baseline_custom_vgg.yaml](configs/sunspot_baseline_custom_vgg.yaml)
6.  VGG+D+fine - [configs/sunspot_depth_baseline.yaml](configs/sunspot_depth_baseline.yaml)

The image classification networks which were pretrained for VGG+D and VGG+D+fine are [mscoco_depth_classification_l2_10e-5_BCE.yaml](configs/mscoco_depth_classification_l2_10e-5_BCE.yaml)


### Training

Define a config file and run the following

    python src/run_network.py <config_file> train <additional config variables>

### Testing

    python src/run_network.py <config_file> test <additional config variables>
    
Will run the most recently saved checkpoint. It will also save generated referring expressions and comprehension results in a file `output/cfg.OUTPUT.CHECKPOINT_PREFIX_cfg.DATASET.NAME_<data_split>.json`

Choose which data splits to run on using the following config variables 

    # Defaults
    cfg.TEST.DO_TRAIN = True # Run on train set
    cfg.TEST.DO_VAL = True # Run on val set
    cfg.TEST.DO_TEST = True # Run on test set
    cfg.TEST.DO_ALL = False # If false, only random sample of <=10000 images are tested from each set

For referring expressions networks, to calculate evaluation metrics, run 

    python src/mt_metrics.py <config_file> <output_file>
    
For image classification networks, use 

    python src/classification_metrics.py <config_file> <output_file>
    

## License 
Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for additional details