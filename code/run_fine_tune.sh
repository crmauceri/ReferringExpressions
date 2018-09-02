#!/bin/bash

python code/MaoEtAl_baseline.py models/maoetal_finetune.checkpoint10 train --epochs 15 --learningrate .0001 --dataset sunspot --splitBy boulder --img_root pyutils/refer_python3/data/images/SUNRGBD/ --l2_fraction 1e-5

python code/MaoEtAl_baseline.py models/maoetal_finetune.checkpoint10 train --epochs 15 --learningrate .0001 --dataset sunspot --splitBy boulder --img_root pyutils/refer_python3/data/images/SUNRGBD/ --l2_fraction 0.0

python code/MaoEtAl_baseline.py models/maoetal_sunspot train --epochs 10 --learningrate .001 --dataset sunspot --splitBy boulder --img_root pyutils/refer_python3/data/images/SUNRGBD/ --l2_fraction 0.0

python code/MaoEtAl_baseline.py models/maoetal_sunspot train --epochs 10 --learningrate .001 --dataset sunspot --splitBy boulder --img_root pyutils/refer_python3/data/images/SUNRGBD/ --l2_fraction 1e-5