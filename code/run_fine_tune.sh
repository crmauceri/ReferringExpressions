#!/bin/bash

python code/MaoEtAl_baseline.py train models/maoetal_finetune.checkpoint10 --epochs 30 --learningrate .0001 --dataset sunspot --splitBy boulder --img_root pyutils/refer_python3/data/images/SUNRGBD/ --l2_fraction 1e-5

python code/MaoEtAl_baseline.py train models/maoetal_finetune.checkpoint10 --epochs 30 --learningrate .0001 --dataset sunspot --splitBy boulder --img_root pyutils/refer_python3/data/images/SUNRGBD/ --l2_fraction 0.0

python code/MaoEtAl_baseline.py train models/maoetal_sunspot --epochs 30 --learningrate .001 --dataset sunspot --splitBy boulder --img_root pyutils/refer_python3/data/images/SUNRGBD/ --l2_fraction 0.0

python code/MaoEtAl_baseline.py train models/maoetal_sunspot --epochs 30 --learningrate .001 --dataset sunspot --splitBy boulder --img_root pyutils/refer_python3/data/images/SUNRGBD/ --l2_fraction 1e-5
