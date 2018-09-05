#!/bin/bash

python code/MaoEtAl_baseline.py train models/maoetal_finetune.checkpoint10 --epochs 15 --learningrate .0001 --dataset sunspot --splitBy boulder --img_root pyutils/refer_python3/data/images/SUNRGBD/ --l2_fraction 1e-5

python code/MaoEtAl_baseline.py train models/maoetal_finetune.checkpoint10 --epochs 15 --learningrate .0001 --dataset sunspot --splitBy boulder --img_root pyutils/refer_python3/data/images/SUNRGBD/ --l2_fraction 0.0

python code/MaoEtAl_baseline.py train models/maoetal_sunspot --epochs 10 --learningrate .001 --dataset sunspot --splitBy boulder --img_root pyutils/refer_python3/data/images/SUNRGBD/ --l2_fraction 0.0

python code/MaoEtAl_baseline.py train models/maoetal_sunspot --epochs 10 --learningrate .001 --dataset sunspot --splitBy boulder --img_root pyutils/refer_python3/data/images/SUNRGBD/ --l2_fraction 1e-5

python code/MaoEtAl_baseline.py train models/maoetal_finetune_batch --l2_fraction 1e-5 --batch_size 32 --learningrate 0.00001 --epochs 35 --dataset sunspot --splitBy boulder --img_root pyutils/refer_python3/data/images/SUNRGBD/