#!/bin/bash

# Command line arguments:
# model new_filename epochs learning_rate

cp models/${1} models/${2}

code/MaoEtAl_baseline.py models/${2} train --epochs ${3} --learningrate ${4} --dataset sunspot --splitBy boulder 