#!/bin/bash

# Command line arguments:
# model prefix total_iterations gpu_devices

r=0.001
for i in ((i=0; i<${3}; i++))
do
    n=$[$i*20]
    CUDA_VISIBLE_DEVICES=${4} python ${1} train ${2} --epochs $n --learningrate $r --batch_size 16
    r=$(echo $r/2.0|bc -l)
    #echo $n
    #echo $r

    CUDA_VISIBLE_DEVICES=${4} python ${1} comprehend ${2}
    CUDA_VISIBLE_DEVICES=${4} python ${1} test ${2}
done