#!/bin/bash

# Declare an array of string with type
declare -a modelArray=("resnext101_32x8d" "efficientnet_b0" "efficientnet_b1" "efficientnet_b2" "efficientnet_b3" "efficientnet_b4" "efficientnet_b5" "efficientnet_b6" "efficientnet_b7" "vgg19" "seresnext101_32x8d" "resnest101e")
declare -a optArray=("SGD")
declare -a batch_size=(8 16 64)

for model in ${modelArray[@]}
do
    for val2 in ${optArray[@]}
    do
        for bs in ${batch_size[@]}
        do
            python isic.py with _model_name="$model" _optimizer="$val2" _batch_size="$bs" _lr_init=0.001
        done
    done
    for val2 in ${optArray[@]}
    do
        for bs in ${batch_size[@]}
        do
            python isic.py with _model_name="$model" _optimizer="$val2" _batch_size="$bs" _lr_init=0.0001
        done
    done
done
