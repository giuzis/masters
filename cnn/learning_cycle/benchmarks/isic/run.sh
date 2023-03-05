#!/bin/bash

# Declare an array of string with type
declare -a modelArray=("efficientnet_b1" "efficientnet_b2" "efficientnet_b3" "efficientnet_b4" "efficientnet_b5" "efficientnet_b6" "efficientnet_b7" "pnasnet" "vgg19" "seresnext101_32x8d" "resnest101e")
declare -a optArray=("Adam" "AdamW" "Adadelta" "NovoGrad")
declare -a optArray2=("AdamW" "Adadelta" "NovoGrad")
declare -a batch_size=(8 16 32 64)

for val2 in ${optArray2[@]}
do
    for bs in ${batch_size[@]}
    do
        python isic.py with _model_name=efficientnet_b0 _optimizer="$val2" _batch_size="$bs" _lr_init=0.001
    done
done

for model in ${modelArray[@]}
do
    for val2 in ${optArray[@]}
    do
        for bs in ${batch_size[@]}
        do
            python isic.py with _model_name="$model" _optimizer="$val2" _batch_size="$bs" _lr_init=0.001
        done
    done
done
