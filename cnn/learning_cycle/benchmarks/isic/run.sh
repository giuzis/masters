#!/bin/bash

# Declare an array of string with type
# declare -a modelArray=("senet154" "resnext101_32x8d" "efficientnet_b0" "efficientnet_b1" "efficientnet_b2" "efficientnet_b3" "efficientnet_b4" "efficientnet_b5" "efficientnet_b6" "efficientnet_b7" "pnasnet" "vgg19" "seresnext101_32x8d" "resnest101e")
declare -a modelArray=("efficientnet_b0" "efficientnet_b1" "efficientnet_b2" "efficientnet_b3" "efficientnet_b4" "efficientnet_b5" "efficientnet_b6" "efficientnet_b7" "pnasnet" "seresnext101_32x8d" "resnest101e" "senet154")
declare -a optArray=("SGD" "Adam" "AdamW" "Adadelta" "NovoGrad")

# Iterate the string array using for loop
for model in ${modelArray[@]}
do
    for val in ${optArray[@]}
    do
        python isic.py with _model_name="$model" _optimizer="$val" _batch_size=4 _lr_init=0.001
        # python isic.py with _model_name=vgg19 _optimizer="$val" _batch_size=32 _lr_init=0.0001
    done

    for val in ${optArray[@]}
    do
        # python isic.py with _model_name=resnext101_32x4d _optimizer="$val" _batch_size=4 _lr_init=0.001
        python isic.py with _model_name="$model" _optimizer="$val" _batch_size=32 _lr_init=0.001
    done

    for val in ${optArray[@]}
    do
        # python isic.py with _model_name=resnext101_32x4d _optimizer="$val" _batch_size=4 _lr_init=0.001
        python isic.py with _model_name="$model" _optimizer="$val" _batch_size=32 _lr_init=0.0001
    done
done