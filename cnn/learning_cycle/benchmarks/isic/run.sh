#!/bin/bash

# Declare an array of string with type
# declare -a modelArray=("senet154" "resnext101_32x8d" "efficientnet_b0" "efficientnet_b1" "efficientnet_b2" "efficientnet_b3" "efficientnet_b4" "efficientnet_b5" "efficientnet_b6" "efficientnet_b7" "pnasnet" "vgg19" "seresnext101_32x8d" "resnest101e")
declare -a optArray=("Adam" "AdamW" "Adadelta" "NovoGrad")

python isic.py with _model_name=efficientnet_b0 _optimizer=AdamW _batch_size=32 _lr_init=0.001 _data_augmentation=True _PP_crop_mode=center

