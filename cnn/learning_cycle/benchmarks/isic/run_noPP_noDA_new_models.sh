#!/bin/bash

# Declare an array of string with type
declare -a modelArray=("resnext101_32x8d" "efficientnet_b0" "efficientnet_b1" "efficientnet_b2" "efficientnet_b3" "efficientnet_b4" "tf_efficientnet_b5" "tf_efficientnet_b6" "vgg19" "seresnext101_32x8d" "resnest101e" "pnasnet5large" "densenet121")
declare -a optArray=("SGD" "Adam" "AdamW")
declare -a batch_size=(8 16 64)

python isic.py with _model_name=resnest50d _lr_init=0.001 _batch_size=8 _optimizer=AdamW

