#!/bin/bash

# Declare an array of string with type
declare -a StringArray=("senet154" "resnext101_32x4d" "resnext101_32x8d" "efficientnet_b0" "efficientnet_b1" "efficientnet_b2" "efficientnet_b3" "efficientnet_b4" "efficientnet_b5" "efficientnet_b6" "efficientnet_b7" "pnasnet" "vgg19" "seresnext101_32x4d" "seresnext101_32x8d" "resnest101e")

# Iterate the string array using for loop
for val in ${StringArray[@]}; do
    python3.7 isic.py with _model_name="$val"
done