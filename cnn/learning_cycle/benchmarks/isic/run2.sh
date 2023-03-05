#!/bin/bash

# Declare an array of string with type
declare -a modelArray=("efficientnet_b0" "efficientnet_b1" "efficientnet_b2" "efficientnet_b3" "efficientnet_b4" "efficientnet_b5" "efficientnet_b6" "efficientnet_b7")

for val2 in ${modelArray[@]}
do
    python isic.py with _model_name="$val2" _optimizer=AdamW _batch_size=16 _lr_init=0.001
done
for val2 in ${modelArray[@]}
do
    python isic.py with _model_name="$val2" _optimizer=AdamW _batch_size=32 _lr_init=0.001
done
