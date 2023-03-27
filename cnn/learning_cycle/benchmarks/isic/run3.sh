#!/bin/bash

# Declare an array of string with type
declare -a modelArray=("tf_efficientnet_b5" "tf_efficientnet_b6" "densenet121")
declare -a optArray=("SGD" "Adam" "AdamW")
declare -a batch_size=(4 8 16 32 64)
declare -a lr_init=(0.001 0.0001)

for model in "${modelArray[@]}"
do
    for opt in "${optArray[@]}"
    do
        for batch in "${batch_size[@]}"
        do
            for lr in "${lr_init[@]}"
            do
                python isic.py with _model_name=$model _lr_init=$lr _batch_size=$batch _optimizer=$opt
            done
        done
    done
done
