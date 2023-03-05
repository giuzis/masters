#!/bin/bash

# Declare an array of string with type
declare -a modelArray=("pnasnet5large")
declare -a optArray=("SGD" "Adam" "AdamW" "Adadelta" "NovoGrad")
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
