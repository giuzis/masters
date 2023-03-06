#!/bin/bash

# Declare an array of string with type
declare -a modelArray=("pnasnet5large")
declare -a optArray=("SGD" "Adam" "AdamW")
declare -a batch_size=(8 16 64)

for val2 in ${optArray[@]}
do
    # python isic.py with _model_name=vgg19 _optimizer="$val2" _batch_size=4 _lr_init=0.001 _csv_path_all_metrics=results/all_metrics_2.csv
    python isic.py with _model_name=vgg19 _optimizer="$val2" _batch_size=4 _lr_init=0.001 _csv_path_all_metrics=results/all_metrics.csv
done
