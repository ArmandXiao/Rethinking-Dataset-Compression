#!/bin/bash

# TO RUN THE SCRIPT:
# bash script/reproduce.sh

reproduce() {
    setting=$1
    model=$2
    method=$3
    ipc=$4
    train_dir="he-yang/2025-rethinkdc-imagenet-${method}-ipc-${ipc}"
    output_dir="./result/${method}_${model}_ipc${ipc}_${setting}"

    rethinkdc $train_dir \
        "--${setting}" \
        --model $model \
        --output-dir $output_dir \
        --ipc $ipc 
}

for method in "random"; do
    reproduce soft resnet18 $method 10
    # reproduce soft resnet18 $method 50
    # reproduce soft resnet18 $method 100
done