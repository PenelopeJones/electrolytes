#! /bin/bash


mkdir bs020_is020
cd bs020_is020
bin_size=0.20
ion_size=0.20

directory=1.0-60/
dataset=1060
num_files=40
box_length=10.0

min_r_value=0.00
max_r_value=1.4
smoothed=True

python ../../../../descriptor_generator.py --directory ${directory} --dataset ${dataset} --num_files ${num_files} \
    --box_length ${box_length} --bin_size ${bin_size} --ion_size ${ion_size} \
    --min_r_value ${min_r_value} --max_r_value ${max_r_value} --smoothed ${smoothed}



