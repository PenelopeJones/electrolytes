#! /bin/bash


mkdir bs015_is015
cd bs015_is015
bin_size=0.15
ion_size=0.15

directory=2.0-40/
dataset=2040
num_files=35
box_length=10.0

min_r_value=0.00
max_r_value=1.4
smoothed=True

python ../../../../descriptor_generator.py --directory ${directory} --dataset ${dataset} --num_files ${num_files} \
    --box_length ${box_length} --bin_size ${bin_size} --ion_size ${ion_size} \
    --min_r_value ${min_r_value} --max_r_value ${max_r_value} --smoothed ${smoothed}

