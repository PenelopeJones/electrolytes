#! /bin/bash


mkdir bs015_is010
cd bs015_is010
bin_size=0.15
ion_size=0.10

directory=0.5-80/
dataset=0580
num_files=49
box_length=8.0

min_r_value=0.00
max_r_value=1.25
smoothed=True

python ../../../../descriptor_generator.py --directory ${directory} --dataset ${dataset} --num_files ${num_files} \
    --box_length ${box_length} --bin_size ${bin_size} --ion_size ${ion_size} \
    --min_r_value ${min_r_value} --max_r_value ${max_r_value} --smoothed ${smoothed}



