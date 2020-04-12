#! /bin/bash


n=8000
split=0
prior=uninformative
alpha0=1.0
beta0=1.0e-11
w0_scalar=1.0
run_number=10
max_iterations=300

for dataset in 0580 1080; do
  for label in bs010_is010 bs015_is010 bs020_is010; do
    path_to_dir="data/results/${dataset}/${label}"
    echo $path_to_dir
    for K in 1 2 3; do
      python inference.py --directory ${path_to_dir} --dataset ${dataset} --n ${n} \
      --split ${split} --K ${K} --prior ${prior} --alpha0 ${alpha0} --beta0 ${beta0} \
      --w0_scalar ${w0_scalar} --run_number ${run_number} --max_iterations ${max_iterations}
    done
  done
done
