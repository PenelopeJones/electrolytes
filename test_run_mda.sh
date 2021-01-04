#! /bin/bash
n=8000
prior=uninformative
alpha0=1.0
beta0=1.0e-11
w0_scalar=1.0
run_number=8
max_iterations=300
label=bs019_is019

for c in 1 2; do
  for type in an cat; do
    path_to_dir="data/sota_results/litfsi_dmedol/${c}m/${label}/"
    echo $path_to_dir
    for K in 4 5; do
      python inference_mda.py --directory ${path_to_dir} --conc ${c} --type ${type} --n ${n} \
      --K ${K} --prior ${prior} --alpha0 ${alpha0} --beta0 ${beta0} \
      --w0_scalar ${w0_scalar} --run_number ${run_number} --max_iterations ${max_iterations}
    done
  done
done