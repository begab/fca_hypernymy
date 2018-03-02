#!/bin/bash

for a in sg cbow;
do
  for s in 3 4 5;
  do
    for d in 200 300;
    do
      for m in sparse-feats not-sparse-feats;
      do
         nice -n6 python evaluate_dag.py --subtask $1 --dense_archit $a --sparse_dim $d --sparse_density 0.${s} --filter-candidates --not-save-gpickle >> ${1}_${a}_${s}_${d}_${m}.log 2>&1 &
      done
    done
  done
done
