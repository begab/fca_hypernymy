#!/bin/bash

for a in sg cbow;
do
  for s in 3 4 5;
  do
    for d in 200 300;
    do
      for m in sparse-feats not-sparse-feats;
      do
         python evaluate_dag.py --subtask $1 --dense_archit $a --sparse_dimensions $d --sparse_density 0.${s} --not-filter-candidates --not-save-gpickle >> ${1}_full${2}.log 2>&1
      done
    done
  done
done
