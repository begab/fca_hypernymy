#!/bin/bash

for m in include exclude;
do
  for s in 3 4 5;
  do
    for d in 200 300;
    do
      for a in sg cbow;
      do
        python evaluate_dag.py --subtask $1 --dense_archit $a --sparse_dimensions $d --sparse_density 0.${s} --${m}-sparse-feats >> ${1}.log 2>&1
      done
    done
  done
done
