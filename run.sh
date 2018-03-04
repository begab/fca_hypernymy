#!/bin/bash

for n in 50 5000;
do
    for f in filter-candidates not-filter-candidates;
    do
        for sf in not-sparse-feats sparse-feats;
        do
            for df in not-dag-feats dag-feats;
            do
                evaluate_dag.py --subtask $1 --sparse_dim $2 --negative_samples $n --${sf} --${df} >> ${1}_${2}_${n}_${f}_${sf}_${df}.log 2>&1 &
            done
        done
    done
done
