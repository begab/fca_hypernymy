#/!bin/bash

if [[ $# -lt 1 ]];
then
    SUBTASK="1A 1B 1C 2A 2B"
else
    SUBTASK=$1
fi

for st in $SUBTASK;
do
    for sd in 200 300;
    do
        for v in cbow sg;
        do
            for s in not-sparse-feats sparse-feats;
            do
                for d in not-dag-feats dag-feats;
                do
                    python evaluate_dag.py --subtask $st --dense_archit $v --sparse_dim $sd --negative_samples 50 --sparse_density 0.3 --${s} --filter-candidates --${d} --not-save-gpickle >> table1experiments_${st}.log 2>&1;
                done
            done
        done
    done
done
