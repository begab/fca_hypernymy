#/!bin/bash

for t in 1A 1B 1C 2A 2B;
do
    for sd in 200 300;
    do
        for v in cbow sg;
        do
            for s in not-sparse-feats sparse-feats;
            do
                for d in not-dag-feats dag-feats;
                do
                    python evaluate_dag.py --subtask $t --dense_archit $v --sparse_dim $sd --negative_samples 50 --sparse_density 0.3 --${s} --filter-candidates --${d} --not-save-gpickle >> table1experiments.log 2>&1;
                done
            done
        done
    done
done
