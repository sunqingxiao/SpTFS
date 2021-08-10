#!/bin/bash
## execute 5-fold training
for((i=0;i<3;i++))
do
    for((j=0;j<5;j++))
    do
        python kmeans.py ../data/mode${i}_cv${j}_train.npz ../data/mode${i}_cv${j}_test.npz result/result_mode${i}_cv${j}.npz
        sleep 3
    done
done
