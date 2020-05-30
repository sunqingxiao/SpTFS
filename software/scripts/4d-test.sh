#!/bin/bash
OP=('train' 'test')
for((k=0;k<10;k++))
do
    for((i=0;i<4;i++))
    do
        for((j=0;j<5;j++))
        do
            python Dl4dNet.py ${OP[0]} data/mode${i}_cv${j}_train.npz data/mode${i}_cv${j}_test.npz
            python -u Dl4dNet.py ${OP[1]} data/mode${i}_cv${j}_train.npz data/mode${i}_cv${j}_test.npz | tee -a data/mode${i}_cv${j}_counter{k}.log
            sleep 1
        done
    done
done
