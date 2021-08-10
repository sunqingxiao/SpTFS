#!/bin/bash
## create model dir
cd model
for((i=0;i<3;i++))
do
    mkdir -p mode${i}
    cd mode${i}
    for((j=0;j<5;j++))
    do
        mkdir -p cv${j}
    done
    cd ..
done
cd ..

## execute 5-fold training
for((i=0;i<3;i++))
do
    for((j=0;j<5;j++))
    do
        python Dl3dNet.py test data/mode${i}_cv${j}_train.npz data/mode${i}_cv${j}_test.npz model/mode${i}/cv${j} result/mode${i}_cv${j}_WrongIds.npz
        sleep 3
    done
done
