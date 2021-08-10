#!/bin/sh
platform=('cpu-3d' 'gpu-3d' 'cpu-4d')
for((i=0;i<3;i++))
do
    echo "********** ${platform[i]} train begin **********"
    cd ${platform[i]}
    python genRand.py
    python genCv.py
    ./train.sh 
    cd ..
    echo "********** ${platform[i]} train end **********"
done
