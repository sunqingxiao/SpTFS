#!/bin/sh
platform=('cpu-3d' 'gpu-3d' 'cpu-4d')
for((i=0;i<3;i++))
do
    echo "********** ${platform[i]} calc begin **********"
    cd ${platform[i]}
    python calcPrecision.py
    python calcSpeedup.py
    cd ..
    echo "********** ${platform[i]} calc end **********"
done
