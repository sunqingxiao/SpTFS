#!/bin/sh
platform=('cpu-3d' 'gpu-3d' 'cpu-4d')
for((i=2;i<3;i++))
do
    echo "********** ${platform[i]} test begin **********"
    cd ${platform[i]}
    python genCv.py
    ./test.sh 
    cd ..
    echo "********** ${platform[i]} test end **********"
done
