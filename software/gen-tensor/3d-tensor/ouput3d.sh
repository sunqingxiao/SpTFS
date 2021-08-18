#!/bin/bash

for((i=0;i<16;i++))
do
    python gen3dTensor.py &
done

#cp -r hicoo-datasets/ splatt-datasets/
