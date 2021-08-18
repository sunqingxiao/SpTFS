#!/bin/bash

for((i=0;i<16;i++))
do
    python gen4dTensor.py &
done

