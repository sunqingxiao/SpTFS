# SpTFS

This directory contains the implementation of `SpTFS` and will be further updated later.

Note that we have conducted the safety checks when collecting the training data. Specifically, we repeat the excution of MTTKRP under each tensor format for `10` times, and collect the mean and stdev execution time for each tensor format. 

When collecting prediction accuracies, we repeat the process of training and inference `5` times, and collected the means, medians and percentiles (i.e., `20%`, `40%`, `60%` and `80%`) of prediction accuracies. The scripts for collecting data and safety checks are detailed in the `scripts` directory.

Now take the prediction for 3-D tensors as an example.

## sample
    
    python Dl3dSample.py <tensor.list> <resolution> <out.npz>

## train

    python genRand.py

    python gen3dCv.py

    python Dl3dNet.py train <train data> <test data> <model data> <result data>

## test

    cd 3d-tensor

    python Dl3dNet.py test <train data> <test data> <model data> <result data>

## calculate precision/speedup
    
    python calPrecision.py

    python calcSpeedup.py
