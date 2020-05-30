# SpTFS

This directory contains the implementation of `SpTFS` and will be further updated later.

Note that we have conducted the safety checks when collecting the training data. Specifically, we repeat the excution of MTTKRP under each tensor format for 10 times, and collect the mean execution time for each tensor format. When collecting prediction accuracies, we repeat the process of training and inference 5 times, and collected the means, medians and percentiles (i.e., 20%, 40%, 60% and 80%) of prediction accuracies. The scripts for collecting data and safety checks are detailed in the `scripts` directory.

Now take the prediction for 3-D tensors as an example.

## sample

### 1. sample sparsity features

    cd 3d-tensor/sample

    python Dl3dSamplyBase.py <tensorlist> <outputlist>

    python Dl3dSamplyCsf.py <tensorlist> <outputlist>

### 2. sample fixed-size matrices (tensor transformation)

    cd 3d-tensor/sample
    
    python Dl3dSampleFlatten.py <tensorlist> <resolution>

    python Dl3dSampleMap.py <tensorlist> <resolution>

## train

    cd 3d-tensor

    python Dl3dCv.py

    python Dl3dNet.py train <train data> <test data>

## test

    cd 3d-tensor

    python Dl3dNet.py test <train data> <test data>
