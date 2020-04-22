# SpTFS

Take the prediction for 3-D tensors as an example.

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

    python Dl3dNet.py train <train data> <test data> <output>

## test

    cd 3d-tensor

    python Dl3dNet.py test <train data> <test data> <output>
