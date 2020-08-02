# SpTFS sampling

This directory contains the sampling process of 3-D tensors. 

## compile

   gcc -o libsample.so -shared -fPIC sample.c

## run

   python Dl3dSample.py <tensor.list> <resolution> <out.npz>
