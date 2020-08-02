# SpTFS sampling

This directory contains the sampling process of 4-D tensors. 

## compile

   gcc -o libsample.so -shared -fPIC sample.c

## run

   python Dl4dSample.py <tensor.list> <resolution> <out.npz>
