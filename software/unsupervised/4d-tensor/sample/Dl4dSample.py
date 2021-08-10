#!/usr/env python

"""
This is a wrapper function for the sampling
"""

import sys, os
import numpy as np
import ctypes
from ctypes import *

c_int_p = POINTER(c_int)
c_float_p = POINTER(c_float)
c_char_p = POINTER(c_char)

def dbg(data):
    """ print data
    """
    print(data)

def compare(x, y):
    """ compare two values
    """
    if x > y:
        return 1
    else:
        return 0

def tns_to_dict(sp4dtns):
    """ convert sparse 4d data to python dict
    """
    return {'id': None, 'dim0': sp4dtns.dim0, 'dim1': sp4dtns.dim1, 'dim2': sp4dtns.dim2, 'dim3': sp4dtns.dim3, 'nnz': sp4dtns.nnz}

class featureSample(object):
    """ sampling for features 
    """
    def base_sample(self, tns_dir):
        """obtain the feature set of a tensor
        """
        lib = CDLL('./libsample.so')  
        input_dir = tns_dir.encode()
        lib.getBaseFeatures.argtypes = [c_char_p]
        lib.getBaseFeatures.restype = c_float_p
        baseFeatures = lib.getBaseFeatures(input_dir)
        return baseFeatures

    def csf_sample(self, tns_dir):
        """obtain the feature set of a tensor
        """
        lib = CDLL('./libsample.so')  
        input_dir = tns_dir.encode()
        lib.getCsfFeatures.argtypes = [c_char_p]
        lib.getCsfFeatures.restype = c_float_p
        csfFeatures = lib.getCsfFeatures(input_dir)
        return csfFeatures

    def tns_batch(self, tensorlist):
        """ obtain the feature set of a batch of tensors
        """
        filenames = []
        ftset_batch = []

        ## record the file dir
        filelist = open(tensorlist)
        for line in filelist:
            list_seg = line.split()
            filenames.append(list_seg[-1])
        filelist.close()

        ## record the feature sets of tensors
        batch_size = len(filenames)
        for findex in range(0, batch_size):
            baseset = self.base_sample(filenames[findex])
            csfset = self.csf_sample(filenames[findex])
            totalset = []
            for i in range(0, 26):
                totalset.append(baseset[i])
            for i in range(0, 13):
                totalset.append(csfset[i])
            ftset_batch.append(totalset)
            print('**** the {}th tns feature sampled finished ({} in total) ****'.format(findex, batch_size))
        return ftset_batch

class flattenSample(object):
    """ wrapper of 4d tensor flatten sampling
    """
    def tns4d_Sample(self, tns_dir, output_resolution):
        ## the basic information of sparse 4d tensor
        lib = CDLL('./libsample.so')  
        input_dir = tns_dir.encode()
        lib.getFlattenInput.argtypes = [c_char_p, c_int]
        lib.getFlattenInput.restype = c_int_p
        flattenInput = lib.getFlattenInput(input_dir, output_resolution)

        ## each mode of 4d-tensor needs one mat (combination of dims)
        flattenImgs = np.zeros((4, output_resolution, output_resolution), dtype='int32')
        for i in range(0, 4):
            for j in range(0, output_resolution):
                for k in range(0, output_resolution):
                    flattenImgs[i][j][k] = flattenInput[output_resolution*(i*output_resolution+j)+k]       
        return flattenImgs

    def tns4d_batch(self, tensorlist, output_resolution):
        """ return data of a batch of 4d tensors
        """
        dimensions = 4
        formats = 5
        filenames = []

        ## record the file dir
        filelist = open(tensorlist)
        for line in filelist:
            list_seg = line.split()
            filenames.append(list_seg[-1])
        filelist.close()

        ## get the batch data of 4d tensors and their basic info
        batch_size = len(filenames)
        tensor_batch = np.zeros((batch_size, dimensions, output_resolution, output_resolution), dtype='int32')
        for findex in range(0, batch_size):
            img = self.tns4d_Sample(filenames[findex], RES)
            tensor_batch[findex, :, :, :] = img
            print('**** the {}th tns flattening sampled finished ({} in total) ****'.format(findex, batch_size))       
        return tensor_batch

class mapSample(object):
    """ wrapper of 4d map sampling
    """
    def tns4d_Sample(self, tns_dir, output_resolution):

        lib = CDLL('./libsample.so')  
        input_dir = tns_dir.encode()
        lib.getMapInput.argtypes = [c_char_p, c_int]
        lib.getMapInput.restype = c_int_p
        mapInput = lib.getMapInput(input_dir, output_resolution)

        ## 4d-tensor needs six mats (combination of dims)
        mapImgs = np.zeros((6, output_resolution, output_resolution), dtype='int32')
        for i in range(0, 6):
            for j in range(0, output_resolution):
                for k in range(0, output_resolution):
                    mapImgs[i][j][k] = mapInput[output_resolution*(i*output_resolution+j)+k]
        return mapImgs

    def tns4d_batch(self, tensorlist, output_resolution):
        """ return data of a batch of 4d tensors
        """
        dimensions = 6
        formats = 5
        filenames = []

        ## record the file dir and corresponding labels
        filelist = open(tensorlist)
        for line in filelist:
            list_seg = line.split()
            filenames.append(list_seg[-1])
        filelist.close()

        ## get the batch data of 4d tensors and their basic info
        batch_size = len(filenames)
        tensor_batch = np.zeros((batch_size, dimensions, output_resolution, output_resolution), dtype='int32')
        for findex in range(0, batch_size):
            mapImgs = self.tns4d_Sample(filenames[findex], RES)
            for i in range(0, 6):
                tensor_batch[findex, i, :, :] = mapImgs[i]
            print('**** the {}th tns mapping sampled finished ({} in total) ****'.format(findex, batch_size))    
        return tensor_batch

if __name__ == '__main__':
    """[summary]

    [description]
    """
    if len(sys.argv) < 4:
        print("Usage: {} <tensor.list> <resolution> <out.npz>".format(sys.argv[0]))
        exit(1)

    TENSORLIST = sys.argv[1]
    RES = int(sys.argv[2]) # 128
    OUTLIST = sys.argv[3]

    if os.path.isfile(TENSORLIST):
        featureSampler = featureSample()
        mapSampler = mapSample()
        flattenSampler = flattenSample()
        
        features = featureSampler.tns_batch(TENSORLIST)
        map_imgs = mapSampler.tns4d_batch(TENSORLIST, RES)
        flatten_imgs = flattenSampler. tns4d_batch(TENSORLIST, RES)    
    
        np.savez(OUTLIST, map_imgs=map_imgs, flatten_imgs=flatten_imgs, features=features)
