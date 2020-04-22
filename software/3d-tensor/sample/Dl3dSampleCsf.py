#!/usr/env python

"""
This is a wrapper function for the sampling
"""

import sys, os
import numpy as np

class featureSet:
    def __init(self):
        self.order = 3
        self.dim = []
        self.nnz = -1
        self.density = -1

        self.numSlices = -1
        self.numFibers = -1
        self.sliceRatio = -1
        self.fiberRatio = -1

        self.maxNnzPerSlice = -1
        self.minNnzPerSlice = -1
        self.aveNnzPerSlice = -1
        self.devNnzPerSlice = -1
        self.adjNnzPerSlice = -1

        self.maxFibersPerSlice = -1
        self.minFibersPerSlice = -1
        self.aveFibersPerSlice = -1
        self.devFibersPerSlice = -1
        self.adjFibersPerSlice = -1

        self.maxNnzPerFiber = -1
        self.minNnzPerFiber = -1
        self.aveNnzPerFiber = -1
        self.devNnzPerFiber = -1
        self.adjNnzPerFiber = -1

def dbg(data):
    """[summary]

    [description]

    Arguments:
        data {[type]} -- [description]
    """

    print(data)

def compare(x, y):
    """ compare two values.
    """
    if x > y:
        return 1
    else:
        return 0


class gbt3dSample(object):
    """[summary]

    [description]
    """

    def tns_sample(self, tns_dir):
        """obtain the feature set of a tensor
        """
        ftset = featureSet()

        ## get nnz, dims, density among the feature set
        tnsfile = open(tns_dir)
        counter = -2
        for line in tnsfile:
            if counter == -1:
                tns_seg = line.split()
                ftset.dim= [int(tns_seg[-3]), int(tns_seg[-2]), int(tns_seg[-1])]
            counter += 1
        tnsfile.close()
        ftset.nnz = counter
        ftset.density = ftset.nnz/(ftset.dim[0]*ftset.dim[1]*ftset.dim[2])
        
        tnsfile = open(tns_dir)

        #TODO:determine the initial value
        fibersPerSlice = []
        nnzPerSlice = []
        nnzPerFiber = []
        tmpfibersPerSlice = 1
        tmpnnzPerSlice = 1
        tmpnnzPerFiber = 1

        for line in tnsfile:
            tns_seg = line.split()
            if len(tns_seg) < 4:
                continue
            else:
                lastdim = [int(int(tns_seg[-4])-1), int(int(tns_seg[-3])-1)]
                break

        for line in tnsfile:
            tns_seg = line.split()
            if len(tns_seg) < 4:
                continue
            valdim = [int(int(tns_seg[-4])-1), int(int(tns_seg[-3])-1)]
            if valdim[0] == lastdim[0]:
                tmpnnzPerSlice += 1
                if valdim[1] == lastdim[1]:
                    tmpnnzPerFiber += 1
                else:
                    nnzPerFiber.append(tmpnnzPerFiber)
                    tmpnnzPerFiber = 1
                    tmpfibersPerSlice += 1
            else:
                nnzPerFiber.append(tmpnnzPerFiber)
                fibersPerSlice.append(tmpfibersPerSlice)
                nnzPerSlice.append(tmpnnzPerSlice)
                tmpnnzPerFiber = 1
                tmpnnzPerSlice = 1
                tmpfibersPerSlice = 1
            lastdim = valdim

        fibersPerSlice.append(tmpfibersPerSlice)
        nnzPerSlice.append(tmpnnzPerSlice)
        nnzPerFiber.append(tmpnnzPerFiber)
        
        tnsfile.close()
        
        # get detaied features
        ftset.numSlices = len(nnzPerSlice)
        ftset.numFibers = len(nnzPerFiber)
        ftset.sliceRatio = ftset.numSlices / ftset.dim[0]
        ftset.fiberRatio = ftset.numFibers / (ftset.dim[0] * ftset.dim[1])
        ftset.maxNnzPerSlice = max(nnzPerSlice)
        ftset.minNnzPerSlice = min(nnzPerSlice)
        ftset.devNnzPerSlice = ftset.maxNnzPerSlice - ftset.minNnzPerSlice
        ftset.maxFibersPerSlice = max(fibersPerSlice)
        ftset.minFibersPerSlice = min(fibersPerSlice)
        ftset.devFibersPerSlice = ftset.maxFibersPerSlice - ftset.minFibersPerSlice
        ftset.maxNnzPerFiber = max(nnzPerFiber)
        ftset.minNnzPerFiber = min(nnzPerFiber)
        ftset.devNnzPerFiber = ftset.maxNnzPerFiber - ftset.minNnzPerFiber
        ftset.aveNnzPerSlice = ftset.nnz / ftset.numSlices
        ftset.aveNnzPerFiber = ftset.nnz / ftset.numFibers
        ftset.aveFibersPerSlice = ftset.numFibers / ftset.numSlices

        # get adjNnz
        totaladjFiberPerSlice = 0.0
        totaladjNnzPerSlice = 0.0
        totaladjNnzPerFiber = 0.0
        for i in range(1, ftset.numSlices):
            totaladjFiberPerSlice += abs(fibersPerSlice[i] - fibersPerSlice[i-1])
            totaladjNnzPerSlice += abs(nnzPerSlice[i] - nnzPerSlice[i-1])
        for i in range(1, ftset.numFibers):
            totaladjNnzPerFiber += abs(nnzPerFiber[i] - nnzPerFiber[i-1])

        if ftset.numSlices == 1:
            ftset.adjFibersPerSlice = 0
            ftset.adjNnzPerSlice = 0
        else:
            ftset.adjFibersPerSlice = totaladjFiberPerSlice / (ftset.numSlices - 1)
            ftset.adjNnzPerSlice = totaladjNnzPerSlice / (ftset.numSlices - 1)

        if ftset.numFibers == 1:
            ftset.adjNnzPerFiber = 0
        else:
            ftset.adjNnzPerFiber = totaladjNnzPerFiber / (ftset.numFibers - 1)

        return ftset, fibersPerSlice, nnzPerSlice, nnzPerFiber


    def tns_batch(self, tensorlist, ftlist):
        """ obtain the feature set of a batch of tensors
        """
        filenames = []
        labels = []
        ftset_batch = []
        fibersPerSlice_batch = []
        nnzPerSlice_batch = []
        nnzPerFiber_batch = []

        ## record the file dir and corresponding labels
        filelist = open(tensorlist)
        for line in filelist:
            list_seg = line.split()
            filenames.append(list_seg[-19])
        filelist.close()

        ## record the feature sets of tensors
        batch_size = len(filenames)
        for findex in range(0, batch_size):
            ftset, fibersPerSlice, nnzPerSlice, nnzPerFiber = self.tns_sample(filenames[findex])
            ftset_batch.append(ftset)
            print('**** {}: the {}th tns sampled finished ({} in total) ****'.format(ftlist, findex, batch_size))

        ## print the feature sets and labels to ftlist
        ftdata = open(ftlist, "w+")
        for findex in range(0, batch_size):
            for i in range(0, 3):
                ftdata.write('{},'.format(ftset_batch[findex].dim[i]))
            ftdata.write('{},{},{},{},{},{},'.format(ftset_batch[findex].nnz,ftset_batch[findex].density,ftset_batch[findex].numSlices,ftset_batch[findex].numFibers,ftset_batch[findex].sliceRatio,ftset_batch[findex].fiberRatio))
            ftdata.write('{},{},{},{},{},'.format(ftset_batch[findex].maxFibersPerSlice,ftset_batch[findex].minFibersPerSlice,ftset_batch[findex].aveFibersPerSlice,ftset_batch[findex].devFibersPerSlice,ftset_batch[findex].adjFibersPerSlice))
            ftdata.write('{},{},{},{},{},'.format(ftset_batch[findex].maxNnzPerSlice,ftset_batch[findex].minNnzPerSlice,ftset_batch[findex].aveNnzPerSlice,ftset_batch[findex].devNnzPerSlice,ftset_batch[findex].adjNnzPerSlice))
            ftdata.write('{},{},{},{},{}'.format(ftset_batch[findex].maxNnzPerFiber,ftset_batch[findex].minNnzPerFiber,ftset_batch[findex].aveNnzPerFiber,ftset_batch[findex].devNnzPerFiber,ftset_batch[findex].adjNnzPerFiber))
            if findex < batch_size -1:
                ftdata.write('\n')
        ftdata.close()

if __name__ == '__main__':
    """[summary]

    [description]
    """
    if len(sys.argv) < 3:
        print("Usage: {} <tensor.list> <output.list>".format(sys.argv[0]))
        exit(1)

    TENSORLIST = sys.argv[1]
    FSLIST = sys.argv[2]
    
    sampler = gbt3dSample()
    sampler.tns_batch(TENSORLIST, FSLIST)
