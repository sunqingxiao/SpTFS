#!/usr/env python

"""
This is a wrapper function for the sampling
"""

import sys, os
import numpy as np

class featureSet:
    def __init(self):
        self.order = 4
        self.dim = []
        self.nnz = -1
        self.density = -1
        self.ave_nnz = []
        self.max_nnz = []
        self.min_nnz = []
        self.dev_nnz = []
        self.bounce = []
        self.mean_neighbor = -1


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


class gbt4dSample(object):
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
                ftset.dim= [int(tns_seg[-4]), int(tns_seg[-3]), int(tns_seg[-2]), int(tns_seg[-1])]
            counter += 1
        tnsfile.close()
        ftset.nnz = counter
        ftset.density = ftset.nnz/(ftset.dim[0]*ftset.dim[1]*ftset.dim[2]*ftset.dim[3])

        ## get nnz of each dim
        dim0_nnz = np.zeros(ftset.dim[0], dtype='int32')
        dim1_nnz = np.zeros(ftset.dim[1], dtype='int32')
        dim2_nnz = np.zeros(ftset.dim[2], dtype='int32')
        dim3_nnz = np.zeros(ftset.dim[3], dtype='int32')

        tnsfile = open(tns_dir)
        for line in tnsfile:
            tns_seg = line.split()
            if len(tns_seg) < 5:
                continue
            valdim = [int(int(tns_seg[-5])-1), int(int(tns_seg[-4])-1), int(int(tns_seg[-3])-1), int(int(tns_seg[-2])-1)]
            dim0_nnz[valdim[0]] += 1
            dim1_nnz[valdim[1]] += 1
            dim2_nnz[valdim[2]] += 1
            dim3_nnz[valdim[3]] += 1
        tnsfile.close()

        ## initialize max_nnz, min_nnz, tot_nnz, etc
        ftset.max_nnz = [dim0_nnz[0], dim1_nnz[0], dim2_nnz[0], dim3_nnz[0]]
        ftset.min_nnz = [dim0_nnz[0], dim1_nnz[0], dim2_nnz[0], dim3_nnz[0]]
        ftset.ave_nnz = [dim0_nnz[0], dim1_nnz[0], dim2_nnz[0], dim3_nnz[0]]
        ftset.dev_nnz = [dim0_nnz[0], dim1_nnz[0], dim2_nnz[0], dim3_nnz[0]]
        ftset.bounce = [dim0_nnz[0], dim1_nnz[0], dim2_nnz[0], dim3_nnz[0]]

        tot_nnz = [dim0_nnz[0], dim1_nnz[0], dim2_nnz[0], dim3_nnz[0]]
        dim_counter = [compare(dim0_nnz[0],0), compare(dim1_nnz[0],0), compare(dim2_nnz[0],0), compare(dim3_nnz[0], 0)]
        adj_nnz = [0, 0, 0, 0]
               
        ## get the values of above variables
        for findex in range(1, ftset.dim[0]):
            if dim0_nnz[findex] > ftset.max_nnz[0]:
                ftset.max_nnz[0] = dim0_nnz[findex]
            if dim0_nnz[findex] < ftset.min_nnz[0]:
                ftset.min_nnz[0] = dim0_nnz[findex]
            adj_nnz[0] += abs(dim0_nnz[findex] - dim0_nnz[findex-1])
            tot_nnz[0] += dim0_nnz[findex]
            dim_counter[0] += compare(dim0_nnz[findex], 0)

        for findex in range(1, ftset.dim[1]):
            if dim1_nnz[findex] > ftset.max_nnz[1]:
                ftset.max_nnz[1] = dim1_nnz[findex]
            if dim1_nnz[findex] < ftset.min_nnz[1]:
                ftset.min_nnz[1] = dim1_nnz[findex]
            adj_nnz[1] += abs(dim1_nnz[findex] - dim1_nnz[findex-1])
            tot_nnz[1] += dim1_nnz[findex]
            dim_counter[1] += compare(dim1_nnz[findex], 0)

        for findex in range(1, ftset.dim[2]):
            if dim2_nnz[findex] > ftset.max_nnz[2]:
                ftset.max_nnz[2] = dim2_nnz[findex]
            if dim2_nnz[findex] < ftset.min_nnz[2]:
                ftset.min_nnz[2] = dim2_nnz[findex]
            adj_nnz[2] += abs(dim2_nnz[findex] - dim2_nnz[findex-1])
            tot_nnz[2] += dim2_nnz[findex]
            dim_counter[2] += compare(dim2_nnz[findex], 0)

        for findex in range(1, ftset.dim[3]):
            if dim3_nnz[findex] > ftset.max_nnz[3]:
                ftset.max_nnz[3] = dim3_nnz[findex]
            if dim3_nnz[findex] < ftset.min_nnz[3]:
                ftset.min_nnz[3] = dim3_nnz[findex]
            adj_nnz[3] += abs(dim3_nnz[findex] - dim3_nnz[findex-1])
            tot_nnz[3] += dim3_nnz[findex]
            dim_counter[3] += compare(dim3_nnz[findex], 0)

        ## get the values of ave_nnz, dev_nnz and bounce
        for i in range(0, 4):
            ftset.dev_nnz[i] = ftset.max_nnz[i] - ftset.min_nnz[i]
            ftset.ave_nnz[i] = tot_nnz[i] / dim_counter[i]
            if ftset.dim[i] < 2:
                ftset.bounce[i] = 0.0
            else:
                ftset.bounce[i] = adj_nnz[i] / (ftset.dim[i] - 1)

        return ftset


    def tns_batch(self, tensorlist, ftlist):
        """ obtain the feature set of a batch of tensors
        """
        filenames = []
        labels = []
        ftset_batch = []

        ## record the file dir and corresponding labels
        filelist = open(tensorlist)
        for line in filelist:
            list_seg = line.split()
            filenames.append(list_seg[-21])
            tmplabel = []
            for mode in range(0, 4):
                codes = [float(list_seg[-20 + mode * 5]), float(list_seg[-19 + mode * 5]), float(list_seg[-18 + mode * 5]), \
                        float(list_seg[-17 + mode * 5]), float(list_seg[-16 + mode * 5])]
                tmplabel.append(codes.index(min(codes)))
            labels.append(tmplabel)
        filelist.close()

        ## record the feature sets of tensors
        batch_size = len(labels)
        for findex in range(0, batch_size):
            ftset = self.tns_sample(filenames[findex])
            ftset_batch.append(ftset)
            print('**** {}: the {}th tns sampled finished ({} in total) ****'.format(ftlist, findex, batch_size))

        ## print the feature sets and labels to ftlist
        ftdata = open(ftlist, "a+")
        for findex in range(0, batch_size):
            for i in range(0, 4):
                ftdata.write(str(ftset_batch[findex].dim[i])+',')
            ftdata.write(str(ftset_batch[findex].nnz)+','+str(ftset_batch[findex].density)+',')
            for i in range(0,4):
                ftdata.write(str(ftset_batch[findex].max_nnz[i])+',')
            for i in range(0,4):
                ftdata.write(str(ftset_batch[findex].min_nnz[i])+',')
            for i in range(0,4):
                ftdata.write(str(ftset_batch[findex].dev_nnz[i])+',')
            for i in range(0,4):
                ftdata.write(str(ftset_batch[findex].ave_nnz[i])+',')
            for i in range(0,4):
                ftdata.write(str(ftset_batch[findex].bounce[i])+',')
            ftdata.write('{},{},{},{}'.format(str(labels[findex][0]), str(labels[findex][1]), str(labels[findex][2]), str(labels[findex][3])))
            if findex < batch_size -1:
                ftdata.write('\n')


if __name__ == '__main__':
    """[summary]

    [description]
    """
    if len(sys.argv) < 3:
        print("Usage: {} <tensor.list> <output.list>".format(sys.argv[0]))
        exit(1)

    TENSORLIST = sys.argv[1]
    FSLIST = sys.argv[2]
    
    sampler = gbt4dSample()
    sampler.tns_batch(TENSORLIST, FSLIST)







            










