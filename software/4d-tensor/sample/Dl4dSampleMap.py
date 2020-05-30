#!/usr/env python

"""
This is a wrapper function for the sampling
"""

import sys, os
import numpy as np
#from cffi import FFI

class Sp4dtns:
    def __init(self):
        self.dim0 = -1
        self.dim1 = -1
        self.dim2 = -1
        self.dim3 = -1
        self.nnz = -1

def dbg(data):
    """[summary]

    [description]

    Arguments:
        data {[type]} -- [description]
    """

    print(data)


def tns_to_dict(sp4dtns):
    """[summary] convert sparse 4d data to python dict

    [description]

    Arguments:
        sp4dtns {[type]} -- [description]
    """

    return {'id': None, 'dim0': sp4dtns.dim0, 'dim1': sp4dtns.dim1, 'dim2': sp4dtns.dim2, 'dim3': sp4dtns.dim3, 'nnz': sp4dtns.nnz}


class Dl4dSample(object):
    """[summary]

    [description]
        wrapper of 4d tensor sampling
    """

    def tns4d_Sample(self, tns_dir, output_resolution):
        ## the basic information of sparse 4d tensor
        sp4dtns = Sp4dtns()

        ## 4d-tensor needs three mats (combination of dims)
        img_01 = np.zeros((output_resolution, output_resolution), dtype='int32')
        img_02 = np.zeros((output_resolution, output_resolution), dtype='int32')
        img_03 = np.zeros((output_resolution, output_resolution), dtype='int32')
        img_12 = np.zeros((output_resolution, output_resolution), dtype='int32')
        img_13 = np.zeros((output_resolution, output_resolution), dtype='int32')
        img_23 = np.zeros((output_resolution, output_resolution), dtype='int32')

		## get the dims and nnz of sp4dtns
        counter = -2
        tnsfile = open(tns_dir)
        for line in tnsfile:
            if counter == -1:
                tns_seg = line.split()
                sp4dtns.dim0 = int(tns_seg[-4])
                sp4dtns.dim1 = int(tns_seg[-3])
                sp4dtns.dim2 = int(tns_seg[-2])
                sp4dtns.dim3 = int(tns_seg[-1])
            counter += 1
        sp4dtns.nnz = counter
        tnsfile.close()

        ## get the histogram sampling of three mats (three figures)
        scaledim = [sp4dtns.dim0/output_resolution, sp4dtns.dim1/output_resolution, sp4dtns.dim2/output_resolution, sp4dtns.dim3/output_resolution]

        tnsfile = open(tns_dir)
        for line in tnsfile:
            tns_seg = line.split()
            if len(tns_seg) < 5:
            	continue
            valdim = [int(int(tns_seg[-5])-1), int(int(tns_seg[-4])-1), int(int(tns_seg[-3])-1), int(int(tns_seg[-2])-1)]
            indexdim = [int(valdim[0]/scaledim[0]), int(valdim[1]/scaledim[1]), int(valdim[2]/scaledim[2]), int(valdim[3]/scaledim[3])]

            ## get the histogram of three imgs
            img_01[indexdim[0]][indexdim[1]] += 1
            img_02[indexdim[0]][indexdim[2]] += 1
            img_03[indexdim[0]][indexdim[3]] += 1
            img_12[indexdim[1]][indexdim[2]] += 1
            img_13[indexdim[1]][indexdim[3]] += 1
            img_23[indexdim[2]][indexdim[3]] += 1
        tnsfile.close()

        return tns_to_dict(sp4dtns), img_01, img_02, img_03, img_12, img_13, img_23

    
    def tns4d_batch(self, tensorlist, output_resolution):
        """ return data of a batch of 4d tensors
        """
        dimensions = 6
        formats = 5
        filenames = []
        labels = []
        sp4d_batch = []

        ## record the file dir and corresponding labels
        filelist = open(tensorlist)
        for line in filelist:
            list_seg = line.split()
            filenames.append(list_seg[-21])
            tmplabels = []
            for mode in range(0, 4):
                tmplabels.append([float(list_seg[-20 + mode * 5]), float(list_seg[-19 + mode * 5]), float(list_seg[-18 + mode * 5]), \
                        float(list_seg[-17 + mode * 5]), float(list_seg[-16 + mode * 5])])
            dbg(tmplabels)
            labels.append(tmplabels)
        filelist.close()
        #dbg(labels)

        ## get the batch data of 4d tensors and their basic info
        batch_size = len(filenames)
        tensor_batch = np.zeros((batch_size, dimensions, output_resolution, output_resolution), dtype='int32')
        for findex in range(0, batch_size):
            sp4dtns, img_01, img_02, img_03, img_12, img_13, img_23 = self.tns4d_Sample(filenames[findex], RES)
            sp4dtns['id'] = findex
            sp4d_batch.append(sp4dtns)
            tensor_batch[findex, 0, :, :] = img_01
            tensor_batch[findex, 1, :, :] = img_02
            tensor_batch[findex, 2, :, :] = img_03
            tensor_batch[findex, 3, :, :] = img_12
            tensor_batch[findex, 4, :, :] = img_13
            tensor_batch[findex, 5, :, :] = img_23

            dbg(sp4dtns)
            #dbg(tensor_batch)
        
        return tensor_batch, labels

if __name__ == '__main__':
    """[summary]

    [description]
    """
    if len(sys.argv) < 3:
        print("Usage: {} <tensorlist> <resolution>".format(sys.argv[0]))
        exit(1)

    TENSORLIST = sys.argv[1] # '../test/Origin.list'
    RES = int(sys.argv[2]) # 128

    if os.path.isfile(TENSORLIST):
        sampler = Dl4dSample()
        imgs, labels = sampler.tns4d_batch(TENSORLIST, RES)
        np.savez('data/map-data.npz'.format(RES), imgs=imgs, labels=labels)
