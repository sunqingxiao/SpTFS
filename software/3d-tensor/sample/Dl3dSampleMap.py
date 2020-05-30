#!/usr/env python

"""
This is a wrapper function for the sampling
"""

import sys, os
import numpy as np
#from cffi import FFI

class Sp3dtns:
	def __init(self):
		self.dim0 = -1
		self.dim1 = -1
		self.dim2 = -1
		self.nnz = -1

def dbg(data):
    """[summary]

    [description]

    Arguments:
        data {[type]} -- [description]
    """

    print(data)


def tns_to_dict(sp3dtns):
    """[summary] convert sparse 3d data to python dict

    [description]

    Arguments:
        sp3dtns {[type]} -- [description]
    """

    return {'id': None, 'dim0': sp3dtns.dim0, 'dim1': sp3dtns.dim1, 'dim2': sp3dtns.dim2, 'nnz': sp3dtns.nnz}


class Dl3dSample(object):
    """[summary]

    [description]
        wrapper of 3d tensor sampling
    """

    def tns3d_Sample(self, tns_dir, output_resolution):
        ## the basic information of sparse 3d tensor
        sp3dtns = Sp3dtns()

        ## 3d-tensor needs three mats (combination of dims)
        img_0 = np.zeros((output_resolution, output_resolution), dtype='int32')
        img_1 = np.zeros((output_resolution, output_resolution), dtype='int32')
        img_2 = np.zeros((output_resolution, output_resolution), dtype='int32')

		## get the dims and nnz of sp3dtns
        counter = -2
        tnsfile = open(tns_dir)
        for line in tnsfile:
            if counter == -1:
                tns_seg = line.split()
                sp3dtns.dim0 = int(tns_seg[-3])
                sp3dtns.dim1 = int(tns_seg[-2])
                sp3dtns.dim2 = int(tns_seg[-1])
            counter += 1
        sp3dtns.nnz = counter
        tnsfile.close()

        ## get the histogram sampling of three mats (three figures)
        scaledim = [sp3dtns.dim0/output_resolution, sp3dtns.dim1/output_resolution, sp3dtns.dim2/output_resolution]

        tnsfile = open(tns_dir)
        for line in tnsfile:
            tns_seg = line.split()
            if len(tns_seg) < 4:
            	continue
            valdim = [int(int(tns_seg[-4])-1), int(int(tns_seg[-3])-1), int(int(tns_seg[-2])-1)]
            indexdim = [int(valdim[0]/scaledim[0]), int(valdim[1]/scaledim[1]), int(valdim[2]/scaledim[2])]

            ## get the histogram of three imgs
            img_0[indexdim[1]][indexdim[2]] += 1
            img_1[indexdim[0]][indexdim[2]] += 1
            img_2[indexdim[0]][indexdim[1]] += 1
        tnsfile.close()

        return tns_to_dict(sp3dtns), img_0, img_1, img_2

    
    def tns3d_batch(self, tensorlist, output_resolution):
        """ return data of a batch of 3d tensors
        """
        dimensions = 3
        formats = 5
        filenames = []
        labels = []
        sp3d_batch = []

        ## record the file dir and corresponding labels
        filelist = open(tensorlist)
        for line in filelist:
            list_seg = line.split()
            filenames.append(list_seg[-16])
            tmplabels = []
            for mode in range(0, 3):
                tmplabels.append([float(list_seg[-15 + mode * 5]), float(list_seg[-14 + mode * 5]), float(list_seg[-13 + mode * 5]), \
                        float(list_seg[-12 + mode * 5]), float(list_seg[-11 + mode * 5])])
            dbg(tmplabels)
            labels.append(tmplabels)
        filelist.close()
        #dbg(labels)

        ## get the batch data of 3d tensors and their basic info
        batch_size = len(filenames)
        tensor_batch = np.zeros((batch_size, dimensions, output_resolution, output_resolution), dtype='int32')
        for findex in range(0, batch_size):
            sp3dtns, img_0, img_1, img_2 = self.tns3d_Sample(filenames[findex], RES)
            sp3dtns['id'] = findex
            sp3d_batch.append(sp3dtns)
            tensor_batch[findex, 0, :, :] = img_0
            tensor_batch[findex, 1, :, :] = img_1
            tensor_batch[findex, 2, :, :] = img_2

            dbg(sp3dtns)
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
        sampler = Dl3dSample()
        imgs, labels = sampler.tns3d_batch(TENSORLIST, RES)
        np.savez('data/map-data.npz'.format(RES), imgs=imgs, labels=labels)

        
            
            
            
            
        
