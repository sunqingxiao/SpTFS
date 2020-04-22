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

    #def __init__(self):
        #self.ffi = FFI()

#    def tns3d_sample(self, sp3dtns, mat, output_resolution):
#        ## matrix has two dimensions (dim0 and dim1)
#        img_dim0 = np.zeros((output_resolution, output_resolution), dtype='int32')
#        img_dim1 = np.zeros((output_resolution, output_resolution), dtype='int32')
#       
#        ## get the histogram sampling matrixes (both dims)
#        scale_dim0 = sp3dtns.dim0 / output_resolution
#        scale_dim1 = sp3dtns.dim1 / output_resolution
#        maxdim = max(sp3dtns.dim0, sp3dtns.dim1)
#
#        dbg(len(mat))
#        dbg(sp3dtns.nnz)
#
#        for findex in range(0, sp3dtns.nnz):
#            #dbg(mat[findex][0])
#            #dbg(mat[findex][1])
#            bindim = int(output_resolution * abs(mat[findex][0]-mat[findex][1]) / maxdim)
#            
#            ## get the histogram of image (dim0)
#            index_dim0 = int(mat[findex][0] / scale_dim0)
#            img_dim0[index_dim0][bindim] += 1
#            
#            ## get the histogram of image (dim1)
#            index_dim1 = int(mat[findex][1] / scale_dim0)
#            img_dim1[index_dim1][bindim] += 1
#        
#        return img_dim0, img_dim1

	
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
            #print('{} {} {}\n'.format(indexdim[0],indexdim[1], indexdim[2]))
        #dbg(img_0)
        tnsfile.close()

        return tns_to_dict(sp3dtns), img_0, img_1, img_2

    
    def tns3d_batch(self, tensorlist, output_resolution):
        """ return data of a batch of 3d tensors
        """
        dimensions = 3
        formats = 6
        filenames = []
        labels = []
        sp3d_batch = []

        ## record the file dir and corresponding labels
        filelist = open(tensorlist)
        for line in filelist:
            list_seg = line.split()
            filenames.append(list_seg[-19])
            tmplabels = []
            for mode in range(0, 3):
                tmplabels.append([float(list_seg[-18 + mode * 6]), float(list_seg[-17 + mode * 6]), float(list_seg[-16 + mode * 6]), \
                        float(list_seg[-15 + mode * 6]), float(list_seg[-14 + mode * 6]), float(list_seg[-13 + mode * 6])])
                #dbg(tmplabels)
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
        
        return sp3d_batch, tensor_batch, labels

if __name__ == '__main__':
    """[summary]

    [description]
    """
    if len(sys.argv) < 3:
        print("Usage: {} <matrix.list> <resolution>".format(sys.argv[0]))
        exit(1)

    TENSORLIST = sys.argv[1] # '../test/Origin.list'
    RES = int(sys.argv[2]) # 128

    if os.path.isfile(TENSORLIST):
        sampler = Dl3dSample()
        metas, imgs, labels = sampler.tns3d_batch(TENSORLIST, RES)
        np.savez('data/full{}.npz'.format(RES), metas=metas, imgs=imgs, labels=labels)

        
            
            
            
            
        
