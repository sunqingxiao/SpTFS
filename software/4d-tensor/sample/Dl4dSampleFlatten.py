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

        ## each mode of 4d-tensor needs one mat (combination of dims)
        img = np.zeros((4, output_resolution, output_resolution), dtype='int32')

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

        ## get the dims of three modes
        dim_mode = []
        dim_mode.append([sp4dtns.dim0, sp4dtns.dim1 * sp4dtns.dim2 * sp4dtns.dim3])
        dim_mode.append([sp4dtns.dim1, sp4dtns.dim0 * sp4dtns.dim2 * sp4dtns.dim3])
        dim_mode.append([sp4dtns.dim2, sp4dtns.dim0 * sp4dtns.dim1 * sp4dtns.dim3])
        dim_mode.append([sp4dtns.dim3, sp4dtns.dim0 * sp4dtns.dim1 * sp4dtns.dim2])

        ## get the histogram sampling of three mats (three figures)
        scaledim = []
        for i in range(0, 4):
            tmpscale = []
            for j in range(0, 2):
                tmpscale.append(dim_mode[i][j] / output_resolution)
            scaledim.append(tmpscale)

        tnsfile = open(tns_dir)
        for line in tnsfile:
            tns_seg = line.split()
            if len(tns_seg) < 5:
            	continue
            valdim = [int(int(tns_seg[-5])-1), int(int(tns_seg[-4])-1), int(int(tns_seg[-3])-1), int(int(tns_seg[-2])-1)]
            mode_valdim = []
            mode_valdim.append([valdim[0], valdim[3]*sp4dtns.dim2*sp4dtns.dim1 + valdim[2]*sp4dtns.dim1 + valdim[1]])
            mode_valdim.append([valdim[1], valdim[3]*sp4dtns.dim2*sp4dtns.dim0 + valdim[2]*sp4dtns.dim0 + valdim[0]])
            mode_valdim.append([valdim[2], valdim[3]*sp4dtns.dim1*sp4dtns.dim0 + valdim[1]*sp4dtns.dim0 + valdim[0]])
            mode_valdim.append([valdim[3], valdim[2]*sp4dtns.dim1*sp4dtns.dim0 + valdim[1]*sp4dtns.dim0 + valdim[0]])
            
            indexdim = []
            for i in range(0, 4):
                tmpindex = []
                for j in range(0, 2):
                    tmpindex.append(int(mode_valdim[i][j] / scaledim[i][j]))
                indexdim.append(tmpindex)

            ## get the histogram of three imgs
            for i in range(0, 4):
                #print('{} {}  {}'.format(i, indexdim[i][0], indexdim[i][1]))
                img[i][indexdim[i][0]][indexdim[i][1]] += 1
        tnsfile.close()
        
        return tns_to_dict(sp4dtns), img

    
    def tns4d_batch(self, tensorlist, output_resolution):
        """ return data of a batch of 4d tensors
        """
        dimensions = 4
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
                #dbg(tmplabels)
            dbg(tmplabels)
            labels.append(tmplabels)

        filelist.close()
        #dbg(labels)

        ## get the batch data of 4d tensors and their basic info
        batch_size = len(filenames)
        tensor_batch = np.zeros((batch_size, dimensions, output_resolution, output_resolution), dtype='int32')
        for findex in range(0, batch_size):
            sp4dtns, img = self.tns4d_Sample(filenames[findex], RES)
            sp4dtns['id'] = findex
            sp4d_batch.append(sp4dtns)
            tensor_batch[findex, :, :, :] = img

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
        np.savez('data/flatten-data.npz'.format(RES), imgs=imgs, labels=labels)
