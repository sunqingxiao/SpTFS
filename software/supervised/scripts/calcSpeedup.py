import os
import sys
import random
import numpy as np

def list2txt(fileName="", myfile=[]):
    fileout = open(fileName, 'w')
    for i in range(len(myfile)):
        for j in range(len(myfile[i])):
            fileout.write(str(myfile[i][j]) + ' , ')
        fileout.write('\r\n')
    fileout.close()

if __name__=='__main__':
    data = np.load('../datasets/dataset-cpu-3d.npz')
    labels = data['labels']
    numformats = labels.shape[2]

    total_speedup = np.zeros((3), dtype='float32')
    ave_speedup = np.zeros((3), dtype='float32')
    counter = np.zeros((3), dtype='int32')
    csftotal_speedup = np.zeros((3), dtype='float32')
    csfave_speedup = np.zeros((3), dtype='float32')

    for mode in range(0, 3):
        for cv in range(0, 5):
            listfile = np.load('data/rand_{}.npz'.format(cv))
            randlist = listfile['tnslist']
            testdata = np.load('result/mode{}_cv{}_WrongIds.npz'.format(mode, cv))
            testIds = testdata['wrongIds']

            num_total = randlist.shape[0]
            counter[mode] += num_total
            for j in range(0, num_total):
                conv_y = testIds[j][1]
                tmp_speedup = labels[randlist[j]][mode][0] / labels[randlist[j]][mode][conv_y]
                csftmp_speedup = labels[randlist[j]][mode][3] / labels[randlist[j]][mode][conv_y]
                total_speedup[mode] += tmp_speedup
                csftotal_speedup[mode] += csftmp_speedup

    ave_speedup = total_speedup / counter
    csfave_speedup = csftotal_speedup / counter
    
    print('Speedup over COO:')
    for i in range(0, 3):
        print('Mode {} speedup: {}'.format(i+1, ave_speedup[i]))

    print('Speedup over CSF-based:')
    for i in range(0, 3):
        print('Mode {} speedup: {}'.format(i+1, csfave_speedup[i]))
