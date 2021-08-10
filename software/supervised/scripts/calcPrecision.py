import os
import sys
import random
import numpy as np

if __name__=='__main__':
    data = np.load('../datasets/dataset-cpu-3d.npz')
    labels = data['labels']
    numformats = labels.shape[2]
    
    print('Total precision:')
    for mode in range(0, 3):
        modeTotal = 0
        modeRight = 0
        
        for cv in range(0, 5):
            testdata = np.load('result/mode{}_cv{}_WrongIds.npz'.format(mode, cv))

            #print('-------- MODE {}  CV  {} ----------'.format(mode, cv))

            testIds = testdata['wrongIds']
            
            num_total = testIds.shape[0]
            num_right = 0

            for i in range(0, num_total):
                y = testIds[i][0]
                conv_y = testIds[i][1]
                if y == conv_y:
                    num_right += 1 
            
            modeTotal += num_total
            modeRight += num_right
    
        modeTotalRatio = modeRight / modeTotal

        print('Mode {} precision: {}'.format(mode+1, modeTotalRatio))
