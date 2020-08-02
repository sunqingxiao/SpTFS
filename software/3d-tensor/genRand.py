import os
import sys
import random
import numpy as np

if __name__=='__main__':
    data = np.load('../datasets/dataset-cpu-3d.npz')
    labels = data['labels']
    batch_size = labels.shape[0]

    tnslist = list(range(0, batch_size))

    # 5-fold cross validation
    for cvindex in range(0, 4):
        testlist = random.sample(tnslist, int(len(tnslist) / (5-cvindex)))
        for i in range(0, len(testlist)):
            tnslist.remove(testlist[i])
        np.savez('data/rand_{}.npz'.format(cvindex), tnslist=testlist)
        print(len(testlist))

    print(len(tnslist))
    np.savez('data/rand_4.npz'.format(cvindex), tnslist=tnslist)
