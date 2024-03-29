import os
import sys
import random
import numpy as np

## generate 5-fold CV training/testing datasets
def main():
    data = np.load('../datasets/dataset-cpu-3d.npz')
    flatten_imgs = data['flatten_imgs']
    features = data['features']
    map_imgs = data['map_imgs']
    labels = data['labels']

    batch_size = flatten_imgs.shape[0]
    numft = features.shape[1]
    numformats = labels.shape[2]
    dimensions = map_imgs.shape[1]
    resolution = map_imgs.shape[2]

    tnslist = []
    for i in range(0, 5):
        listfile = np.load('rand_{}.npz'.format(i))
        tnslist.append(listfile['tnslist'])

    for cvindex in range(0, 1):
        testlist = tnslist[cvindex]
        tmptrain = []
        for i in range(0, 5):
            if i == cvindex:
                continue
            tmptrain.append(i)
        trainlist = np.concatenate((tnslist[tmptrain[0]], tnslist[tmptrain[1]], tnslist[tmptrain[2]], tnslist[tmptrain[3]]), axis=0)

        for mode in range(0, 1):
            train_imgs = np.zeros((len(trainlist), 2, resolution, resolution), dtype='float32')
            test_imgs = np.zeros((len(testlist), 2, resolution, resolution), dtype='float32')
            train_features = np.zeros((len(trainlist), numft), dtype='float32')
            test_features = np.zeros((len(testlist), numft), dtype='float32')
            train_labels = np.zeros((len(trainlist), numformats), dtype='int32')
            test_labels = np.zeros((len(testlist), numformats), dtype='int32')

            flattenMax = flatten_imgs[:, mode, :, :].max()
            mapMax = map_imgs[:, mode, :, :].max()

            for i in range(0, len(trainlist)):
                train_imgs[i, 0, :, :] = flatten_imgs[trainlist[i], mode, :, :] / flatten_imgs[trainlist[i], mode, :, :].max()
                train_imgs[i, 1, :, :] = map_imgs[trainlist[i], mode, :, :] / map_imgs[trainlist[i], mode, :, :].max()
                for j in range(0, numft):
                    train_features[i][j] = features[trainlist[i]][j] / features[:, j].max()

                tmp_labels = labels[trainlist[i]][mode]
                train_labels[i][np.argmin(tmp_labels)] = 1

            for i in range(0, len(testlist)):
                test_imgs[i, 0, :, :] = flatten_imgs[testlist[i], mode, :, :] / flatten_imgs[testlist[i], mode, :, :].max()
                test_imgs[i, 1, :, :] = map_imgs[testlist[i], mode, :, :] / map_imgs[testlist[i], mode, :, :].max()
                for j in range(0, numft):
                    test_features[i][j] = features[testlist[i]][j] / features[:, j].max()

                tmp_labels = labels[testlist[i]][mode]
                test_labels[i][np.argmin(tmp_labels)] = 1

            np.savez('mode{}_cv{}_train.npz'.format(mode, cvindex), imgs=train_imgs, features=train_features, labels=train_labels)
            np.savez('mode{}_cv{}_test.npz'.format(mode, cvindex), imgs=test_imgs, features=test_features, labels=test_labels)
            print('mode {} cv {} finished'.format(mode, cvindex))

if __name__=='__main__':
    main()
