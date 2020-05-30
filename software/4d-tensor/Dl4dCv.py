import os
import sys
import random
import numpy as np

if __name__=='__main__':
    flatten_data = np.load('data/flatten-data.npz')
    map_data = np.load('data/map-data.npz')
    ftdata = np.load('data/features.npz')
    labeldata = np.load('data/labels.npz')

    flatten_imgs = flatten_data['imgs']
    features = ftdata['features']
    map_imgs = map_data['imgs']
    labels = labeldata['labels']

    batch_size = flatten_imgs.shape[0]
    numft = features.shape[1]
    dimensions = map_imgs.shape[1]
    resolution = map_imgs.shape[2]

    tnslist = []
    for i in range(0, 5):
        listfile = np.load('data/rand_{}.npz'.format(i))
        tnslist.append(listfile['tnslist'])
    print(tnslist)

    for cvindex in range(0, 5):
        testlist = tnslist[cvindex]
        tmptrain = []
        for i in range(0, 5):
            if i == cvindex:
                continue
            tmptrain.append(i)
        trainlist = np.concatenate((tnslist[tmptrain[0]], tnslist[tmptrain[1]], tnslist[tmptrain[2]], tnslist[tmptrain[3]]), axis=0)

        print(testlist.shape)
        print(testlist)

        print(trainlist.shape)
        print(trainlist)


        for mode in range(0, 4):
            maplist = [-1 , -1, -1]
            if mode == 0:
                maplist = [3, 4, 5]
            elif mode == 1:
                maplist = [1, 2, 5]
            elif mode == 2:
                maplist = [0, 2, 4]
            else:
                maplist = [0, 1, 3]
            train_imgs = np.zeros((len(trainlist), 4, resolution, resolution), dtype='float32')
            test_imgs = np.zeros((len(testlist), 4, resolution, resolution), dtype='float32')
            train_features = np.zeros((len(trainlist), numft), dtype='float32')
            test_features = np.zeros((len(testlist), numft), dtype='float32')
            train_labels = np.zeros((len(trainlist), 5), dtype='int32')
            test_labels = np.zeros((len(testlist), 5), dtype='int32')

            flattenMax = flatten_imgs[:, mode, :, :].max()
            mapMax = map_imgs[:, mode, :, :].max()

            for i in range(0, len(trainlist)):
                train_imgs[i, 0, :, :] = flatten_imgs[trainlist[i], mode, :, :] / flatten_imgs[trainlist[i], mode, :, :].max()
                for j in range(0, 3):
                    train_imgs[i, 1+j, :, :] = map_imgs[trainlist[i], maplist[j], :, :] / map_imgs[trainlist[i], maplist[j], :, :].max()

                for j in range(0, numft):
                    train_features[i][j] = features[trainlist[i]][j] / features[:, j].max()

                tmp_labels = labels[trainlist[i]][mode]
                train_labels[i][np.argmin(tmp_labels)] = 1

            print(train_labels)

            for i in range(0, len(testlist)):
                test_imgs[i, 0, :, :] = flatten_imgs[testlist[i], mode, :, :] / flatten_imgs[testlist[i], mode, :, :].max()
                for j in range(0, 3):
                    test_imgs[i, 1+j, :, :] = map_imgs[testlist[i], maplist[j], :, :] / map_imgs[testlist[i], maplist[j], :, :].max()
                for j in range(0, numft):
                    test_features[i][ftcounter] = features[testlist[i]][j] / features[:, j].max()

                tmp_labels = labels[testlist[i]][mode]
                test_labels[i][np.argmin(tmp_labels)] = 1

            print(test_labels)

            np.savez('data/mode{}_cv{}_train.npz'.format(mode, cvindex), imgs=train_imgs, features=train_features, labels=train_labels)
            np.savez('data/mode{}_cv{}_test.npz'.format(mode, cvindex), imgs=test_imgs, features=test_features, labels=test_labels)
