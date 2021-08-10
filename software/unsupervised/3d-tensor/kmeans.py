import sys
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import time

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import homogeneity_score
from sklearn.metrics import accuracy_score

from metric import *
from Dl3dDataset import DataSet
from Dl3dNet import *

tf.enable_eager_execution()

def main():
    if len(sys.argv) < 4:
    
    train_data = sys.argv[1]
    test_data = sys.argv[2]
    result_data = sys.argv[3]
    num_output = 256

    print(train_data)
    print(test_data)

    trainset = load_data(train_data)
    testset = load_data(test_data)

    print("Training: {}".format(trainset.features.shape))
    print("Test: {}".format(testset.features.shape))

    num_features = testset.features.shape[1]

    n_digits = len(np.unique(trainset.labels))

    autoencoder = ConvAutoencoder(num_output)
    
    ## flatten images
    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
    autoencoder.fit(trainset.images[:,0,:,:,tf.newaxis], trainset.images[:,0,:,:,tf.newaxis],
                    epochs=100,
                    batch_size=100,
                    shuffle=True)

    flatten_encoded_imgs = autoencoder.encoder(testset.images[:,0,:,:,tf.newaxis]).numpy()

    ## map images
    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
    autoencoder.fit(trainset.images[:,1,:,:,tf.newaxis], trainset.images[:,1,:,:,tf.newaxis],
                    epochs=100,
                    batch_size=100,
                    shuffle=True)
    map_encoded_imgs = autoencoder.encoder(testset.images[:,1,:,:,tf.newaxis]).numpy()

    encoded_imgs = np.concatenate((testset.features, flatten_encoded_imgs, map_encoded_imgs, testset.features), axis=1)

    print(encoded_imgs.shape)

    autoencoder.encoder.summary()
    autoencoder.decoder.summary()

    clusters = [5, 8, 16, 32, 64, 128, 256, 512]

    iner_list = []
    homo_list = []
    top1_acc_list = []
    top2_acc_list = []
    prediction1st_list = []
    prediction2nd_list = []

    labels = np.argmax(testset.labels, axis=1)
    print(labels.shape)
    
    for n_clusters in clusters:
        # test max_iters and batch size
        estimator = MiniBatchKMeans(n_clusters=n_clusters, batch_size=200)
        estimator.fit(encoded_imgs)
            
        inertia, homo = calc_metrics(estimator, encoded_imgs, labels)
        iner_list.append(inertia)
        homo_list.append(homo)

        # Determine predicted labels
        cluster_labels_1, cluster_labels_2 = infer_cluster_labels(estimator, labels)
        prediction_1 = infer_data_labels(estimator.labels_, cluster_labels_1)
        prediction_2 = infer_data_labels(estimator.labels_, cluster_labels_2)
        
        top1_acc = accuracy_score(labels, prediction_1)
        top2_acc = calc_top2_accuracy(labels, prediction_1, prediction_2)
        top2_acc_list.append(top2_acc)
        top1_acc_list.append(top1_acc)
        prediction1st_list.append(prediction_1)
        prediction2nd_list.append(prediction_2)

    np.savez('{}'.format(result_data), top1acc=top1_acc_list, top2acc=top2_acc_list, prediction1st=prediction1st_list, prediction2nd=prediction2nd_list, inertia=iner_list, homo=homo_list)

    print(iner_list)
    print(homo_list)
    print(top1_acc_list)
    print(top2_acc_list)
    print(prediction1st_list)
    print(prediction2nd_list)

if __name__ == '__main__':
    main()
