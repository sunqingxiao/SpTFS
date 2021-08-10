import sys
import sklearn
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import homogeneity_score
from sklearn.metrics import accuracy_score

def infer_cluster_labels(kmeans, actual_labels):
    """
    Associates most probable label with each cluster in KMeans model
    returns: dictionary of clusters assigned to each label
    """

    inferred_labels_1 = {}
    inferred_labels_2 = {}

    # Loop through the clusters
    for i in range(kmeans.n_clusters):
        # find index of points in cluster
        labels = []
        index = np.where(kmeans.labels_ == i)

        # append actual labels for each point in cluster
        labels.append(actual_labels[index])

        # determine most common label
        if len(labels[0]) == 0:
            continue
        elif len(labels[0]) == 1:
            counts = np.bincount(labels[0])
        else:
            counts = np.bincount(np.squeeze(labels))
        topN_index = np.argsort(counts)

        # assign the cluster to a value in the inferred_labels dictionary
        if counts.shape[0] >= 2:
            top2_index = [topN_index[counts.shape[0]-1], topN_index[counts.shape[0]-2]]
        else:
            top2_index = [topN_index[counts.shape[0]-1], topN_index[counts.shape[0]-1]]
        
        # assign the 1st label to the cluster
        if top2_index[0] in inferred_labels_1:
            # append the new number to the existing array at this slot
            inferred_labels_1[top2_index[0]].append(i)
        else:
            # create a new array in this slot
            inferred_labels_1[top2_index[0]] = [i]

        # assign the 2nd label to the cluster
        if top2_index[1] in inferred_labels_2:
            # append the new number to the existing array at this slot
            inferred_labels_2[top2_index[1]].append(i)
        else:
            # create a new array in this slot
            inferred_labels_2[top2_index[1]] = [i]

    return inferred_labels_1, inferred_labels_2


def infer_data_labels(X_labels, cluster_labels):
    """
    Determines label for each array, depending on the cluster it has been assigned to.
        returns: predicted labels for each array
        """       
    # empty array of len(X)
    predicted_labels = np.zeros(len(X_labels)).astype(np.uint8)
    
    for i, cluster in enumerate(X_labels):
        for key, value in cluster_labels.items():
            if cluster in value:
                predicted_labels[i] = key
                                                    
    return predicted_labels


def calc_metrics(estimator, data, labels):
    # Inertia
    inertia = estimator.inertia_
    # Homogeneity Score
    homogeneity = homogeneity_score(labels, estimator.labels_)
    return inertia, homogeneity


def calc_top2_accuracy(labels, prediction_1, prediction_2):
    num_right = 0
    for i in range(labels.shape[0]):
        if labels[i] == prediction_1[i] or labels[i] == prediction_2[i]:
            num_right += 1
    acc = num_right / labels.shape[0]
    return acc
