import collections
import numpy as np

DataSets = collections.namedtuple('DataSets', ['train', 'test'])

class DataSet(object):
    def __init__(self, images, features, labels):
        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape %s' % (images.shape, labels.shape)
        )
        assert features.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape %s' % (images.shape, labels.shape)
        )
        self._num_examples = images.shape[0]

        self._images = images
        self._features = features
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        # shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._features = self.features[perm0]
            self._labels = self.labels[perm0]
        # next epoch
        if start + batch_size > self._num_examples:
            # finished epoch
            self._epochs_completed += 1
            # rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            features_rest_part = self._features[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._features = self.features[perm]
                self._labels = self.labels[perm]
            # start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            features_new_part = self._features[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate((features_rest_part, features_new_part), axis=0), np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._features[start:end], self._labels[start:end]
