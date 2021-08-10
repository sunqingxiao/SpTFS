import os
import sys
import tensorflow as tf
import numpy as np
from Dl3dDataset import DataSet
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

## read numpy data
def load_data(filename):
    try:
        data = np.load(filename)
        ds = DataSet(data['imgs'], data['features'], data['labels'])
    except:
        print("Can not find data file")
        ds = None
    finally:
        return ds

class FcAutoencoder(Model):
    def __init__(self, hidden_dim):
        super(FlAutoencoder, self).__init__()
        self.hidden_dim = hidden_dim

        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(self.hidden_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(128*128, activation='sigmoid'),
            layers.Reshape((128, 128))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class ConvAutoencoder(Model):
    def __init__(self, num_output):
        super(ConvAutoencoder, self).__init__()
        self.num_output = num_output
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(128, 128, 1)),
            layers.Conv2D(64, (3,3), activation='relu', padding='same', strides=2),
            layers.Conv2D(16, (3,3), activation='relu', padding='same', strides=2),
            layers.Conv2D(4, (3,3), activation='relu', padding='same', strides=2),
            layers.Flatten(),
            layers.Dense(num_output, activation='relu')
            ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(1024, activation='sigmoid'),
            layers.Reshape((16, 16, 4)),
            layers.Conv2DTranspose(4, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(64, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2D(1, kernel_size=(3,3), activation='sigmoid', padding='same')]) # decoder激活函数用的sigmoid

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
