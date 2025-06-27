# example of loading the mnist dataset
from numpy import mean
from numpy import std
from sklearn import datasets
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.datasets import cifar10,cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import keras
import keras.layers as layers
import numpy as np
def bn():
    return keras.layers.BatchNormalization()

#!pip install tensorflow_addons
import tensorflow_addons as tfa
#import tensorflow_datasets as tfds
import numpy as np

class L1BatchNorm(layers.Layer):
    def __init__(self, block_size=6, epsilon=1e-5, **kwargs):
        super(L1BatchNorm, self).__init__(**kwargs)
        if not isinstance(block_size, int) or block_size <= 0:
            raise ValueError("block_size must be a positive integer.")
        self.block_size = block_size
        self.epsilon = epsilon

    def build(self, input_shape):
        # input_shape will be (batch_size, height, width, channels)
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('Channel dimension must be known for LocalInstanceNormalizationL1.')
        self.gamma = self.add_weight(
            name='gamma',
            shape=(channels,), # One gamma value per channel
            initializer='ones', # Typically initialized to ones
            trainable=True
        )
        self.beta = self.add_weight(
            name='beta',
            shape=(channels,), # One beta value per channel
            initializer='zeros', # Typically initialized to zeros
            trainable=True
        )

    def call(self, inputs):
        # Ensure inputs are float32 for consistent calculations
        inputs = tf.cast(inputs, tf.float32)
        batch_mean, batch_sigma = tf.nn.moments(inputs, axes=[0], keepdims=True)
        batch_sigma = tf.reduce_mean(tf.abs(inputs-batch_mean),[0],keepdims=True)
        x_batch = (inputs - batch_mean) / (batch_sigma + self.epsilon)
        return x_batch * self.gamma+ self.beta;
        
        
class LocalBatchInstanceNormalization(layers.Layer):
    def __init__(self, block_size=6, epsilon=1e-5, **kwargs):
        super(LocalBatchInstanceNormalization, self).__init__(**kwargs)
        if not isinstance(block_size, int) or block_size <= 0:
            raise ValueError("block_size must be a positive integer.")
        self.block_size = block_size
        self.epsilon = epsilon
        self.bn = bn()
        self.glo = layers.GlobalAveragePooling2D()

    def build(self, input_shape):
        # input_shape will be (batch_size, height, width, channels)
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('Channel dimension must be known for LocalInstanceNormalizationL1.')
        self.gamma = self.add_weight(
            name='gamma',
            shape=(channels,), # One gamma value per channel
            initializer='ones', # Typically initialized to ones
            trainable=True
        )
        self.beta = self.add_weight(
            name='beta',
            shape=(channels,), # One beta value per channel
            initializer='zeros', # Typically initialized to zeros
            trainable=True
        )
        self.lbinweight = self.add_weight(
            name='nweight',
            shape=(channels,), # One beta value per channel
            initializer='zeros', # Typically initialized to zeros
            trainable=True,
            constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0)
        )
        
        super(LocalBatchInstanceNormalization, self).build(input_shape)

    def call(self, inputs):
        # Ensure inputs are float32 for consistent calculations
        inputs = tf.cast(inputs, tf.float32)

        # Reshape gamma and beta to (1, 1, 1, channels) for broadcasting across
        # batch, height, and width dimensions during the final scaling and shifting.
        gamma_reshaped = tf.reshape(self.gamma, [1, 1, 1, -1])
        beta_reshaped = tf.reshape(self.beta, [1, 1, 1, -1])
        
        
        batch_mean, batch_sigma = tf.nn.moments(inputs, axes=[0, 1, 2], keepdims=True)
        batch_sigma = tf.reduce_mean(tf.abs(inputs-batch_mean),[0,1,2],keepdims=True)
        
        x_batch = (inputs - batch_mean) / (batch_sigma + self.epsilon)

        # Compute local mean of each feature within windows of size block_sizeXblock_size around each pixel
        local_mean = tf.nn.avg_pool(
            inputs,
            ksize=[1, self.block_size, self.block_size, 1],
            strides=[1, 1, 1, 1],
            padding='SAME',
            data_format='NHWC'
        )

        # Then, compute the mean of these absolute differences within the same local windows.
        abs_diff = tf.abs(inputs - local_mean)

        local_mad = tf.nn.avg_pool(
            abs_diff,
            ksize=[1, self.block_size, self.block_size, 1],
            strides=[1, 1, 1, 1],
            padding='SAME',
            data_format='NHWC'
        )

        
        normalized_output = (inputs - local_mean) / (local_mad + self.epsilon)
        output = self.lbinweight*normalized_output + (1-self.lbinweight)*x_batch
        
        output = output * gamma_reshaped + beta_reshaped; 
        return output