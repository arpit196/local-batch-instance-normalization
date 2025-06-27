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
from tensorflow.keras.layers import Conv2D,Input
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
import numpy as np
from LBIN import *
def bn():
    return keras.layers.BatchNormalization()

from keras.layers import LeakyReLU
from keras.regularizers import L2
def conc(*inp1):
  return layers.Concatenate()(inp1)

#!pip install tensorflow_addons
import tensorflow_addons as tfa

import tensorflow.keras.layers as layers

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers

def VGG16_lbin(input_shape=(32, 32, 3), classes=10):
    """
    VGG16 architecture with Leaky ReLU activation and Dropout.

    Args:
        input_shape: Shape of the input images (height, width, channels).
        classes: Number of output classes.

    Returns:
        A tf.keras.Model instance.
    """
    alpha=0.0005;alpha2=0.1
    input_layer = Input(shape=input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha2),kernel_regularizer=L2(alpha))(input_layer)
    x=LocalBatchInstanceNormalization()(x)
    
    x = Dropout(0.3)(x)
    x = Conv2D(64, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha2),kernel_regularizer=L2(alpha))(x)
    x=LocalBatchInstanceNormalization()(x) 
    
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
     # Added dropout

    # Block 2
    x = Conv2D(128, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha2),kernel_regularizer=L2(alpha))(x)
    x=LocalBatchInstanceNormalization()(x)
    
    x = Dropout(0.4)(x)
    x = Conv2D(128, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha2),kernel_regularizer=L2(alpha))(x)
    x=LocalBatchInstanceNormalization()(x) 
    
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 3
    x = Conv2D(256, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha2),kernel_regularizer=L2(alpha))(x)
    x=LocalBatchInstanceNormalization()(x)
    
    x = Dropout(0.4)(x)
    x = Conv2D(256, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha2),kernel_regularizer=L2(alpha))(x)
    x=LocalBatchInstanceNormalization()(x) 
    
    x = Dropout(0.4)(x)
    x = Conv2D(256, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha2),kernel_regularizer=L2(alpha))(x)
    x=LocalBatchInstanceNormalization()(x)
    
    xlow2 = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 4
    x = Conv2D(512, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha2),kernel_regularizer=L2(alpha))(xlow2)
    x=LocalBatchInstanceNormalization()(x)
    
    x = Dropout(0.4)(x)
    x = Conv2D(512, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha2),kernel_regularizer=L2(alpha))(x)
    x=LocalBatchInstanceNormalization()(x) 
    
    x = Dropout(0.4)(x)
    x = Conv2D(512, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha2),kernel_regularizer=L2(alpha))(x)
    x=LocalBatchInstanceNormalization()(x) 
    
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 5
    x = Conv2D(512, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha2),kernel_regularizer=L2(alpha))(x)
    x=LocalBatchInstanceNormalization()(x)
    
    x = Dropout(0.4)(x)
    x = Conv2D(512, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha2),kernel_regularizer=L2(alpha))(x)
    x=LocalBatchInstanceNormalization()(x) 
    x = Dropout(0.4)(x)
    x = Conv2D(512, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha2),kernel_regularizer=L2(alpha))(x)
    x=LocalBatchInstanceNormalization()(x)
    
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Classification part
    x = Flatten()(x)
    x = Dropout(0.5)(x)  # Added dropout
    x = Dense(512, activation=LeakyReLU(alpha=0.1),kernel_regularizer=L2(alpha))(conc(x,layers.GlobalAveragePooling2D()(xlow2)))
    x = L1BatchNorm()(x)
    output_layer = Dense(classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model
