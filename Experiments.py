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
from vgg_model import *
from LBIN import *
import tensorflow as tf
import keras
import numpy as np
def conc(*inp1):
  return layers.Concatenate()(inp1)

def mpool(psize,strides=2):
  return MaxPooling2D(pool_size=psize,strides=strides,padding="VALID")

def apool(psize,strides=None):
  if(strides is None):
    return AveragePooling2D(pool_size=psize,padding='SAME')
  else:
    return AveragePooling2D(pool_size=psize,strides=strides,padding="SAME")

def ln():
  return layers.LayerNormalization()

def bn():
  return layers.BatchNormalization()

def dense(size,act='relu'):
  return Dense(size,activation=act)

from keras.callbacks import ModelCheckpoint
import tensorflow.keras.layers as layers
filepath = 'my_best_model.hdf5'

learning_rate = 0.0001
lr_drop=20
def lr_scheduler(epoch):
        return learning_rate * (0.5 ** (epoch // lr_drop))

reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

def train(model,path,epochs=100):
  
  checkpoint1 = ModelCheckpoint(filepath='./'+path,save_format=tf,monitor='val_loss',
                            save_weights_only=True,
                             verbose=1, 
                             save_best_only=True,
                             mode='min')
  model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate,momentum=0.9,clipvalue=0.01,global_clipnorm=0.01),
    loss='categorical_crossentropy',
    metrics='categorical_accuracy')
  model.fit(trainXA, trainY, epochs=epochs, batch_size=200, validation_data=(testXA, testY), callbacks=[checkpoint1, reduce_lr], verbose=1)

def bnconv(inp,units,kernel_size):
  return BatchNormalization()(Conv2D(units,kernel_size,activation='relu',kernel_regularizer=tf.keras.regularizers.L2(0.0005))(inp))

def dense(units,act='relu'):
  return layers.Dense(units,activation=act)

def conv(inp,units,kernel_size):
  return Conv2D(kernel_size,units,activation='relu',kernel_regularizer=tf.keras.regularizers.L2(0.0005))(inp)

def cutout(image,size):
  return tfa.image.cutout(image,mask_size=(size,size))

#import tensorflow_datasets as tfds
import numpy as np

class MultiScale(keras.layers.Layer):
  def __init__(self,scale):
    super(MultiScale,self).__init__()
    self.scale = scale
  
  def call(self,inputs):
    return tfa.image.gaussian_filter2d(image=inputs,filter_shape=(self.scale, self.scale),sigma = 2.0,padding = 'REFLECT')

def bnconv(inp,units,kernel_size):
  return BatchNormalization()(Conv2D(units,kernel_size,activation='relu',kernel_regularizer=tf.keras.regularizers.L2(0.0005),padding="SAME")(inp))


def load_dataset():
	# load dataset
  (trainX, trainY), (testX, testY) = mnist.load_data()
  trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
  testX = testX.reshape((testX.shape[0], 28, 28, 1))
  return trainX, trainY, testX, testY
  
def prep_pixels(train, test):
	# convert from integers to floats
  train_norm = train.astype('float32')
  test_norm = test.astype('float32')
	# normalize to range 0-1
  train_norm = train_norm / 255.0
  test_norm = test_norm / 255.0
  normalization = layers.Normalization()
  normalization.adapt(train_norm)
  #train_norm=normalization(train_norm)
  #test_norm=normalization(test_norm)
  return train_norm, test_norm
	
def load_images(cfr100=False):
    if(cfr100):
        (train_images, train_labels), (test_images, test_labels) = cifar100.load_data()
    else:
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    train_images = train_images.astype(np.float32)
    test_images = test_images.astype(np.float32)

    (train_images, test_images) = normalization(train_images, test_images)
    l=10
    if(cfr100):
        l=100
    train_labels = to_categorical(train_labels, l)
    test_labels = to_categorical(test_labels, l)

    return train_images, train_labels, test_images, test_labels

def normalization(train_images, test_images, noise=0.2):
    mean = np.mean(train_images, axis=(0, 1, 2, 3))
    std = np.std(train_images, axis=(0, 1, 2, 3))
    print(std)
    print(mean)
    train_images = (train_images - mean) / (std + 1e-7)
    
    test_images = (test_images - mean) / (std + 1e-7)
    
    return train_images, test_images
    

train_images, train_labels, test_images, test_labels = load_images(cfr100=False)


# data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(train_images)

learning_rate = 0.1
def train(model,path,epochs=100,learning_rate=0.1):
  print('....Training the model....')
  checkpoint1 = ModelCheckpoint(filepath='./'+path,save_format=tf,monitor='val_loss',
                            save_weights_only=True,
                             verbose=1, 
                             save_best_only=True,
                             mode='min')
  model.compile(optimizer=tf.keras.optimizers.SGD(lr=learning_rate, momentum=0.9,nesterov=True),
    loss='categorical_crossentropy',
    metrics='categorical_accuracy')
  return model.fit_generator(datagen.flow(train_images, train_labels,batch_size=100), epochs=epochs, validation_data=(test_images, test_labels), callbacks=[checkpoint1, reduce_lr], verbose=1)

def main():
    vgg19 = VGG16_lbin()
    train(vgg19,path='LBIN_model',epochs=200)
    return 0
    
if __name__ == "__main__":
    main()
