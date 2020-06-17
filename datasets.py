import os
import util
import keras
from keras.datasets import mnist
from keras.datasets import cifar10
import numpy as np
import pandas as pd
from PIL import Image
import pickle
import cv2
import keras.backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


class Dataset:
    """docstring for Dataset"""
    def __init__(self, dataset_name):
        super(Dataset, self).__init__()
        self.dataset_name = dataset_name
        self.width = 28
        self.height = 28
        self.channels = 0
        self.sep = util.get_separator()
        self.testPath = ''
        self.num_classes = 0
        self.trainPath = ''
        self.testPath = ''
        self.validation_size = None
    

    def load_mnist(self, onehotencoder=True):
        self.num_classes = 10
        # input image dimensions
        img_rows, img_cols, img_dim = 28, 28, 1

        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], img_dim, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], img_dim, img_rows, img_cols)
            input_shape = (img_dim, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_dim)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_dim)
            input_shape = (img_rows, img_cols, img_dim)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        #x_train, x_valid, y_train, y_valid = train_test_split(x_train,y_train,test_size = self.validation_size)

        #x_train, x_valid = train_test_split(x_train, test_size=self.validation_size, shuffle=False)
        #y_train, y_valid = train_test_split(y_train, test_size=self.validation_size, shuffle=False)

        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        if onehotencoder:
            y_train = keras.utils.to_categorical(y_train, self.num_classes)
            y_valid = keras.utils.to_categorical(y_valid, self.num_classes)
            y_test = keras.utils.to_categorical(y_test, self.num_classes)

        return x_train, y_train, x_valid, y_valid, x_test, y_test, input_shape


    def load_GTSRB_csv(self, filename):
        n_inputs = self.height * self.width * self.channels
        y_test=pd.read_csv(self.trainPath+filename)
        labels=y_test['Path'].values
        y_test=y_test['ClassId'].values
        y_test=np.array(y_test)

        d=[]
        for j in labels:
            path = self.trainPath+j.replace("/", "\\")
            
            i1=cv2.imread(path)
            i2=Image.fromarray(i1,'RGB')
            i3=i2.resize((self.height, self.width))
            d.append(np.array(i3))

        X_test=np.array(d)
        X_test = X_test.astype('float32')/255 
        print("Shape :", X_test.shape, y_test.shape)
        return X_test, y_test


    def load_cifar10(self, trainPath, testPath):
        def unpickle(file):
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict

        num_batches = 5
        data = unpickle(self.trainPath+'1')
        x_train = data[b'data']
        y_train = data[b'labels']
        
        for i in range(1, num_batches+1):
            data = unpickle(self.trainPath+str(i))
            x_train = np.append(x_train, data[b'data'])
            y_train = np.append(y_train, data[b'labels'])

        print('x_train.shape', x_train.shape)

        data = unpickle(testPath)
        x_test, y_test = data[b'data'], data[b'labels']
        print('x_test.shape', x_test.shape)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        return x_train, y_train, x_test, y_test


    def load_dataset(self, ):
        img_rows, img_cols, img_dim = 0, 0, 0
        data = []

        if self.dataset_name == 'mnist':
            self.num_classes = 10
            self.channels = 1
            img_rows, img_cols = 28, 28
         
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            data = x_train, y_train, x_test, y_test

        elif self.dataset_name == 'gtsrb':
            self.num_classes = 43
            self.channels = 3
            img_rows, img_cols = 28, 28

            self.trainPath = 'data'+self.sep+'original'+self.sep+'GTS_dataset'+self.sep
            X_train, y_train = self.load_GTSRB_csv("Train.csv")
            X_test, y_test = self.load_GTSRB_csv("Test.csv") #self.load_GTRSB_csv(self.testPath)

            data = X_train, y_train, X_test, y_test

        elif self.dataset_name == 'cifar10':
            self.num_classes = 10
            self.channels = 3
            img_rows, img_cols = 32, 32
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()

            data = x_train, y_train, x_test, y_test
        
        else:
            print("Dataset not found!!")

        return data


    def load_dataset_variation(self, variation):
        # TODO
        pass