import os
import util
import keras
from keras.datasets import mnist
#import tensorflow_datasets.public_api as tfds
from tensorflow.keras.datasets import cifar10
import numpy as np
import pandas as pd
from PIL import Image
import pickle
import cv2
import keras.backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import plot_funcs


class Dataset:
    """docstring for Dataset"""
    def __init__(self, dataset_name):
        super(Dataset, self).__init__()
        self.dataset_name = dataset_name
        self.data_dir = os.path.join('data', 'original')
        self.width = 28
        self.height = 28
        self.channels = 0
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
        y=pd.read_csv(os.path.join(self.data_dir, filename))
        labels=y['Path'].values
        y=y['ClassId'].values
        y=np.array(y)

        d=[]
        for j in labels:
            path = os.path.join(self.data_dir, j) #.replace("/", "\\")
            
            i1=cv2.imread(path)
            i2=Image.fromarray(i1,'RGB')
            i3=i2.resize((self.height, self.width))
            d.append(np.array(i3))

        X=np.array(d)
        X = X.astype('float32')/255 
        print("Shape :", X.shape, y.shape)
        return X, y


    def load_cifar10(self, trainPath, testPath):
        def unpickle(file):
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict

        num_batches = 5
        data = unpickle(trainPath+'1')
        x_train = data[b'data']
        y_train = data[b'labels']
        
        for i in range(1, num_batches+1):
            data = unpickle(trainPath+str(i))
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
        x_train = np.reshape(self.height, self.width, self.channels)
        x_test = np.reshape(self.height, self.width, self.channels)

        return x_train, y_train, x_test, y_test


    def load_BTSC_dataset(self, folder, onehotencoder=False):
        # Reading the input images and putting them into a numpy array
        images=[]
        labels=[]
        
        directories = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
        
        n_inputs = self.height * self.width * self.channels

        for d in directories:
            label_dir = os.path.join(folder, d)
            file_names = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(".ppm")]

            for f in file_names:
                image=cv2.imread(f)
                #image=skimage.data.imread(f)
                image_from_array = Image.fromarray(image, 'RGB')
                size_image = image_from_array.resize((self.height, self.width))
                image = np.array(size_image)

                images.append(image)
                labels.append(int(d))

        X=np.array(images)
        X= X/255.0
        y=np.array(labels)
        '''
        s=np.arange(X.shape[0])
        np.random.seed(self.num_classes)
        np.random.shuffle(s)

        X=X[s]
        y=y[s]
        '''
        if onehotencoder:
            #Using one hote encoding for the train and validation labels
            y = to_categorical(y, self.num_classes)

        print("data shape :", X.shape)
        print("label shape :", y.shape)
                
        return X, y


    def load_dataset(self):

        if self.dataset_name == 'mnist':
            self.num_classes = 10
            self.channels = 1
            self.height, self.width = 28, 28
         
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            return x_train, y_train, x_test, y_test

        elif self.dataset_name == 'gtsrb':
            self.num_classes = 43
            self.channels = 3
            self.height, self.width = 28, 28

            self.data_dir = os.path.join(self.data_dir, 'gtsrb')
            X_train, y_train = self.load_GTSRB_csv("Train.csv")
            X_test, y_test = self.load_GTSRB_csv("Test.csv")

            return X_train, y_train, X_test, y_test

        elif self.dataset_name == 'cifar10':
            self.num_classes = 10
            self.channels = 3
            self.height, self.width = 32, 32
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            
            #plot_funcs.plot_images(x_train[:50], y_train[:50], 5, 10)
            #plot_funcs.plot_images(x_test[:50], y_test[:50], 5, 10)
            return x_train/255., np.squeeze(y_train), x_test/255., np.squeeze(y_test)

        elif self.dataset_name == 'btsc':
            self.num_classes = 62
            self.channels = 3
            self.height, self.width = 28, 28

            folder = os.path.join(self.data_dir, 'BTSC', "Training")
            x_train, y_train = self.load_BTSC_dataset(folder)
            folder = os.path.join(self.data_dir, 'BTSC', "Testing")
            x_test, y_test = self.load_BTSC_dataset(folder)
        
            return x_train, y_train, x_test, y_test

        else:
            print("Dataset {} not found!!".format(self.dataset_name))
            return None

    '''
    def load_dataset_adv(self, variation):
        img_rows, img_cols, img_dim = 0, 0, 0

        if self.dataset_name == 'mnist':
            self.num_classes = 10
            self.channels = 1
            img_rows, img_cols = 28, 28
        elif self.dataset_name == 'gtsrb':
            self.num_classes = 43
            self.channels = 3
            img_rows, img_cols = 28, 28
        elif self.dataset_name == 'cifar10':
            self.num_classes = 10
            self.channels = 3
            img_rows, img_cols = 32, 32
   
        try:
            (x_train, y_train, y_train_miss), (x_test, y_test, y_test_miss) = util.load_adv_data(self.dataset_name, variation)
            return x_train/255., np.squeeze(y_train), np.squeeze(y_train_miss), x_test/255., np.squeeze(y_test), np.squeeze(y_test_miss)
        except:
            print("Dataset not found!!")
            return None

    '''


'''
def load_cifar_10(self, subtract_pixel_mean = True, onehotencoder=True):
        img_width, img_height, img_num_channels = 32, 32, 3

        (input_train, target_train), (input_test, target_test) = cifar10.load_data()  
        input_train = input_train.astype('float32')
        input_test = input_test.astype('float32')
        input_train = input_train / 255
        input_test = input_test / 255

        # Subtracting pixel mean improves accuracy
        if subtract_pixel_mean:
            x_train_mean = np.mean(input_train, axis=0)
            input_train -= x_train_mean
            input_test -= x_train_mean

        x_train, x_valid, y_train, y_valid = train_test_split(input_train,target_train,test_size = self.validation_size)
        x_test, y_test = input_test, target_test

        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_valid.shape[0], 'validation samples')
        print(x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        if onehotencoder:
            y_train = keras.utils.to_categorical(y_train, self.num_classes)
            y_valid = keras.utils.to_categorical(y_valid, self.num_classes)
            y_test = keras.utils.to_categorical(y_test, self.num_classes)

        return x_train, y_train, x_valid, y_valid, x_test, y_test


    def load_mnist(self, onehotencoder):
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

        x_train, x_valid, y_train, y_valid = train_test_split(x_train,y_train,test_size = self.validation_size)

        #x_train, x_valid = train_test_split(x_train, test_size=self.validation_size, shuffle=False)
        #y_train, y_valid = train_test_split(y_train, test_size=self.validation_size, shuffle=False)

        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_valid.shape[0], 'validation samples')
        print(x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        if onehotencoder:
            y_train = keras.utils.to_categorical(y_train, self.num_classes)
            y_valid = keras.utils.to_categorical(y_valid, self.num_classes)
            y_test = keras.utils.to_categorical(y_test, self.num_classes)

        return x_train, y_train, x_valid, y_valid, x_test, y_test, input_shape


    


    def load_GTRSB_dataset(self, onehotencoder):
        # Reading the input images and putting them into a numpy array
        data=[]
        labels=[]
        
        n_inputs = self.height * self.width * self.channels

        for i in range(self.num_classes) :
            path = os.path.join(self.trainPath, str(i))
            #print(path)
            Class=os.listdir(path)
            for a in Class:
                try:
                    image=cv2.imread(os.path.join(path, a))
                    image_from_array = Image.fromarray(image, 'RGB')
                    size_image = image_from_array.resize((self.height, self.width))
                    data.append(np.array(size_image))
                    labels.append(i)
                except AttributeError as e:
                    print(e)
          
        x_train=np.array(data)
        x_train= x_train/255.0
        y_train=np.array(labels)

        s=np.arange(x_train.shape[0])
        np.random.seed(self.num_classes)
        np.random.shuffle(s)

        x_train=x_train[s]
        y_train=y_train[s]
        # Split Data
        X_train,X_valid,Y_train,Y_valid = train_test_split(x_train,y_train, test_size = self.validation_size)
        
        if onehotencoder:
            #Using one hote encoding for the train and validation labels
            Y_train = to_categorical(Y_train, self.num_classes)
            Y_valid = to_categorical(Y_valid, self.num_classes)
        print("Training set shape :", X_train.shape)
        #print(X_valid.shape[0], 'validation samples')
        print("Validation set shape :", X_valid.shape)
        
        return X_train,X_valid,Y_train,Y_valid


    def load_GTRSB_csv(self, filename):
        n_inputs = self.height * self.width * self.channels
        path = os.path.join(self.testPath, filename)
        y_test=pd.read_csv(path)
        labels=y_test['Path'].values
        y_test=y_test['ClassId'].values
        y_test=np.array(y_test)

        d=[]
        for j in labels:
            img_path = os.path.join(self.testPath, j)
            i1=cv2.imread(img_path)
            i2=Image.fromarray(i1,'RGB')
            i3=i2.resize((self.height, self.width))
            d.append(np.array(i3))

        X_test=np.array(d)
        X_test = X_test.astype('float32')/255 
        print("Test :", X_test.shape, y_test.shape)
        return X_test, y_test



    def load_dataset(self, mode=None, onehotencoder=False):
        data = []

        if self.dataset_name == 'MNIST':
            self.num_classes = 10
            self.channels = 1
            if mode == 'train':
                X_train, Y_train, X_valid, Y_valid, _, _, _ = self.load_mnist(onehotencoder=True)
                data = X_train, Y_train, X_valid, Y_valid
            elif mode == 'test':
                _, _, _, _, X_test, y_test, _ = self.load_mnist(onehotencoder)
                data = X_test, y_test
            elif mode == 'test_entire_data':
                X_train, Y_train, X_valid, Y_valid, X_test, y_test, _ = self.load_mnist(onehotencoder)
                data = X_train, Y_train, X_valid, Y_valid, X_test, y_test

        elif self.dataset_name == 'GTSRB':
            self.num_classes = 43
            self.channels = 3
            self.trainPath = os.path.join(self.root_dir, 'GTS_dataset', "kaggle", "Train")
            self.testPath = os.path.join(self.root_dir, 'GTS_dataset', "kaggle")
            
            if mode == 'train': 
                X_train, X_valid, Y_train, Y_valid = self.load_GTRSB_dataset(onehotencoder=True)
                data = X_train, Y_train, X_valid, Y_valid
            elif mode == 'test':
                X_test, y_test = self.load_GTRSB_csv("Test.csv")
                data = X_test, y_test
            elif mode == 'test_entire_data':
                X_train, X_valid, Y_train, Y_valid = self.load_GTRSB_dataset(onehotencoder)
                X_test, y_test = self.load_GTRSB_csv("Test.csv")
                data = X_train, Y_train, X_valid, Y_valid, X_test, y_test

        elif self.dataset_name == 'CIFAR-10':
            self.num_classes = 10
            self.channels = 3
            if mode == 'train':
                X_train, y_train, X_valid, y_valid, _, _ = self.load_cifar_10(onehotencoder=True)
                data = X_train, y_train, X_valid, y_valid
            elif mode == 'test':
                _, _, _, _, X_test, y_test = self.load_cifar_10(onehotencoder)
                data = X_test, y_test

        elif self.dataset_name == 'BTSC':
            self.num_classes = 62
            self.channels = 3
            self.trainPath = os.path.join(self.root_dir, 'BTSC', "Training")
            self.testPath = os.path.join(self.root_dir, 'BTSC', "Testing")
            if mode == 'train':
                X_train, X_valid, Y_train, Y_valid = self.load_BTSC_dataset(mode, onehotencoder=True)
                data = X_train, Y_train, X_valid, Y_valid
            elif mode == 'test':
                X_test, y_test = self.load_BTSC_dataset(mode, onehotencoder)
                data = X_test, y_test
            elif mode == 'test_entire_data':
                X_train, X_valid, Y_train, Y_valid = self.load_BTSC_dataset('train', onehotencoder)
                X_test, y_test = self.load_BTSC_dataset('test', onehotencoder)
                
                X = np.vstack([X_train, X_valid, X_test])
                y = np.hstack([Y_train, Y_valid, y_test])
                data = X, y

        else:
            print("Dataset not found!!")

        return data
'''