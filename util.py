from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import os
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
from scipy import ndimage
import gzip
from PIL import Image
import scipy.io as spio
from skimage.transform import resize
import idx2numpy


def get_separator():
    is_windows = sys.platform.startswith('win')
    sep = '\\'

    if is_windows == False:
        sep = '/'

    return sep

sep = get_separator()


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def rotating_data(images, labels, correcting=None):

    X = []
    y = []
    bg_value = -0.5 # this is regarded as background's value black
    
    for image, label in zip(images, labels):

        if correcting==None:

            # register original data
            X.append(image)
            y.append(label)
            
            angles = [-45, -22.5, 22.5, 45]
            
            for angle in angles:
                
                new_img = ndimage.rotate(image,angle,reshape=False, cval=bg_value)
                # register new training data
                X.append(new_img)
                y.append(label)
        else:
            new_img = ndimage.rotate(image,correcting,reshape=False, cval=bg_value)
            # register new training data
            X.append(new_img)
            y.append(label)            

    # return them as arrays
    return np.asarray(X), np.asarray(y)


def decoding_data(images, labels, num, dim):
    #emnist_map = {}
    data = np.zeros(num * dim, dtype=np.uint8).reshape((num, dim))
    target = np.zeros(num, dtype=np.uint8).reshape((num, ))

    with gzip.open(images, 'rb') as f_images, gzip.open(labels, 'rb') as f_labels:
        f_images.read(16)
        f_labels.read(8)
        for i in range(num):
            target[i] = ord(f_labels.read(1))
            #emnist_map[target[i]]=f_labels.read(1)
            for j in range(dim):
                data[i, j] = ord(f_images.read(1))

    return data, target    


def load_balanced_emnist():
    train_images = 'data'+sep+'original'+sep+'e_mnist'+sep+'emnist-balanced-train-images-idx3-ubyte.gz'
    train_labels = 'data'+sep+'original'+sep+'e_mnist'+sep+'emnist-balanced-train-labels-idx1-ubyte.gz'
    test_images = 'data'+sep+'original'+sep+'e_mnist'+sep+'emnist-balanced-test-images-idx3-ubyte.gz'
    test_labels = 'data'+sep+'original'+sep+'e_mnist'+sep+'emnist-balanced-test-labels-idx1-ubyte.gz'
    num_train = 112800
    num_test = 18800
    dim = 28*28
    data_train, target_train = decoding_data(train_images, train_labels, num_train, dim)
    data_test, target_test = decoding_data(test_images, test_labels, num_test, dim)

    return (data_train, target_train), (data_test, target_test)


def load_batches_moving_mnist():
    path = 'data'+sep+'mnist_test_seq.npy'
    data = np.load(path)
    #print(data.shape)
    c = [i for i in range(len(data[0]))]
    x_train = data[0]
    y_train = c
    x_test = data[0]
    y_test = c
    return (x_train, y_train), (x_test, y_test)


def load_mnist_rand_back():
    
    train_path = 'data'+sep+'back_rand_mnist'+sep+'mnist_background_random_train.amat'
    test_path = 'data'+sep+'back_rand_mnist'+sep+'mnist_background_random_test.amat'
    train = np.loadtxt(train_path)
    test = np.loadtxt(test_path)
    print(train)


def load_mnist_img_back():
    
    train_path = 'data'+sep+'back_img_mnist'+sep+'mnist_background_images_train.amat'
    test_path = 'data'+sep+'back_img_mnist'+sep+'mnist_background_images_test.amat'
    train = np.loadtxt(train_path)
    test = np.loadtxt(test_path)

    # get train image datas
    x_train = data[:, :-1] / 1.0
    print(x_train)
    # get test image labels
    y_train = data[:, -1:]
    print(y_train)


def load_batches_affnist(num_file):
    train_path = 'data'+sep+'affnist'+sep+'training_and_validation_batches'+sep+str(num_file)+'.mat'
    train_set = loadmat(train_path)
    
    test_path = 'data'+sep+'affnist'+sep+'test_batches'+sep+str(num_file)+'.mat'
    test_set = loadmat(test_path)
   
    x_train = train_set['affNISTdata']['image']
    x_train = np.transpose(x_train)
    y_train = train_set['affNISTdata']['label_int']
    
    x_test = test_set['affNISTdata']['image']
    x_test = np.transpose(x_test)
    y_test = test_set['affNISTdata']['label_int']

    return (x_train, y_train), (x_test, y_test)


def save_data(x_train, y_train, x_test, y_test, dataset, drift_type, root_path='data'):
    
    train_path = root_path+sep+'modified'+sep+dataset+sep+drift_type+sep
    train_images = train_path+'train-images-npy.gz'
    train_labels = train_path+'train-labels-npy.gz'
    
    test_path = root_path+sep+'modified'+sep+dataset+sep+drift_type+sep
    test_images = test_path+'test-images-npy.gz'
    test_labels = test_path+'test-labels-npy.gz'
    
    dim = x_train.shape[1]
    
    if x_test.shape[1] != x_test.shape[1]:
        print("dimensions from train and test are different")
        return False

    #checking/creating directories
    os.makedirs(os.path.dirname(train_images), exist_ok=True)
    os.makedirs(os.path.dirname(train_labels), exist_ok=True)
    os.makedirs(os.path.dirname(test_images), exist_ok=True)
    os.makedirs(os.path.dirname(test_labels), exist_ok=True)
    
    #writing images
    f = gzip.GzipFile(train_images, "w")
    np.save(file=f, arr=x_train)
    f.close()

    f = gzip.GzipFile(train_labels, "w")
    np.save(file=f, arr=y_train)
    f.close()
    
    f = gzip.GzipFile(test_images, "w")
    np.save(file=f, arr=x_test)
    f.close()

    f = gzip.GzipFile(test_labels, "w")
    np.save(file=f, arr=y_test)
    f.close()

    return True


def load_drift_mnist(dataset, drift_type, num_train, num_test, dim, root_path='data'):

    train_path = root_path+sep+'modified'+sep+dataset+sep+drift_type+sep
    train_images = train_path+'train-images-npy.gz'
    train_labels = train_path+'train-labels-npy.gz'
    
    test_path = root_path+sep+'modified'+sep+dataset+sep+drift_type+sep
    test_images = test_path+'test-images-npy.gz'
    test_labels = test_path+'test-labels-npy.gz'

    f = gzip.GzipFile(train_images, "r")
    x_train = np.load(f)
    #x_train = np.frombuffer(x_train)#, dtype=i.dtype
    #x_train = np.fromfile(f)
    
    f = gzip.GzipFile(train_labels, "r")
    y_train = np.load(f)

    f = gzip.GzipFile(test_images, "r")
    x_test = np.load(f)

    f = gzip.GzipFile(test_labels, "r")
    y_test = np.load(f)
    
    #print("load_drift_mnist: ", x_train.shape)

    return (x_train, y_train), (x_test, y_test)


def transformations(name):
        
    step_size = 4 
    boundary = 20 #image limits from the center
    repetitions = int(boundary/2)+1 #number of interpolations

    a = [1]*repetitions
    b = [0]*repetitions
    c = [0]*repetitions #left/right (i.e. 5/-5)
    d = [0]*repetitions
    e = [1]*repetitions
    f = [0]*repetitions #up/down (i.e. 5/-5)

    if name == 'cht':
        c = np.arange(boundary, -boundary-step_size, -step_size).tolist()
    elif name == 'cvt':
        f = np.arange(-boundary, boundary+step_size, step_size).tolist()
    elif name == 'cdt':
        c = np.arange(boundary, -boundary-step_size, -step_size).tolist()
        f = np.arange(-boundary, boundary+step_size, step_size).tolist()

    return (a,b,c,d,e,f)


def generate_data_translations(images, labels, name):

    X = []
    y = []

    pixels_added = 18
    transf = transformations(name)
    num_interp = len(transf[0])

    for image, label in zip(images, labels):
        #increasing image
        image = np.pad(image, ((pixels_added,pixels_added),(pixels_added,pixels_added)), 'constant')
        #print(image.shape)    
        for n in range(0, num_interp):
            #print(n)
            #print(transf[0][n])
            t = (transf[0][n], transf[1][n], transf[2][n], transf[3][n], transf[4][n], transf[5][n])
            #print(t)
            img = Image.fromarray(image)
            new_img = img.transform(img.size, Image.AFFINE, t)
            #new_img = resize(new_img, (32, 32))
            new_img = img.resize((28, 28), Image.ANTIALIAS)
            new_img = np.array(new_img)
            # register new training data
            X.append(new_img)
            y.append(label)

    # return them as arrays
    return np.asarray(X), np.asarray(y)


def reshaping_data(x_train, x_test, img_rows, img_cols, img_dim):
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], img_dim, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], img_dim, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_dim)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_dim)
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    return x_train, x_test