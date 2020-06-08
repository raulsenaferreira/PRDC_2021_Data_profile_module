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


def save_rotated_MNIST(train_data, train_labels, test_data, test_labels):
    if not os.path.isdir("data/train-images"):
        os.makedirs("data/train-images")
    if not os.path.isdir("data/test-images"):
        os.makedirs("data/test-images")
    # process train data
    with open("data/train-labels.csv", 'w') as csvFile:
        writer = csv.writer(csvFile, delimiter=',', quotechar='"')
        for i in range(len(train_data)):
            if i<20:
                imsave("data/train-images/" + str(i) + ".jpg", train_data[i][:,:,0])
            writer.writerow(["train-images/" + str(i) + ".jpg", train_labels[i]])
    # repeat for test data
    with open("data/test-labels.csv", 'w') as csvFile:
        writer = csv.writer(csvFile, delimiter=',', quotechar='"')
        for i in range(len(test_data)):
            #imsave("mnist/test-images/" + str(i) + ".jpg", test_data[i][:,:,0])
            writer.writerow(["test-images/" + str(i) + ".jpg", test_labels[i]])



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


def write_labeldata(labeldata, outputfile):
  header = np.array([0x0801, len(labeldata)], dtype='>i4')
  with open(outputfile, "wb") as f:
    f.write(header.tobytes())
    f.write(labeldata.tobytes())

def write_imagedata(imagedata, outputfile):
  header = np.array([0x0803, len(imagedata), 28, 28], dtype='>i4')
  with open(outputfile, "wb") as f:
    f.write(header.tobytes())
    f.write(imagedata.tobytes())

def save_data(x_train, y_train, x_test, y_test, drift_type, root_path='data'):
    train_path = root_path+sep+'modified'+sep+drift_type+sep
    train_images = train_path+'train-images-idx3-ubyte.gz'
    train_labels = train_path+'train-labels-idx1-ubyte.gz'
    
    test_path = root_path+sep+'modified'+sep+drift_type+sep
    test_images = test_path+'test-images-idx3-ubyte.gz'
    test_labels = test_path+'test-labels-idx1-ubyte.gz'
    
    dim = x_train.shape[1]
    
    if x_test.shape[1] != x_test.shape[1]:
        print("dimensions from train and test are different")
        return False

    header_train_img = np.array([0x0803, len(x_train), dim, dim], dtype='>i4')  
    header_train_lbl = np.array([0x0803, len(y_train)], dtype='>i4')
    header_test_img = np.array([0x0801, len(x_test), dim, dim], dtype='>i4')  
    header_test_lbl = np.array([0x0801, len(y_test)], dtype='>i4')

    #checking/creating directories
    os.makedirs(os.path.dirname(train_images), exist_ok=True)
    os.makedirs(os.path.dirname(train_labels), exist_ok=True)
    os.makedirs(os.path.dirname(test_images), exist_ok=True)
    os.makedirs(os.path.dirname(test_labels), exist_ok=True)
    
    #writing images
    with open(train_images, "wb") as f:
        f.write(header_train_img.tobytes())
        f.write(x_train.tobytes())
    with open(train_labels, "wb") as f:
        f.write(header_train_lbl.tobytes())
        f.write(y_train.tobytes())
    with open(test_images, "wb") as f:
        f.write(header_test_img.tobytes())
        f.write(x_test.tobytes())
    with open(test_labels, "wb") as f:
        f.write(header_test_lbl.tobytes())
        f.write(y_test.tobytes())

    return True


def transformations(name):
        
    num_interp = 4 #number of interpolations
    boundary = 20 #image limits from the center
    repetitions = int(boundary/2)+1

    a = [1]*repetitions
    b = [0]*repetitions
    c = [0]*repetitions #left/right (i.e. 5/-5)
    d = [0]*repetitions
    e = [1]*repetitions
    f = [0]*repetitions #up/down (i.e. 5/-5)

    if name == 'cht':
        c = np.arange(boundary, -boundary-num_interp, -num_interp).tolist()
    elif name == 'cvt':
        f = np.arange(-boundary, boundary+num_interp, num_interp).tolist()
    elif name == 'cdt':
        c = np.arange(boundary, -boundary-num_interp, -num_interp).tolist()
        f = np.arange(-boundary, boundary+num_interp, num_interp).tolist()

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