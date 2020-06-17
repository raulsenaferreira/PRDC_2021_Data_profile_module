import sys
import numpy as np
import keras
import keras.backend as K
import gzip
from PIL import Image
import scipy.io as spio


def get_separator():
    is_windows = sys.platform.startswith('win')
    sep = '\\'

    if is_windows == False:
        sep = '/'

    return sep

#globals
sep = get_separator()
img_rows = 28
img_cols = 28
img_dim = 1

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
    dim = img_cols*img_dim
    data_train, target_train = decoding_data(train_images, train_labels, num_train, dim)
    data_test, target_test = decoding_data(test_images, test_labels, num_test, dim)
    data_train, data_test = reshaping_data(data_train, data_test, img_rows, img_cols, img_dim)

    return (data_train, target_train), (data_test, target_test)


def load_ardis_mnist():
    #Reading data:
    data_train=np.loadtxt('data'+sep+'original'+sep+'ardis_mnist'+sep+'ARDIS_train_2828.csv', dtype='float')
    target_train=np.loadtxt('data'+sep+'original'+sep+'ardis_mnist'+sep+'ARDIS_train_labels.csv', dtype='float')
    data_test=np.loadtxt('data'+sep+'original'+sep+'ardis_mnist'+sep+'ARDIS_test_2828.csv', dtype='float')
    target_test=np.loadtxt('data'+sep+'original'+sep+'ardis_mnist'+sep+'ARDIS_test_labels.csv', dtype='float')
    data_train, data_test = reshaping_data(data_train, data_test, img_rows, img_cols, img_dim)
    
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


def load_data(dataset, variation_type, root_path='data'):

    fixed_path = root_path+sep+'modified'+sep+dataset+sep+variation_type+sep
    train_images = fixed_path+'train-images-npy.gz'
    train_labels = fixed_path+'train-labels-npy.gz'
    
    test_images = fixed_path+'test-images-npy.gz'
    test_labels = fixed_path+'test-labels-npy.gz'

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


def reshaping_data(x_train, x_test, img_rows, img_cols, img_dim):
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], img_dim, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], img_dim, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols)
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    return x_train, x_test