import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.models import Model
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from datasets import Dataset
from math import sqrt, floor
import matplotlib.pyplot as plt
import corruptions
from PIL import Image


def plot_images(data, labels, num_row, num_col):

    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
    for i in range(num_row*num_col):
        ax = axes[i//num_col, i%num_col]
        ax.imshow(np.squeeze(data[i]), cmap='gray')
        ax.set_title('{}'.format(labels[i]))
        ax.set_axis_off()
    plt.tight_layout(pad=3.0)
    plt.show()


#testing
datasets = ['mnist', 'cifar10', 'gtsrb']
cd_types = ['cvt', 'cht', 'cdt', 'rotated']
anomaly_types = ['pixel_trap', 'row_add_logic', 'shifted_pixel']
attack_types = ['FGSM']

data = Dataset(datasets[1])
x_train, y_train, x_test, y_test = data.load_dataset()


import collections

print('Using CIFAR-10 data')

d = collections.OrderedDict()
#d['Gaussian Noise'] = corruptions.gaussian_noise
#d['Shot Noise'] = corruptions.shot_noise
#d['Impulse Noise'] = corruptions.impulse_noise
#d['Defocus Blur'] = corruptions.defocus_blur
#d['Glass Blur'] = corruptions.glass_blur
#d['Motion Blur'] = corruptions.motion_blur #failure
#d['Zoom Blur'] = corruptions.zoom_blur
#d['Snow'] = corruptions.snow #incluir figura
#d['Frost'] = corruptions.frost #incluir figura
#d['Fog'] = corruptions.fog #incluir figura
#d['Brightness'] = corruptions.brightness
#d['Contrast'] = corruptions.contrast
#d['Elastic'] = corruptions.elastic_transform
#d['Pixelate'] = corruptions.pixelate #failure
#d['JPEG'] = corruptions.jpeg_compression #failure

#d['Speckle Noise'] = corruptions.speckle_noise
#d['Gaussian Blur'] = corruptions.gaussian_blur
#d['Spatter'] = corruptions.spatter
#d['Saturate'] = corruptions.saturate


for method_name in d.keys():
    print('Creating images for the corruption', method_name)
    cifar_c, labels = [], []

    for severity in [1, 5]:
        corruption = lambda clean_img: d[method_name](clean_img, severity)

        for img, label in zip(x_train[:20], y_train[:20]):
            labels.append(label)
            cifar_c.append(np.uint8(corruption(img)))
    
    plot_images(cifar_c, labels, num_row=4, num_col=10)



'''
for dataset in datasets:
    data = Dataset(dataset)
    x_train, y_train, x_test, y_test = data.load_dataset()
    uniques, ind_uniques = np.unique(y_train, return_index=True)
    num_row_col = floor(sqrt(len(ind_uniques)))
    #ts_ixs = [i for i in range(len(y_test)) if y_test[i] == 7]
    print(num_row_col, len(ind_uniques))
    plot_images(x_train[ind_uniques], uniques, num_row=num_row_col, num_col=num_row_col)
'''

'''
dataset = dl.multiMNIST('mnist')
# if you want to generate drift on the fly
#(x_train, y_train), (x_test, y_test) = dataset.drift_data('cvt', mode='generate')
# if you already generated drifted data and want to load it
(x_train, y_train), (x_test, y_test) = dataset.drift_data('cvt', mode='load')
#if you want the original dataset
#x_train, y_train, x_test, y_test = dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test
print("x_train: ", x_train.shape)
print("y_train: ", y_train.shape)
print("x_test: ", x_test.shape)
print("y_test: ", y_test.shape)
#if you need to do transformations on data
#x_train, x_test = util.reshaping_data(x_train, x_test, img_rows, img_cols, img_dim)
#x_train, x_valid, y_train, y_valid = train_test_split(x_train,y_train, test_size = validation_size)
#print('x_train shape:', x_train.shape)
#print(x_train.shape[0], 'train samples')
#print(x_test.shape[0], 'test samples')
#y_valid = keras.utils.to_categorical(y_valid, num_classes)
'''