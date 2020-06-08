import keras
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import util


class MNIST():
    def load_data(mnist_type='original', onehotencoder=False, validation_size=None):

        num_classes = 10
        # input image dimensions
        img_rows, img_cols, img_dim = 28, 28, 1

        x_train, y_train, x_test, y_test = None, None, None, None

        # the data, split between train and test sets
        if mnist_type=='original':
            (x_train, y_train), (x_test, y_test) = mnist.load_data()

        elif mnist_type=='rotated':
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            x_train, y_train = util.rotating_mnist(x_train, y_train)
            x_test, y_test = util.rotating_mnist(x_test, y_test)
        
        elif mnist_type=='extended':
            (x_train, y_train), (x_test, y_test) = util.load_extended_mnist()
            #x_train, x_test = util.reshaping_data(x_train, x_test, img_rows, img_cols, img_dim)
            #correcting a bug with rotation of the images in this dataset
            x_test, y_test = util.rotating_mnist(x_test, y_test, 270)
            #x_test = mirroring_image(x_test)
        
        elif mnist_type=='back_round':
            (x_train, y_train), (x_test, y_test) = util.load_mnist_rand_back()

        elif mnist_type=='affnist':    
            (x_train, y_train), (x_test, y_test) = util.load_batches_affnist()
            x_train, x_test = util.reshaping_data(x_train, x_test, 40, 40, img_dim)

        elif mnist_type=='moving_mnist':    
            (x_train, y_train), (x_test, y_test) = load_batches_moving_mnist()
            x_train, x_test = util.reshaping_data(x_train, x_test, 64, 64, img_dim)

        elif mnist_type=='cht_mnist':    
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            x_train, y_train = util.load_concept_mnist(x_train, y_train, mnist_type)
            x_test, y_test = util.load_concept_mnist(x_test, y_test, mnist_type)
            x_train, x_test = util.reshaping_data(x_train, x_test, 64, 64, img_dim)

        elif mnist_type=='cvt_mnist':    
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            x_train, y_train = util.load_concept_mnist(x_train, y_train, mnist_type)
            x_test, y_test = util.load_concept_mnist(x_test, y_test, mnist_type)
            x_train, x_test = util.reshaping_data(x_train, x_test, 64, 64, img_dim)

        elif mnist_type=='cdt_mnist':    
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            x_train, y_train = util.load_concept_mnist(x_train, y_train, mnist_type)
            x_test, y_test = util.load_concept_mnist(x_test, y_test, mnist_type)
            x_train, x_test = util.reshaping_data(x_train, x_test, 64, 64, img_dim)         

        # convert class vectors to binary class matrices
        if onehotencoder:
            y_train = keras.utils.to_categorical(y_train, num_classes)
            y_test = keras.utils.to_categorical(y_test, num_classes)

        return (x_train, y_train), (x_test, y_test)