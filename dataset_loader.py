import keras
from keras.datasets import mnist
import util
import generate_data as gd


class multiMNIST():
    def __init__(self, dataset = 'mnist'):        
        self.dataset = dataset
        self.dim = 28*28
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.load_data(self.dataset)

    def load_data(self, dataset):
        if dataset == 'mnist':
            self.num_classes = 10
            return mnist.load_data()
        elif dataset == 'emnist':
            self.num_classes = 47
            return util.load_balanced_emnist()

    def drift_data(self, drift_type, mode, persist_data=False): 
        if mode=='generate':
            status = gd.generate_drift_data((self.x_train, self.y_train), (self.x_test, self.y_test), self.dataset,
                drift_type, mode=mode, persist_data=persist_data)
            return status
        
        elif mode=='load':
            if drift_type == 'cvt' or drift_type == 'cht' or drift_type == 'cdt':
                num_train = len(self.y_train)*11
                num_test = len(self.y_test)*11
            (self.x_train, self.y_train), (self.x_test, self.y_test) = util.load_drift_mnist(
                self.dataset, drift_type, num_train, num_test, self.dim)
            
            return (self.x_train, self.y_train), (self.x_test, self.y_test)