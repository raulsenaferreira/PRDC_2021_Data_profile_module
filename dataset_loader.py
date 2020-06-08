import keras
from keras.datasets import mnist
import util
import generate_data as gd


class multiMNIST():
    def __init__(self, dataset = 'mnist'):        
        self.dataset = dataset
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.load_data(self.dataset)

    def load_data(self, dataset):
        if dataset == 'mnist':
            self.num_classes = 10
            return mnist.load_data()
        elif dataset == 'emnist':
            self.num_classes = 47
            return util.load_balanced_emnist()

    def drift_data(self, drift_type, mode, persist_data): 
        if mode=='generate':
            gd.drift_data((self.x_train, self.y_train), (self.x_test, self.y_test), drift_type, 
            mode=mode, persist_data=persist_data)
        
        elif mode=='load':
            #TODO
            return (x_train, y_train), (x_test, y_test)