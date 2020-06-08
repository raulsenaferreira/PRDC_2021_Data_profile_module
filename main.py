import dataset_loader as dl
import DataGenerator as dg

#generating datasets
dataset = dl.multiMNIST('mnist')
status = dataset.drift_data('cvt', mode='generate', persist_data=True)
print(status)