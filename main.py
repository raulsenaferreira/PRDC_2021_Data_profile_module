import os
import sys
from models import Models
import generate_data as gd
from datasets import Dataset
import tensorflow as tf
from tensorflow import keras



def get_separator():
	is_windows = sys.platform.startswith('win')
	sep = '\\'

	if is_windows == False:
		sep = '/'

	return sep


def generate_models(filename, folder, datasets):
	ml = Models()
	for dataset in datasets:
		data = Dataset(dataset)
		x_train, y_train, x_test, y_test = data.load_dataset()
		model_file = folder+sep+dataset+filename
		
		if dataset == 'mnist':
			ml.create_mnist_model(x_train, y_train, x_test, y_test, model_file)
		elif dataset == 'gtsrb':
			ml.create_gtsrb_model(x_train, y_train, x_test, y_test, model_file)
		elif dataset == 'cifar10':
			ml.create_cifar10_model(x_train, y_train, x_test, y_test, model_file)
	
	print("succesfully built all models")



# globals
sep =  get_separator()
folder = 'models'
filename = '_tf_keras.h5'

# uncomment and execute this line below (once) to build the base models
#generate_models(filename, folder, datasets)

# datasets
datasets = ['cifar10', 'gtsrb'] #  'mnist', 

# ML threats
cd_types = [] #['cvt', 'cht', 'cdt', 'rotated']
anomaly_types = [] #['pixel_trap', 'row_add_logic', 'shifted_pixel']
attack_types = [] #['FGSM']
corruption_types = ['spatter', 'elastic_transform', 'gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise',
                    'defocus_blur', 'glass_blur', 'zoom_blur', 'gaussian_blur', 'brightness', 'contrast', 'saturate']

for dataset in datasets:
	data = Dataset(dataset)
	x_train, y_train, x_test, y_test = data.load_dataset()
	train = x_train, y_train
	test = x_test, y_test
	#load model
	model_file = folder+sep+dataset+filename
	ml_model = tf.keras.models.load_model(model_file)
	
	for cd_type in cd_types:
		status = gd.generate_drift_data((train, test), dataset, cd_type, persist_data = True)
		print(dataset, cd_type, status)

	for anomaly_type in anomaly_types:
		status = gd.generate_anomaly_data((train, test), dataset, anomaly_type, persist_data = True)
		print(dataset, anomaly_type, status)

	for attack_type in attack_types:
		status = gd.generate_adversarial_data((train, test), dataset, ml_model, attack_type, persist_data = True)
		print(dataset, attack_type, status)

	for corruption_type in corruption_types:
		status = gd.generate_corrupted_data((train, test), dataset, corruption_type, persist_data = True)
		print(dataset, corruption_type, status)