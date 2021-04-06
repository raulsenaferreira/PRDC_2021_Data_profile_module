import os
import sys
from models import Models
import generate_data as gd
from datasets import Dataset
import tensorflow as tf
from tensorflow import keras
import argparse
import util
import plot_funcs
import numpy as np


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


def generate_transformed_data(dataset_name, transformation_type, threat_type, severity):
	data = Dataset(dataset_name)
	x_train, y_train, x_test, y_test = data.load_dataset()
	train = x_train, y_train
	test = x_test, y_test
	print("generating", transformation_type, dataset_name)
	
	if threat_type == 'anomaly_detection':
		status = gd.generate_anomaly_data(train, test, dataset_name, transformation_type, severity, persist_data = True)
	else:
		if transformation_type == 'rotated':
			status = gd.generate_translated_data(train, test, dataset_name, transformation_type, persist_data = True)
		else:
			status = gd.generate_corrupted_data(train, test, dataset_name, transformation_type, threat_type, severity, persist_data = True)

	return status


# globals
arr_transformations = None

# source datasets
original_dataset_names = ['cifar10', 'gtsrb']

#rotation_types = ['cvt', 'cht', 'cdt', 'rotated']

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	
	#parser.add_argument("experiment_type_arg", help="Type of experiment (ID or OOD)")
	
	parser.add_argument("sub_field_arg", help="Type of ML problem (novelty_detection, distributional_shift,\
	 anomaly_detection, adversarial_attack, noise)")

	parser.add_argument("save_experiments", type=int, help="Save experiments (1 for True or 0 for False)")

	parser.add_argument("parallel_execution", type=int, help="Parallelize experiments up to the number of physical \
		cores in the machine (1 for True or 0 for False)")

	parser.add_argument("verbose", type=int, help="Print the processing progress (1 for True or 0 for False)")

	parser.add_argument("dir_path_write", default='data', help="root folder which  generated data will be saved")

	args = parser.parse_args()

	if args.sub_field_arg == 'novelty_detection':
		# each tuple corresponds to (ID data, OOD data)
		novelty_types = [('gtsrb', 'btsc')] #, ('cifar10', 'gtsrb'), ('gtsrb', 'cifar10')
		for dataset_names in novelty_types:
			status = gd.generate_novelty_data(dataset_names, args.save_experiments, args.parallel_execution, args.verbose, args.dir_path_write) # , proportion=0.3 = percentage of ID data kept for test data
			print(dataset_names, status)

	elif args.sub_field_arg == 'distributional_shift':
		array_severity = [2, 5]
		arr_transformations = ['snow', 'fog', 'brightness', 'contrast', 'saturate', 'rotated']

	elif args.sub_field_arg == 'anomaly_detection':
		array_severity = [1, 3]
		arr_transformations = ['pixel_trap', 'row_add_logic', 'shifted_pixel']

	elif args.sub_field_arg == 'noise':
		array_severity = [2, 5]
		# Specifically for cifar10 and gtsrb datasets. For mnist one can load the same corruptions from tf datasets
		arr_transformations = ['spatter', 'gaussian_noise', 'shot_noise', 'speckle_noise', 'defocus_blur',
		'elastic_transform', 'impulse_noise', 'glass_blur', 'zoom_blur', 'gaussian_blur', 'pixelate']

	elif args.sub_field_arg == 'adversarial_attack': #Not fully working yet
		folder = 'models'
		filename = '_tf_keras.h5'
		# uncomment and execute this line below (once) to build the base models
		#generate_models(filename, folder, datasets)

		attack_types = ['FGSM']
		#load model
		model_file = os.path.join(folder,dataset_name+filename)
		ml_model = tf.keras.models.load_model(model_file)
		for attack_type in attack_types:
			status = gd.generate_adversarial_data(train, test, dataset_name, ml_model, attack_type, args.save_experiments)
			print(dataset_name, attack_type, status)

	else:
		print("OOD type not found!!")

	for dataset_name in original_dataset_names:
		if arr_transformations != None:
			for transformation_type in arr_transformations:
				for severity in array_severity:
					status = generate_transformed_data(dataset_name, transformation_type, args.sub_field_arg, severity)
					#status = gd.generate_anomaly_data(train, test, dataset_name, anomaly_type, args.save_experiments)
					print(dataset_name, transformation_type, 'severity:'+str(severity), status)