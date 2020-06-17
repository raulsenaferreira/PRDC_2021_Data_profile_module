from plot_funcs import plot_images
from models import Models
import generate_data as gd
from datasets import Dataset
import tensorflow as tf


def test_generate_drift_data(train, test, cd_types, amount):
	train = x_train, y_train
	test = x_test, y_test
	for cd_type in cd_types:
		status = gd.generate_drift_data((train[:amount], test[:amount]), dataset, cd_type, persist_data = True)
		print(dataset, cd_type, status)

def test_generate_anomaly_data(train, test, anomaly_types, amount):
	train = x_train, y_train
	test = x_test, y_test
	for anomaly_type in anomaly_types:
		status = gd.generate_anomaly_data((train[:amount], test[:amount]), dataset, anomaly_type, persist_data = True)
		print(dataset, anomaly_type, status)

def test_generate_adversarial_data(train, test, ml_model, attack_types, amount):
	train = x_train, y_train
	test = x_test, y_test
	for attack_type in attack_types:
		status = gd.generate_adversarial_data((train, test), dataset, ml_model, attack_type, persist_data = True)
		print(dataset, attack_type, status)

def test_generate_corrupted_data(train, test, corruption_types, amount):
	train = x_train, y_train
	test = x_test, y_test
	for corruption_type in corruption_types:
		status = gd.generate_corrupted_data((train, test), dataset, corruption_type, persist_data = True)
		print(dataset, corruption_type, status)

def test_plot_generated_dataset(variation, num_samples, num_row, num_col):
	x_train, y_train, x_test, y_test = dataset.load_dataset_variation(variation)
	plot_images(x_train[:num_samples], y_train[:num_samples], num_row=num_row, num_col=num_col)

def test_plot_original_dataset(dataset_name, num_samples, num_row, num_col):
	data = Dataset(dataset_name)
	x_train, y_train, x_test, y_test = data.load_dataset()
	plot_images(x_train[:num_samples], y_train[:num_samples], num_row=num_row, num_col=num_col)



datasets = ['mnist', 'cifar10', 'gtsrb'] # 
#load model
model_file = folder+sep+dataset+filename
ml_model = tf.keras.models.load_model(model_file)

# ML threats
cd_types = ['cvt', 'cht', 'cdt', 'rotated']

anomaly_types = ['pixel_trap', 'row_add_logic', 'shifted_pixel']

attack_types = ['FGSM']

corruption_types = ['spatter', 'elastic_transform', 'gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise', 'defocus_blur', 
'glass_blur', 'zoom_blur', 'gaussian_blur', 'brightness', 'contrast', 'saturate'] 

#running tests
