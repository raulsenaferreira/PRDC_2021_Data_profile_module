import sys
from plot_funcs import plot_images
from models import Models
import generate_data as gd
from datasets import Dataset
import tensorflow as tf



def get_separator():
	is_windows = sys.platform.startswith('win')
	sep = '\\'
	if is_windows == False:
		sep = '/'
	return sep

def test_plot_original_dataset(dataset_name, num_samples, num_row, num_col):
	data = Dataset(dataset_name)
	x_train, y_train, x_test, y_test = data.load_dataset()
	print("printing training:", dataset_name)
	plot_images(x_train[:num_samples], y_train[:num_samples], num_row=num_row, num_col=num_col)
	print("printing testing:", dataset_name)
	plot_images(x_test[:num_samples], y_test[:num_samples], num_row=num_row, num_col=num_col)

def test_generate_drift_data(dataset_name, cd_type, amount):
	data = Dataset(dataset_name)
	x_train, y_train, x_test, y_test = data.load_dataset()
	train = x_train, y_train
	test = x_test, y_test
	print("generating", cd_type, dataset_name)
	status = gd.generate_translated_data(train[:amount], test[:amount], dataset_name, cd_type, persist_data = True)
	print(dataset_name, cd_type, status)

def test_generate_anomaly_data(dataset_name, anomaly_type, amount):
	data = Dataset(dataset_name)
	x_train, y_train, x_test, y_test = data.load_dataset()
	train = x_train, y_train
	test = x_test, y_test
	print("generating", anomaly_type, dataset_name)
	status = gd.generate_anomaly_data(train[:amount], test[:amount], dataset_name, anomaly_type, persist_data = True)
	print(dataset_name, anomaly_type, status)

def test_generate_adversarial_data(dataset_name, model_file, attack_type, amount):
	ml_model = tf.keras.models.load_model(model_file)
	data = Dataset(dataset_name)
	x_train, y_train, x_test, y_test = data.load_dataset()
	train = x_train[:amount], y_train[:amount]
	test = x_test[:amount], y_test[:amount]
	print("generating", attack_type, dataset_name)
	status = gd.generate_adversarial_data((train, test), dataset_name, ml_model, attack_type, persist_data = True)
	print(dataset_name, attack_type, status)

def test_generate_corrupted_data(dataset_name, corruption_type, amount):
	data = Dataset(dataset_name)
	x_train, y_train, x_test, y_test = data.load_dataset()
	train = x_train[:amount], y_train[:amount]
	test = x_test[:amount], y_test[:amount]
	print("generating", corruption_type, dataset_name)
	status = gd.generate_corrupted_data(train, test, dataset_name, corruption_type, persist_data = True)
	print(dataset_name, corruption_type, status)

def test_plot_generated_dataset(dataset_name, variation, num_samples, num_row, num_col):
	dataset = Dataset(dataset_name)
	x_train, y_train, x_test, y_test = dataset.load_dataset()
	x_train, y_train, x_test, y_test = dataset.load_dataset_variation(variation)
	plot_images(x_train[:num_samples], y_train[:num_samples], num_row=num_row, num_col=num_col)

def test_plot_adv_generated_dataset(dataset_name, variation, num_samples, num_row, num_col):
	dataset = Dataset(dataset_name)
	x_train, y_train, x_test, y_test = dataset.load_dataset()
	x_train, y_train, y_train_miss, x_test, y_test, y_test_miss = dataset.load_dataset_adv(variation)
	plot_images(x_train[:num_samples], y_train_miss[:num_samples], num_row=num_row, num_col=num_col)



print("running test...")
sep =  get_separator()
num_samples = 40
num_row = 4
num_col = 10

# Testing MNIST dataset and its variants
dataset_name = 'mnist'
# testing ploting original dataset ### OK
#test_plot_original_dataset(dataset_name, num_samples, num_row, num_col)

'''
# Testing distributional shift  ### NOT OK (no difference in the image)
cd_types = ['cvt', 'cht', 'cdt', 'rotated']
for cd_type in cd_types:
	test_generate_drift_data(dataset_name, cd_type, num_samples)
	test_plot_generated_dataset(dataset_name, cd_type, num_samples, num_row, num_col)


# Testing anomalies  ### NOT OK (no difference in the image
anomaly_types = ['pixel_trap', 'row_add_logic', 'shifted_pixel']
for anomaly_type in anomaly_types:
	test_generate_anomaly_data(dataset_name, anomaly_type, num_samples)
	test_plot_generated_dataset(dataset_name, anomaly_type, num_samples, num_row, num_col)
'''

# testing adversarial examples
attack_types = ['FGSM']
folder = 'models'
filename = '_tf_keras.h5'
model_file = folder+sep+dataset_name+filename
for attack_type in attack_types:
	test_generate_adversarial_data(dataset_name, model_file, attack_type, num_samples)
	test_plot_adv_generated_dataset(dataset_name, attack_type, 10, 2, 5)

# testing noise
corruption_types = ['spatter', 'elastic_transform', 'gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise', 'defocus_blur', 
'glass_blur', 'zoom_blur', 'gaussian_blur', 'brightness', 'contrast', 'saturate'] 
for corruption_type in corruption_types:
	test_generate_corrupted_data(dataset_name, corruption_type, num_samples)
	test_plot_generated_dataset(dataset_name, corruption_type, num_samples, num_row, num_col)