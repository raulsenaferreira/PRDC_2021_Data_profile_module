import sys
from plot_funcs import plot_images
from models import Models
import generate_data as gd
from datasets import Dataset
import tensorflow as tf
import util


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

def test_plot_generated_dataset(dataset_name, variation, threat_type, num_samples, num_row, num_col):
	dataset = Dataset(dataset_name)
	#x_train, y_train, x_test, y_test = dataset.load_dataset()
	#x_train, y_train, x_test, y_test = dataset.load_dataset_variation(variation)
	(_, _), (x_test, y_test) = util.load_dataset_variation(threat_type, variation, dataset_name, 'test')
	plot_images(x_test[:num_samples], y_test[:num_samples], num_row=num_row, num_col=num_col)

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
dataset_name = 'gtsrb'
# testing ploting original dataset ### OK
#test_plot_original_dataset(dataset_name, num_samples, num_row, num_col)

'''
# Testing distributional shift  ### NOT OK (no difference in the image)
cd_types = ['cvt', 'cht', 'cdt', 'rotated']
for cd_type in cd_types:
	test_generate_drift_data(dataset_name, cd_type, num_samples)
	test_plot_generated_dataset(dataset_name, cd_type, num_samples, num_row, num_col)

'''

# Testing anomalies  ### NOT OK (no difference in the image)
threat_type = 'anomaly_detection'
anomaly_types = ['pixel_trap', 'row_add_logic', 'shifted_pixel']
for anomaly_type in anomaly_types:
	test_generate_anomaly_data(dataset_name, anomaly_type, num_samples)
	test_plot_generated_dataset(dataset_name, anomaly_type, threat_type, num_samples, num_row, num_col)

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
'''








'''
elif args.sub_field_arg == 'test_distributional_shift':
			
			threat_type = 'distributional_shift'
			arr_variation = ['brightness_severity_1', 'contrast_severity_1', 'defocus_blur_severity_1',
			'brightness_severity_5', 'contrast_severity_5', 'defocus_blur_severity_5',
			'elastic_transform_severity_1', 'gaussian_blur_severity_1', 'glass_blur_severity_1',
			'elastic_transform_severity_5', 'gaussian_blur_severity_5', 'glass_blur_severity_5',
			'saturate_severity_1', 'zoom_blur_severity_1', 'saturate_severity_5', 'zoom_blur_severity_5']

			for variation in arr_variation:
				(_, _), (x, y) = util.load_dataset_variation(threat_type, variation, dataset_name, 'test')
				
				print('Test set for {}, {}, {} with shape: {} {}'.format(threat_type, variation, dataset_name, np.shape(x), np.shape(y)))
				plot_funcs.plot_images(x[:50], y[:50], 5, 10)

			# good images CIFAR: brightness_severity_5 (better change to 3), 
			# good images GTSRB: brightness_severity_5 (better change to 3)

		elif args.sub_field_arg == 'test_noise':
			
			threat_type = 'noise'
			arr_variation = ['gaussian_noise_severity_1', 'impulse_noise_severity_1', 'shot_noise_severity_1',
			'gaussian_noise_severity_5', 'impulse_noise_severity_5', 'shot_noise_severity_5',
			'spatter_severity_1', 'speckle_noise_severity_1', 'spatter_severity_5', 'speckle_noise_severity_5']

			for variation in arr_variation:
				(_, _), (x, y) = util.load_dataset_variation(threat_type, variation, dataset_name, 'test')
				
				print('Test set for {}, {}, {} with shape: {} {}'.format(threat_type, variation, dataset_name, np.shape(x), np.shape(y)))
				plot_funcs.plot_images(x[:50], y[:50], 5, 10)

			#no good images

		elif args.sub_field_arg == 'test_novelty_detection':
			# it is ok
			threat_type = 'novelty_detection'
			arr_variation = ['gtsrb_btsc', 'gtsrb_cifar10', 'cifar10_gtsrb']

			for variation in arr_variation:
				(_, _), (x, y) = util.load_dataset_variation(threat_type, variation, None, 'test')
				
				print('Test set for {}, {} with shape: {} {}'.format(threat_type, variation, np.shape(x), np.shape(y)))
				plot_funcs.plot_images(x[:50], y[:50], 5, 10)
'''