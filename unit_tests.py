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

def test_generate_anomaly_data(dataset_name, anomaly_type, severity, amount):
	data = Dataset(dataset_name)
	x_train, y_train, x_test, y_test = data.load_dataset()
	train = x_train, y_train
	test = x_test, y_test
	print("generating", anomaly_type, dataset_name)
	status = gd.generate_anomaly_data(train[:amount], test[:amount], dataset_name, anomaly_type, severity, persist_data = True)
	print(dataset_name, anomaly_type, status)

def test_generate_adversarial_data(dataset_name, model_file, attack_type, epsilon, amount):
	ml_model = tf.keras.models.load_model(model_file)
	data = Dataset(dataset_name)
	x_train, y_train, x_test, y_test = data.load_dataset()
	train = x_train[:amount], y_train[:amount]
	test = x_test[:amount], y_test[:amount]
	
	print("generating", attack_type, dataset_name)
	status = gd.generate_adversarial_data(train, test, dataset_name, ml_model, attack_type, epsilon, persist_data = True)
	print(dataset_name, attack_type, status)

def test_generate_corrupted_data(dataset_name, corruption_type, threat_type, severity, amount):
	data = Dataset(dataset_name)
	x_train, y_train, x_test, y_test = data.load_dataset()
	train = x_train[:amount], y_train[:amount]
	test = x_test[:amount], y_test[:amount]
	print("generating", corruption_type, dataset_name)
	status = gd.generate_corrupted_data(train, test, dataset_name, corruption_type, threat_type, severity, persist_data = True)
	print(dataset_name, corruption_type, status)

def test_plot_generated_dataset(dataset_name, variation, threat_type, severity, num_samples, num_row, num_col):
	compl = '' if severity == None else '_severity_{}'.format(severity)
	#dataset = Dataset(dataset_name)
	#x_train, y_train, x_test, y_test = dataset.load_dataset()
	#x_train, y_train, x_test, y_test = dataset.load_dataset_variation(variation)
	(_, _), (x_test, y_test) = util.load_dataset_variation(threat_type, variation+compl, dataset_name, 'test')
	plot_images(x_test[:num_samples], y_test[:num_samples], num_row=num_row, num_col=num_col)

def test_plot_adv_generated_dataset(dataset_name, variation, num_samples, num_row, num_col):
	dataset = Dataset(dataset_name)
	#x_train, y_train, x_test, y_test = dataset.load_dataset()
	(x_train, y_train, y_train_miss), (x_test, y_test, y_test_miss) = util.load_adv_data(dataset_name, variation)
	plot_images(x_test[:num_samples], y_test[:num_samples], num_row=num_row, num_col=num_col)



print("running test...")
sep =  get_separator()
num_samples = 100
num_row = 10
num_col = 10


dataset_name = 'cifar10'#'cifar10','gtsrb'
# testing ploting original dataset ### OK
#test_plot_original_dataset(dataset_name, num_samples, num_row, num_col)

# Testing novelty detection OK for gtsrb and cifar10


#'''
# testing adversarial examples ### OK
attack_types = ['FGSM']
epsilon = 0.05
folder = 'models'
filename = '_tf_keras.h5'
model_file = folder+sep+dataset_name+filename

for attack_type in attack_types:
	test_generate_adversarial_data(dataset_name, model_file, attack_type, epsilon, 1000)
	test_plot_adv_generated_dataset(dataset_name, attack_type, num_samples, num_row, num_col)
#'''


'''
# Testing anomalies  ### OK for gtsrb and cifar10
array_severity = [1, 3]
threat_type = 'anomaly_detection'
anomaly_types = ['pixel_trap', 'row_add_logic', 'shifted_pixel']
for anomaly_type in anomaly_types:
	for severity in array_severity:
		#test_generate_anomaly_data(dataset_name, anomaly_type, severity, num_samples)
		test_plot_generated_dataset(dataset_name, anomaly_type, threat_type, severity, num_samples, num_row, num_col)
'''


'''
# testing noise ### OK for gtsrb and cifar10
array_severity = [5] #2, 5
corruption_types = ['spatter', 'gaussian_noise', 'shot_noise', 'speckle_noise', 'defocus_blur', spatter,
'elastic_transform', 'impulse_noise', 'glass_blur', 'zoom_blur', 'gaussian_blur'] #, 'pixelate'

threat_type = 'noise'
for corruption_type in corruption_types:
	for severity in array_severity:
		#test_generate_corrupted_data(dataset_name, corruption_type, threat_type, severity, num_samples)
		test_plot_generated_dataset(dataset_name, corruption_type, threat_type, severity, num_samples, num_row, num_col)
'''


'''
# testing distributional shift ### OK for gtsrb and cifar10
threat_type = 'distributional_shift'
arr_variation = ['snow'] #, 'fog', 'brightness', 'contrast', 'saturate'
array_severity = [2, 5]

for corruption_type in arr_variation:
	for severity in array_severity:
		#test_generate_corrupted_data(dataset_name, corruption_type, threat_type, severity, num_samples)
		test_plot_generated_dataset(dataset_name, corruption_type, threat_type, severity, num_samples, num_row, num_col)

'''
'''
# Testing rotating data  ### OK (does not make sense for signs)
threat_type = 'distributional_shift'
cd_types = ['rotated']
for cd_type in cd_types:
	#test_generate_drift_data(dataset_name, cd_type, num_samples)
	test_plot_generated_dataset(dataset_name, cd_type, threat_type, None, num_samples, num_row, num_col)
#'''


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