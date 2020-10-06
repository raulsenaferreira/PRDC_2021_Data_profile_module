import util
import numpy as np
import collections
from threats import adv_attack
from threats import corruptions
from threats import anomalies
from threats import geometric_transformations



def generate_translated_data(train, test, dataset, drift_type, persist_data = False):
    row, col, dim = 28, 28, 1 #default for mnist
    success = False
    (x_train, y_train) = train
    (x_test, y_test) = test
    
    if dataset == 'cifar10':
        row, col, dim = 32, 32, 3
    elif dataset =='gtsrb':
        row, col, dim = 28, 28, 3
    
    if drift_type == 'cvt' or drift_type == 'cht' or drift_type == 'cdt':
        x_train, y_train = geometric_transformations.generate_data_translations(x_train, y_train, drift_type, row, col, dim)
        x_test, y_test = geometric_transformations.generate_data_translations(x_test, y_test, drift_type, row, col, dim)
    
    elif drift_type=='rotated':
        x_train, y_train = geometric_transformations.rotating_data(x_train, y_train)
        x_test, y_test = geometric_transformations.rotating_data(x_test, y_test)
        #img_rows, img_cols, img_dim = 28, 28, 1

    #elif drift_type=='extended':
        #x_train, x_test = util.reshaping_data(x_train, x_test, img_rows, img_cols, img_dim)
        #correcting a bug with rotation of the images in this dataset
        #x_test, y_test = util.rotating_data(x_test, y_test, 270)
        #x_test = mirroring_image(x_test)

    #elif drift_type=='affnist':    
        #(x_train, y_train), (x_test, y_test) = util.load_batches_affnist()
        #x_train, x_test = util.reshaping_data(x_train, x_test, 40, 40, img_dim)

    #reshaping data
    #x_train, x_test = util.reshaping_data(x_train, x_test, img_rows, img_cols, img_dim) 

    if persist_data:
        success = util.save_data(x_train, y_train, x_test, y_test, dataset, drift_type)
        
        return success
        

def generate_anomaly_data(train, test, dataset, anomaly_type, persist_data = False):
    success = False
    (x_train, y_train) = train
    (x_test, y_test) = test
    x_train = anomalies.anomaly(x_train, anomaly_type)
    x_test = anomalies.anomaly(x_test, anomaly_type)

    if persist_data:
        success = util.save_data(x_train, y_train, x_test, y_test, dataset, anomaly_type)
        
        return success


def generate_adversarial_data(data, dataset_name, ml_model, attack_type, persist_data = False):
    success = False
    x_train, y_train, y_wrong_train, x_test, y_test, y_wrong_test = None, None, None, None, None, None
    
    if attack_type == 'FGSM':
        x_train, y_train, y_wrong_train, x_test, y_test, y_wrong_test = adv_attack.perform_attacks(data, ml_model, dataset_name)

    if persist_data:
        success = util.save_adversarial_data(x_train, y_train, y_wrong_train, x_test, y_test, y_wrong_test, dataset_name, attack_type)

    return success


def perform_corruptions(train, test, corruption, array_severity):
    corrupted_x_train, corrupted_y_train, corrupted_x_test, corrupted_y_test = [], [], [], []
    (x_train, y_train) = train
    (x_test, y_test) = test

    print("corrupting train set")
    for img, label in zip(x_train, y_train):
        corrupted_y_train.append(label)
        corrupted_x_train.append(np.uint8(corruption(img)))
    
    print("corrupting test set")
    for img, label in zip(x_test, y_test):
        corrupted_y_test.append(label)
        corrupted_x_test.append(np.uint8(corruption(img)))

    return np.asarray(corrupted_x_train), np.asarray(corrupted_y_train), np.asarray(corrupted_x_test), np.asarray(corrupted_y_test)


def generate_corrupted_data(train, test, dataset_name, corruption_type, persist_data = False):
    success = False
    array_severity = [1, 5]

    d = collections.OrderedDict()
    #d['Snow'] = corruptions.snow #incluir figura
    #d['Frost'] = corruptions.frost #incluir figura
    #d['Fog'] = corruptions.fog #incluir figura
    d['spatter'] = corruptions.spatter
    d['elastic_transform'] = corruptions.elastic_transform
    d['gaussian_noise'] = corruptions.gaussian_noise
    d['shot_noise'] = corruptions.shot_noise
    d['impulse_noise'] = corruptions.impulse_noise
    d['speckle_noise'] = corruptions.speckle_noise
    d['defocus_blur'] = corruptions.defocus_blur
    d['glass_blur'] = corruptions.glass_blur
    d['zoom_blur'] = corruptions.zoom_blur
    d['gaussian_blur'] = corruptions.gaussian_blur
    #d['Motion Blur'] = corruptions.motion_blur #failure
    d['brightness'] = corruptions.brightness
    d['contrast'] = corruptions.contrast
    d['saturate'] = corruptions.saturate
    #d['Pixelate'] = corruptions.pixelate #failure
    #d['JPEG'] = corruptions.jpeg_compression #failure    

    for severity in array_severity:
        corruption = lambda clean_img: d[corruption_type](clean_img, severity)
        x_train, y_train, x_test, y_test = perform_corruptions(train, test, corruption, severity)

        if persist_data:
            success = util.save_data(x_train, y_train, x_test, y_test, dataset_name, corruption_type+"_severity_"+str(severity))
            print(dataset_name, corruption_type+"_severity_"+str(severity), success)
    
    return success


def generate_novelty():
    #loading OOD dataset
    OOD_dataset = Dataset(ood_dataset_name)
    
    ood_X, ood_y = OOD_dataset.load_dataset(mode='test_entire_data')
    ood_y += classes_to_monitor_ID #avoiding same class numbers for the two datasets

    #concatenate and shuffling ID and OOD datasets
    X = np.vstack([X, ood_X])
    y = np.hstack([y, ood_y])
    X, y = unison_shuffled_copies(X, y)

    print("Final dataset shape", X.shape, y.shape)
    dataset.dataset_ID_name = dataset_name
    dataset.dataset_OOD_name = ood_dataset_name