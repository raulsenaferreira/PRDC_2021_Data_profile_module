import util
import numpy as np
import collections
from threats import adv_attack
from threats import corruptions
from threats import anomalies
from threats import geometric_transformations
from datasets import Dataset
from skimage.transform import resize


def generate_translated_data(train, test, dataset, drift_type, persist_data = False):
    row, col, dim = 28, 28, 1 #default for mnist
    success = False
    (x_train, y_train) = train
    (x_test, y_test) = test

    # copy the images before modify them
    x_train_modified = np.copy(x_train)
    y_train_modified = np.copy(y_train)
    x_test_modified = np.copy(x_test)
    y_test_modified = np.copy(y_test)
    
    if dataset == 'cifar10':
        row, col, dim = 32, 32, 3
    elif dataset =='gtsrb':
        row, col, dim = 28, 28, 3
    
    if drift_type == 'cvt' or drift_type == 'cht' or drift_type == 'cdt':
        x_train, y_train = geometric_transformations.generate_data_translations(x_train, y_train, drift_type, row, col, dim)
        x_test, y_test = geometric_transformations.generate_data_translations(x_test, y_test, drift_type, row, col, dim)
    
    elif drift_type=='rotated':
        x_train_modified, y_train_modified = geometric_transformations.rotating_data(x_train_modified, y_train_modified)
        x_test_modified, y_test_modified = geometric_transformations.rotating_data(x_test_modified, y_test_modified)
        # concatenating and shuffling ID and OOD data for the test set
        x_test = np.vstack([x_train_modified, x_test, x_test_modified])
        y_test = np.hstack([y_train_modified, y_test, y_test_modified])
        x_test, y_test = util.unison_shuffled_copies(x_test, y_test)
        print("Final testing set shape", x_test.shape, y_test.shape)        

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
        success = util.save_data(x_train, y_train, x_test, y_test, dataset, 'distributional_shift', drift_type)
        
        return success
        

def generate_anomaly_data(train, test, dataset, anomaly_type, severity, persist_data):
    if dataset == 'cifar10':
        num_classes = 10
    elif dataset == 'gtsrb':
        num_classes = 43

    success = False
    (x_train, y_train) = train
    (x_test, y_test) = test

    # copy the images before modify them
    x_train_anomaly = np.copy(x_train)
    y_train_anomaly = np.copy(y_train)
    x_test_anomaly = np.copy(x_test)
    y_test_anomaly = np.copy(y_test)
    
    x_train_anomaly = anomalies.anomaly(x_train_anomaly, anomaly_type, severity)
    y_train_anomaly = y_train_anomaly+num_classes

    x_test_anomaly = anomalies.anomaly(x_test_anomaly, anomaly_type, severity)
    y_test_anomaly = y_test_anomaly+num_classes

    # concatenating and shuffling ID and OOD data for the test set
    x_test = np.vstack([x_train_anomaly, x_test, x_test_anomaly])
    y_test = np.hstack([y_train_anomaly, y_test, y_test_anomaly])
    x_test, y_test = util.unison_shuffled_copies(x_test, y_test)
    print("Final testing set shape", x_test.shape, y_test.shape)

    if persist_data:
        success = util.save_data(x_train, y_train, x_test, y_test, dataset, 'anomaly_detection', anomaly_type+"_severity_"+str(severity))
        
        return success


def generate_adversarial_data(data, dataset_name, ml_model, attack_type, persist_data = False):
    success = False
    x_train, y_train, y_wrong_train, x_test, y_test, y_wrong_test = None, None, None, None, None, None
    
    if attack_type == 'FGSM':
        x_train, y_train, y_wrong_train, x_test, y_test, y_wrong_test = adv_attack.perform_attacks(data, ml_model, dataset_name)

    if persist_data:
        success = util.save_adversarial_data(x_train, y_train, y_wrong_train, x_test, y_test, y_wrong_test, dataset_name, attack_type)

    return success


def perform_corruptions(train, test, num_classes, corruption, array_severity):
    corrupted_x_train, corrupted_y_train, corrupted_x_test, corrupted_y_test = [], [], [], []
    (x_train, y_train) = train
    (x_test, y_test) = test

    # copy the images before modify them
    copy_x_train = np.copy(x_train)
    copy_y_train = np.copy(y_train)
    copy_x_test = np.copy(x_test)
    copy_y_test = np.copy(y_test)

    print("corrupting train set")
    for img, label in zip(copy_x_train, copy_y_train):
        corrupted_y_train.append(label+num_classes) # avoiding same class numbers for the two datasets
        corrupted_x_train.append(np.uint8(corruption(img)))
    
    print("corrupting test set")
    for img, label in zip(copy_x_test, copy_y_test):
        corrupted_y_test.append(label+num_classes) # avoiding same class numbers for the two datasets
        corrupted_x_test.append(np.uint8(corruption(img)))

    # concatenating and shuffling ID and OOD data for the test set
    x_test = np.vstack([x_test, corrupted_x_train, corrupted_x_test])
    y_test = np.hstack([y_test, corrupted_y_train, corrupted_y_test])
    x_test, y_test = util.unison_shuffled_copies(x_test, y_test)
    print("Final testing set shape", x_test.shape, y_test.shape)

    #return np.asarray(corrupted_x_train), np.asarray(corrupted_y_train), np.asarray(corrupted_x_test), np.asarray(corrupted_y_test)
    return x_train, y_train, np.asarray(x_test), np.asarray(y_test)


def generate_corrupted_data(train, test, dataset_name, corruption_type, threat_type, severity, persist_data = False):
    if dataset_name == 'gtsrb':
        num_classes = 43
    elif dataset_name == 'cifar10':
        num_classes = 10

    ### rule for building corrupted dataset
    # train with ID training data
    # test with 100% of OOD data (modified training + modified test) + ID testing data
    success = False

    d = collections.OrderedDict()
    d['snow'] = corruptions.snow
    #d['frost'] = corruptions.frost #failure
    d['fog'] = corruptions.fog
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
    #d['motion_blur'] = corruptions.motion_blur #failure
    d['brightness'] = corruptions.brightness
    d['contrast'] = corruptions.contrast
    d['saturate'] = corruptions.saturate
    d['pixelate'] = corruptions.pixelate
    #d['JPEG'] = corruptions.jpeg_compression #failure    

    corruption = lambda clean_img: d[corruption_type](clean_img, severity)
    x_train, y_train, x_test, y_test = perform_corruptions(train, test, num_classes, corruption, severity)

    if persist_data:
        success = util.save_data(x_train, y_train, x_test, y_test, dataset_name, threat_type, corruption_type+"_severity_"+str(severity))
        print(dataset_name, corruption_type+"_severity_"+str(severity), success)
    
    return success


def map_btsc_gtsrb(labels_btsc):
    gtsrb_num_classes = 43 #number of classes for GTSRB dataset
    BTSC_to_GTSRB = {21:14, 0:22, 3:19, 4:20, 5:21, 10:25, 7:28, 11:26, 13:18,\
    16:24, 17:11, 19:13, 22:17, 28:15, 32:4, 34:35, 36:36, 61:12}
    
    final_labels = []

    for y_btsc in labels_btsc:
        try:
            final_labels.append(BTSC_to_GTSRB[y_btsc])
        except:
            final_labels.append(y_btsc+gtsrb_num_classes) 

    return np.array(final_labels)


def generate_novelty_data(dataset_names, save_experiments, parallel_execution, verbose, dir_path_write):
    ### rule for building novelty data
    # train with ID training data
    # test with 100% of OOD data (training + test) + ID testing data
    success = False

    #loading ID dataset
    ID_dataset = Dataset(dataset_names[0])
    x_train, y_train, x_ID_test, y_ID_test = ID_dataset.load_dataset()
    print("Training set shape", x_train.shape, y_train.shape)
    print("Testing set shape", x_ID_test.shape, y_ID_test.shape)

    #loading OOD dataset
    OOD_dataset = Dataset(dataset_names[1])
    x_OOD_train, y_OOD_train, x_OOD_test, y_OOD_test = OOD_dataset.load_dataset()

    ood_X = np.vstack([x_OOD_train, x_OOD_test])
    ood_y = np.concatenate((y_OOD_train, y_OOD_test), axis=None)

    if dataset_names[0] == 'gtsrb' and dataset_names[1] == 'btsc':
        # BTSC and GTSRB have 18 classes in common
        ood_y = map_btsc_gtsrb(ood_y)
    else:
        ood_y += ID_dataset.num_classes # avoiding same class numbers for the two datasets
    
    # resizing images to be equally if necessaire
    if ood_X.shape[1] != x_ID_test.shape[1]:
        ood_X = resize(ood_X, (len(ood_X), x_ID_test.shape[1], x_ID_test.shape[2], x_ID_test.shape[3]))

    # concatenating and shuffling ID and OOD datasets for test
    x_test = np.vstack([x_ID_test, ood_X])
    y_test = np.hstack([y_ID_test, ood_y])
    x_test, y_test = util.unison_shuffled_copies(x_test, y_test)
    print("Final testing set shape", x_test.shape, y_test.shape)

    if save_experiments:
        success = util.save_data_novelty(x_train, y_train, x_test, y_test, dataset_names, dir_path_write)

    return success