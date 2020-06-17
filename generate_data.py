import util
import numpy as np
from threats import adv_attack
from threats import corruptions
from threats import anomalies
from threats import geometric_transformations


def generate_translated_data(train, test, dataset, drift_type, persist_data = False):
    success = persist_data
    (x_train, y_train) = train
    (x_test, y_test) = test

    
    if drift_type == 'cvt' or drift_type == 'cht' or drift_type == 'cdt':
        x_train, y_train = geometric_transformations.generate_data_translations(x_train, y_train, drift_type)
        x_test, y_test = geometric_transformations.generate_data_translations(x_test, y_test, drift_type)
    
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
    success = persist_data
    (x_train, y_train) = train
    (x_test, y_test) = test
    x_train = anomalies(x_train, anomaly_type)
    x_test = anomalies(x_test, anomaly_type)

    if persist_data:
        success = util.save_data(x_train, y_train, x_test, y_test, dataset, anomaly_type)
        
        return success


def generate_adversarial_data(data, dataset_name, ml_model, attack_type, persist_data = False):
    success = persist_data
    
    if attack_type == 'FGSM':
        x_train, y_train, x_test, y_test = adv_attack.perform_attacks(data, ml_model, dataset_name)

    if persist_data:
        success = util.save_data(x_train, y_train, x_test, y_test, dataset_name, attack_type)

    return success


def generate_corrupted_data(data, dataset_name, ml_model, corruption_type, persist_data = False):
    success = persist_data
    
    #if attack_type == 'FGSM':
    #    x_train, y_train, x_test, y_test = adv_attack.perform_attacks(data, ml_model, dataset_name)

    #if persist_data:
    #    success = util.save_data(x_train, y_train, x_test, y_test, dataset_name, attack_type)

    return success