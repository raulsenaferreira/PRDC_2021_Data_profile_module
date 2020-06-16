import util
import numpy as np
import adv_attack
import corruptions


def generate_drift_data(train, test, dataset, drift_type, persist_data = False):
    success = persist_data
    (x_train, y_train) = train
    (x_test, y_test) = test

    # input image dimensions for doing some types of drift
    #img_rows, img_cols, img_dim = 28, 28, 1
    
    if drift_type == 'cvt' or drift_type == 'cht' or drift_type == 'cdt':
        x_train, y_train = util.generate_data_translations(x_train, y_train, drift_type)
        x_test, y_test = util.generate_data_translations(x_test, y_test, drift_type)
    
    elif drift_type=='rotated':
        x_train, y_train = util.rotating_data(x_train, y_train)
        x_test, y_test = util.rotating_data(x_test, y_test)
        #img_rows, img_cols, img_dim = 28, 28, 1
    
    elif drift_type=='back_round':
        (x_train, y_train), (x_test, y_test) = util.load_mnist_rand_back()

    elif drift_type=='moving':    
        (x_train, y_train), (x_test, y_test) = util.load_batches_moving_mnist()  

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
    
    if anomaly_type == 'pixel_trap':
        indices = np.random.choice(x_train.shape[0], 2, replace=False)
        x_train[indices] = 0
        indices = np.random.choice(x_test.shape[0], 2, replace=False)
        x_test[indices] = 0

    elif anomaly_type == 'row_add_logic':
        ind = int(x_train.shape[0]/2)-2
        x_train[ind+1] = x_train[ind]
        x_train[ind+2] = x_train[ind]
        x_train[ind+3] = x_train[ind]
        x_train[ind+4] = x_train[ind]

        ind = int(x_test.shape[0]/2)-2
        x_test[ind+1] = x_test[ind]
        x_test[ind+2] = x_test[ind]
        x_test[ind+3] = x_test[ind]
        x_test[ind+4] = x_test[ind]
    
    elif anomaly_type == 'shifted_pixel':
        max_shift = 5
        m,n = x_train.shape[0], x_train.shape[1]
        col_start = np.random.randint(0, max_shift, x_train.shape[0])
        idx = np.mod(col_start[:,None] + np.arange(n), n)
        x_train = x_train[np.arange(m)[:,None], idx]

        m,n = x_test.shape[0], x_test.shape[1]
        col_start = np.random.randint(0, max_shift, x_test.shape[0])
        idx = np.mod(col_start[:,None] + np.arange(n), n)
        x_test = x_test[np.arange(m)[:,None], idx]

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


def generate_corrupted_data(data, dataset_name, ml_model, attack_type, persist_data = False):
    success = persist_data
    
    if attack_type == 'FGSM':
        x_train, y_train, x_test, y_test = adv_attack.perform_attacks(data, ml_model, dataset_name)

    if persist_data:
        success = util.save_data(x_train, y_train, x_test, y_test, dataset_name, attack_type)

    return success