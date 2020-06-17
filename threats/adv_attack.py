import tensorflow as tf
#from datasets import Dataset
#from tensorflow.keras.datasets import mnist, cifar10, cifar100
import numpy as np
import random
import matplotlib.pyplot as plt

#based on this tutorial: 
#https://github.com/EvolvedSquid/tutorials/blob/master/adversarial-attacks-defenses/adversarial-tutorial.py

def perform_attacks(data, ml_model, dataset_name):

    # Function to create adversarial pattern
    def adversarial_pattern(image, label):#, model):
        image = tf.cast(image, tf.float32)
        # print(type(image))
        
        with tf.GradientTape() as tape:
            tape.watch(image)
            prediction = ml_model(image)
            loss = tf.keras.losses.MSE(label, prediction)
        
        gradient = tape.gradient(loss, image)
        signed_grad = tf.sign(gradient)
        return signed_grad

    # Adversarial data generator
    def generate_adversarials(batch_size, data, labels):
        while True:
            x = []
            y = []
            for N in range(batch_size):
                #N = random.randint(0, 100)

                label = labels[N]
                
                perturbations = adversarial_pattern(data[N].reshape((1, img_rows, img_cols, channels)), label).numpy()
                
                image = data[N]
                
                epsilon = 0.1
                adversarial = image + perturbations * epsilon
                
                x.append(adversarial)
                y.append(labels[N])
            
            
            x = np.asarray(x)
            x = x.reshape((batch_size, img_rows, img_cols, channels))
            y = np.asarray(y)
            
            yield  x, y
            
    (x_train, y_train) = data[0]
    (x_test, y_test) = data[1]
    dim = 1 if dataset_name =='mnist' else 3
    img_rows, img_cols, channels, num_classes = x_train.shape[1], x_train.shape[2], dim, len(np.unique(y_train))
    #(x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Cifar10
    # labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # MNIST
    #labels = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

    x_train = x_train / 255
    x_test = x_test / 255

    x_train = x_train.reshape((-1, img_rows, img_cols, channels))
    x_test = x_test.reshape((-1, img_rows, img_cols, channels))
    
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    print("Data shape", x_test.shape, y_test.shape, x_train.shape, y_train.shape)


    num_adversaries_train = len(y_train)
    num_adversaries_test = len(y_test)

    #train
    adversarials, correct_labels = next(generate_adversarials(num_adversaries_train, x_train, y_train))
    x_train_adv = []
    y_train_adv = []
    y_train_miss = [] # wrong label given by the classifier

    for adversarial, correct_label in zip(adversarials, correct_labels):    
        adversarial.reshape((1, img_rows, img_cols, channels))
        ind_adv = ml_model.predict(adversarial.reshape((1, img_rows, img_cols, channels))).argmax()
        adv_exple = y_train[ind_adv].argmax()
        y_true = correct_label.argmax()
        #print(adv_exple, y_true)
        if adv_exple != y_true:
            y_train_adv.append(y_true)
            y_train_miss.append(adv_exple)
            if channels == 1:
                #plt.imshow(adversarial.reshape(img_rows, img_cols))
                x_train_adv.append(adversarial.reshape(img_rows, img_cols))
            else:
                #plt.imshow(adversarial)
                x_train_adv.append(adversarial)
            #plt.show()
    x_train_adv = np.asarray(x_train_adv)
    print("x_train_adv.shape: ", x_train_adv.shape)

    #test
    adversarials, correct_labels = next(generate_adversarials(num_adversaries_test, x_test, y_test))
    x_test_adv = []
    y_test_adv = []
    y_test_miss = [] # wrong label given by the classifier

    for adversarial, correct_label in zip(adversarials, correct_labels):    
        adversarial.reshape((1, img_rows, img_cols, channels))
        ind_adv = ml_model.predict(adversarial.reshape((1, img_rows, img_cols, channels))).argmax()
        adv_exple = y_test[ind_adv].argmax()
        y_true = correct_label.argmax()
        #print(adv_exple, y_true)
        if adv_exple != y_true:
            y_test_adv.append(y_true)
            y_test_miss.append(adv_exple)
            if channels == 1:
                #plt.imshow(adversarial.reshape(img_rows, img_cols))
                x_test_adv.append(adversarial.reshape(img_rows, img_cols))
            else:
                #plt.imshow(adversarial)
                x_test_adv.append(adversarial)
            #plt.show()
    x_test_adv = np.asarray(x_test_adv)
    print("x_test_adv.shape: ", x_test_adv.shape)

    return x_train_adv, y_train_adv, y_train_miss, x_test_adv, y_test_adv, y_test_miss

    '''
    # Generate adversarial data
    x_adversarial_train, y_adversarial_train = next(generate_adversarials(20000))
    x_adversarial_test, y_adversarial_test = next(generate_adversarials(10000))

    # Assess base model on adversarial data
    print("Base accuracy on adversarial images:", model.evaluate(x=x_adversarial_test, y=y_adversarial_test, verbose=0))

    # Learn from adversarial data
    model.fit(x_adversarial_train, y_adversarial_train,
              batch_size=32,
              epochs=10,
              validation_data=(x_test, y_test))

    # Assess defended model on adversarial data
    print("Defended accuracy on adversarial images:", model.evaluate(x=x_adversarial_test, y=y_adversarial_test, verbose=0))

    # Assess defended model on regular data
    print("Defended accuracy on regular images:", model.evaluate(x=x_test, y=y_test, verbose=0))
    '''