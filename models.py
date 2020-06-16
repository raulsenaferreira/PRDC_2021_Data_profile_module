import util
import generate_data as gd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, Dense, Flatten, Activation
import tensorflow_datasets as tfds
from tensorflow.keras.datasets import mnist, cifar10



class Models():
    def __init__(self, dataset = 'mnist'):        
        self.dataset = dataset
        self.dim = 28*28
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.load_data(self.dataset)

    def load_data(self, dataset):
        if dataset == 'mnist':
            self.num_classes = 10
            return mnist.load_data()
        elif dataset == 'emnist':
            self.num_classes = 47
            return tfds.as_numpy(tfds.load('emnist/balanced', split=['train', 'test'], batch_size=-1, as_supervised=True)) #util.load_balanced_emnist()
            #return (x_train.T, y_train), (x_test.T, y_test) #correcting emnist images
        elif dataset=='ardis':
            self.num_classes = 10
            return util.load_ardis_mnist()
        elif dataset=='kmnist':
            self.num_classes = 10
            return tfds.as_numpy(tfds.load('kmnist', split=['train', 'test'], batch_size=-1, as_supervised=True))
            

    def load_drift_data(self, drift_type, persist_data=False): 
        
        if drift_type == 'cvt' or drift_type == 'cht' or drift_type == 'cdt':
            num_train = len(self.y_train)*11
            num_test = len(self.y_test)*11
        (self.x_train, self.y_train), (self.x_test, self.y_test) = util.load_drift_mnist(
            self.dataset, drift_type, num_train, num_test, self.dim)
        
        return (self.x_train, self.y_train), (self.x_test, self.y_test)


    # Create model
    def create_mnist_model(self, x_train, y_train, x_test, y_test, model_file):
        img_rows, img_cols, channels, num_classes = 28, 28, 1, 10
        #preprocessing
        x_train = x_train / 255
        x_test = x_test / 255

        x_train = x_train.reshape((-1, img_rows, img_cols, channels))
        x_test = x_test.reshape((-1, img_rows, img_cols, channels))

        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)
        print(x_train.shape)
        print(x_test.shape)

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), strides=(3, 3), padding='same', activation='relu', 
            input_shape=(img_rows, img_cols, channels)))
        model.add(Conv2D(64, kernel_size=(3, 3), strides=(3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, kernel_size=(3, 3), strides=(3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(32))
        model.add(Dropout(0.2))
        model.add(Dense(32))
        model.add(Dropout(0.2))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        
        model.summary()

        model.fit(x_train, y_train,
                  batch_size=32,
                  epochs=20,
                  validation_data=(x_test, y_test))
        # Assess base model accuracy on regular images
        print("Base accuracy on regular images:", model.evaluate(x=x_test, y=y_test, verbose=0))

        model.save(model_file)

        return model


    def create_gtsrb_model(self, x_train, y_train, x_test, y_test, model_file):
        img_rows, img_cols, channels, num_classes = 28, 28, 3, 43
        #preprocessing
        x_train = x_train / 255
        x_test = x_test / 255

        #x_train = x_train.reshape((-1, img_rows, img_cols, channels))
        #x_test = x_test.reshape((-1, img_rows, img_cols, channels))

        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

        print(x_train.shape)
        print(x_test.shape)

        model = Sequential()
        model.add(tf.keras.layers.Conv2D(32, (5, 5), padding='same', 
            activation='relu', input_shape=(img_rows, img_cols, channels)))
        model.add(tf.keras.layers.BatchNormalization(axis=-1))      
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.2))
                
        model.add(tf.keras.layers.Conv2D(64, (5, 5), padding='same', 
                                         activation='relu'))
        model.add(tf.keras.layers.BatchNormalization(axis=-1))
        model.add(tf.keras.layers.Conv2D(128, (5, 5), padding='same', 
                                         activation='relu'))
        model.add(tf.keras.layers.BatchNormalization(axis=-1))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.2))
         
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(512, activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.4))
         
        model.add(tf.keras.layers.Dense(43, activation='softmax'))

        #Compilation of the model

        optimizer = tf.keras.optimizers.Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, 
                      metrics=['accuracy'])

        history = model.fit(x_train, y_train, 
                           validation_data=(x_test, y_test),
                           epochs=10)
        
        model.summary()

        #model.fit(x_train, y_train,
        #          batch_size=32,
        #          epochs=20,
        #          validation_data=(x_test, y_test))
        # Assess base model accuracy on regular images
        print("Base accuracy on regular images:", model.evaluate(x=x_test, y=y_test, verbose=0))

        model.save(model_file)

        return model


    def optimizing_model(self, trainX, trainY, testX, testY, model):
        datagen = keras.preprocessing.image.ImageDataGenerator(
            width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
        # prepare iterator
        it_train = datagen.flow(trainX, trainY, batch_size=64)
        # fit model
        steps = int(trainX.shape[0] / 64)
        history = model.fit_generator(it_train, steps_per_epoch=steps, epochs=25, validation_data=(testX, testY), verbose=0)
        return model


    def optimizing_model_2(self, train_X, trainY, test_X, testY, model):
        learning_rate = 0.001
        epochs = 20
        batch_size = 64
        class_totals = trainY.sum(axis=0)
        class_weight = class_totals.max() / class_totals
        print("class_weight", class_weight)

        data_aug = keras.preprocessing.image.ImageDataGenerator(rotation_range=10, zoom_range=0.15, 
            width_shift_range=0.1, height_shift_range=0.1, shear_range=0.15, horizontal_flip=False, vertical_flip=False)
        optimizer = keras.optimizers.Adam(lr=learning_rate, decay=learning_rate / (epochs))
        # fit model
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        fit = model.fit_generator(
            data_aug.flow(train_X, trainY, batch_size=batch_size), 
            epochs=epochs,
            validation_data=(test_X, testY),
            #class_weight=class_weight,
            verbose=1)
        return model


    def create_cifar10_model(self, x_train, y_train, x_test, y_test, model_file):
        img_rows, img_cols, channels, num_classes = 32, 32, 3, 10
        #preprocessing
        x_train = x_train / 255
        x_test = x_test / 255

        x_train = x_train.reshape((-1, img_rows, img_cols, channels))
        x_test = x_test.reshape((-1, img_rows, img_cols, channels))

        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

        print(x_train.shape)
        print(x_test.shape)

        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.3))
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))
        # compile model
        opt = keras.optimizers.SGD(lr=0.001, momentum=0.9)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        
        
        model.summary()

        # with augmentation
        #model = self.optimizing_model(x_train, y_train, x_test, y_test, model)
        
        # without augmentation
        model.fit(x_train, y_train,
                  batch_size=64,
                  epochs=25,
                  validation_data=(x_test, y_test))
        
        # Assess base model accuracy on regular images
        print("Base accuracy on regular images:", model.evaluate(x=x_test, y=y_test, verbose=0))

        model.save(model_file)
        
        return model


    def create_cifar10_model_2(self, x_train, y_train, x_test, y_test, model_file):
        img_rows, img_cols, channels, num_classes = 32, 32, 3, 10
        #preprocessing
        x_train = x_train / 255
        x_test = x_test / 255

        x_train = x_train.reshape((-1, img_rows, img_cols, channels))
        x_test = x_test.reshape((-1, img_rows, img_cols, channels))

        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

        print(x_train.shape)
        print(x_test.shape)

        model = Sequential()
        model.add(Conv2D(8, (5, 5), input_shape=(img_rows, img_cols, channels), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(16, (3, 3), activation="relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(16, (3, 3), activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
 
        model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(512, activation="relu"))
        model.add(Dense(num_classes, activation="softmax"))

        # with augmentation
       # model = self.optimizing_model_2(x_train, y_train, x_test, y_test, model)
        
        # without augmentation
        model.fit(x_train, y_train,
                  batch_size=64,
                  epochs=25,
                  validation_data=(x_test, y_test))
        
        # Assess base model accuracy on regular images
        print("Base accuracy on regular images:", model.evaluate(x=x_test, y=y_test, verbose=0))

        model.save(model_file)
        
        return model