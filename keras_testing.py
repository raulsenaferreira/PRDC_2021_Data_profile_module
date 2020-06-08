import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.models import Model
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import dataset_loader as dl
import DataGenerator as dg



(x_train, y_train), (x_test, y_test) = dl.MNIST.load_data()

#if you need to do transformations on data
#x_train, x_test = util.reshaping_data(x_train, x_test, img_rows, img_cols, img_dim)
#x_train, x_valid, y_train, y_valid = train_test_split(x_train,y_train, test_size = validation_size)
#print('x_train shape:', x_train.shape)
#print(x_train.shape[0], 'train samples')
#print(x_test.shape[0], 'test samples')
#y_valid = keras.utils.to_categorical(y_valid, num_classes)

n_classes = 10
input_shape = (28, 28)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28, 28 , 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

train_generator = dg.DataGenerator(x_train, y_train, batch_size = 64,
                                dim = input_shape,
                                n_classes=10, 
                                to_fit=True, shuffle=True)
val_generator =  dg.DataGenerator(x_test, y_test, batch_size=64, 
                               dim = input_shape, 
                               n_classes= n_classes, 
                               to_fit=True, shuffle=True)

images, labels = next(train_generator)
print(images.shape)
print(labels.shape)
steps_per_epoch = len(train_generator)
validation_steps = len(val_generator)
model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=10,
        validation_data=val_generator,
        validation_steps=validation_steps)
