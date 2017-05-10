import os
import csv
import random

from keras.models import Sequential, optimizers, load_model
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda, SpatialDropout2D
from keras.layers.convolutional import Convolution2D, Cropping2D, Conv2D
from keras.layers.pooling import MaxPooling2D,AveragePooling2D
from sklearn.utils import shuffle
from keras.callbacks import ModelCheckpoint

#path_file = 'C:/Users/linhnd16/Desktop/RecordData/driving_log.csv'
path_file = './data_thanh_ok/driving_log.csv'
samples = []
with open(path_file) as csvfile:
    reader = csv.DictReader(csvfile)
    for line in reader:
        steering = float(line["steering"])
        center = [line["center"], steering, False]
        left =   [line["left"],   steering + 0.12, False]
        right =  [line["right"],  steering - 0.12, False]
        flip_center = [line["center"], -steering, True]
        flip_left =   [line["left"],   - (steering + 0.12) , True]
        flip_right =  [line["right"],  - (steering - 0.12) , True]
        samples.append(center)
        samples.append(left)
        samples.append(right)
        samples.append(flip_center)
        samples.append(flip_left)
        samples.append(flip_right)

samples = shuffle(samples)
print("Total samples = ", len(samples))
        
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn
name_module_save = "./finalModule.h5"

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data_thanh_ok/IMG/' + batch_sample[0].split('\\')[-1] #batch_sample[0]#
                center_image = cv2.imread(name)
                center_angle = batch_sample[1]
                if batch_sample[2] == True:
                    center_image = cv2.flip(center_image,1)
                images.append(center_image)
                angles.append(center_angle)
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            X_train = X_train[...,::-1]
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size= 32)

def resize_images(img):
   import tensorflow as tf
   return tf.image.resize_images(img, (160, 200), tf.image.ResizeMethod.AREA)

if os.path.isfile(name_module_save):
   print()
   print("load model: ", name_module_save)
   print("continue training ...")
   checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=False,
                                 mode='auto') 
   
   model = load_model(name_module_save)
   adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.001, decay=0.0)
   model.compile(loss='mse', optimizer=adam)
   model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch = 5, callbacks=[checkpoint])
   model.save(name_module_save)
   print("name module save: ", name_module_save)
else:
   print()
   print("Start training ...")
   model = Sequential()
   model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=(160, 320, 3)))
   model.add(Cropping2D(cropping=((70,25), (0,0))))
   model.add(AveragePooling2D(pool_size = (2,5)))
   model.add(Convolution2D(32, 5, 5, subsample=(2, 2)))
   model.add(Activation('elu'))
   model.add(Convolution2D(48, 5, 5, subsample=(1, 1)))
   model.add(Activation('elu'))
   model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
   model.add(Activation('elu'))
   model.add(Convolution2D(96, 3, 3, subsample=(2, 2)))
   model.add(Activation('elu'))
   model.add(Convolution2D(128, 3, 3, subsample=(1, 1)))
   model.add(Activation('elu'))
   
   model.add(Flatten())
   model.add(Dense(500))
   model.add(Activation('elu'))
   model.add(Dropout(0.4))
   model.add(Dense(100))
   model.add(Activation('elu'))
   model.add(Dense(50))
   model.add(Activation('elu'))
   model.add(Dense(10))
   model.add(Activation('elu'))
   model.add(Dense(1))
   model.summary()
   
   checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto') 
   adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.001, decay=0.0)
   model.compile(loss='mse', optimizer=adam)
   model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=10, callbacks=[checkpoint])
   model.save(name_module_save)
   print("name module save: ", name_module_save)

