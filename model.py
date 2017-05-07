print ("import library ...")
import os
import csv
import random

from keras.models import Sequential, optimizers
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda, SpatialDropout2D
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D,AveragePooling2D
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
import numpy as np

print("done.")
print(".....................")

print("read data file csv driving_log.csv ... ")
samples = []
with open('./data/driving_log.csv') as csvfile:
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
print("Total dataInput = ", len(samples))
print("done read data file csv.")
samples = shuffle(samples)
print(".....................")

print("create data_train (80% total data) and data_validation(20% total data).")
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print("Total data train: ", len(train_samples))
print("Total data valid: ", len(validation_samples))
print(".....................")

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = 'data/IMG/'+batch_sample[0].split('/')[-1]
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
validation_generator = generator(validation_samples, batch_size=32)



def resize_images(img):
	import tensorflow as tf
	return tf.image.resize_images(img, (160, 200), method = tf.image.ResizeMethod.BICUBIC)

model = Sequential()
model.add(Lambda(resize_images, input_shape=(160,320,3)))
model.add(Lambda(lambda x: x/255 - 0.5))
model.add(Cropping2D(cropping=((67,27), (0,0))))
model.add(Convolution2D(24, 5, 5, border_mode="same", subsample=(2,2), activation="elu"))
model.add(Dropout(0.8))
model.add(Convolution2D(36, 5, 5, border_mode="same", subsample=(2,2), activation="elu"))
model.add(Dropout(0.8))
model.add(Convolution2D(48, 5, 5, border_mode="valid", subsample=(2,2), activation="elu"))
model.add(Dropout(0.7))
model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="elu"))
model.add(Dropout(0.6))
model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="elu"))
model.add(Dropout(0.6))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100, activation="elu"))
model.add(Dense(50, activation="elu"))
model.add(Dense(10, activation="elu"))
model.add(Dropout(0.5))
model.add(Dense(1))

adam = optimizers.Adam(lr=0.001, epsilon=0.001)
model.compile(loss='mse', optimizer=adam)

model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch= 2)

model.save('./model1.h5')
