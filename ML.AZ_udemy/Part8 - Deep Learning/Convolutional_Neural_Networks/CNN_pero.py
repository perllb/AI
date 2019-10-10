#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 17:12:21 2019

@author: med-pvb
"""

## CNN

# Part 1 - building CNN

# import libs 
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initializing the CNN
classifier = Sequential()

# Step 1 - convolution
classifier.add(Convolution2D(filters = 32, kernel_size = (3, 3), padding = 'same', input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - flattening
classifier.add(Flatten())

# Step 4 - full connection
# hidden layer
classifier.add(Dense(units = 128, activation = 'relu')) # to first hidden layer (128 hidden nodes)
# output 
classifier.add(Dense(units = 1, activation = 'sigmoid')) # to output (sigmoid because of binary classification)  

# Compiling CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - fit the CNN to images
from keras.preprocessing.image import ImageDataGenerator
 
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
        steps_per_epoch=8000, # number of images in training set
        epochs=25, # number of epochs (training rounds)
        validation_data=test_set, # number of images in test set 
        validation_steps=2000) # 

### - fitting this CNN takes forever (~30 min per epoch)
### - accuracy should be around 0.71 for test set, 0.84 for training set
### - increase accuracy by making CNN deeper
### - can also be extra hidden layer, or increasing larger target size in ImageDataGenerator, in general feeding the network more info

# Increase accurasy by add an extra Conv Layer (can also add an extra hidden layer)
### This can increase test set accuracy, without overfitting
# Initializing the CNN
classifier = Sequential()

# Step 1 - convolution
# filters= #featuremaps applied 
classifier.add(Convolution2D(filters = 32, kernel_size = (3, 3), padding = 'same', input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(Convolution2D())
# Step 2 - pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding second Convolutional layer
# on the pooled feature maps from Conv layer 1
# do not need to specify input_shape, because Keras knows it from the previous steps
classifier.add(Convolution2D(filters = 32, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
# pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Step 3 - flattening
classifier.add(Flatten())

# Step 4 - full connection
# hidden layer
classifier.add(Dense(units = 128, activation = 'relu')) # to first hidden layer (128 hidden nodes)
# output 
classifier.add(Dense(units = 1, activation = 'sigmoid')) # to output (sigmoid because of binary classification)  

# Compiling CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - fit the CNN to images
from keras.preprocessing.image import ImageDataGenerator
 
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
        steps_per_epoch=8000, # number of images in training set
        epochs=25, # number of epochs (training rounds)
        validation_data=test_set, # number of images in test set 
        validation_steps=2000) # 

