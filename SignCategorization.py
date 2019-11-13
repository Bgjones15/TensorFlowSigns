# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 16:43:38 2019

@author: sn0368hj
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

PATH = 'data'

train_dir = os.path.join(PATH, 'training')
validation_dir = os.path.join(PATH, 'validation')

categories = ['AddedLane','KeepRight','LaneEnds','Merge','PedestrianCrossing','School','SignalAhead','Stop','Yield']

training_dirs = []
validation_dirs = []
testing_dirs = []

total_train = 0;
total_val = 0;

for category in categories:
    training_dirs.append(os.path.join(os.path.join(PATH,'training'), category))
    validation_dirs.append(os.path.join(os.path.join(PATH,'validation'), category))
    testing_dirs.append(os.path.join(os.path.join(PATH,'testing'), category))
    
    print('total amount of training ',category,': ',len(os.listdir(os.path.join(os.path.join(PATH,'training'), category))))
    total_train += len(os.listdir(os.path.join(os.path.join(PATH,'training'), category)));
    print('total amount of validation ',category,': ',len(os.listdir(os.path.join(os.path.join(PATH,'validation'), category))))
    total_val += len(os.listdir(os.path.join(os.path.join(PATH,'validation'), category)));
    print('total amount of testing ',category,': ',len(os.listdir(os.path.join(os.path.join(PATH,'testing'), category))))
    


batch_size = 128
epochs = 20
IMG_HEIGHT = 37
IMG_WIDTH = 37

train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH))

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH))

sample_training_images, _ = next(train_data_gen)

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
plotImages(sample_training_images[:5])

model = Sequential([
    Flatten(input_shape=(IMG_HEIGHT,IMG_WIDTH, 3)),
    Dense(128,activation='relu'),
    Dense(9,activation='softmax')
])
    
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()



