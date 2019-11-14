# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 16:43:38 2019

@author: sn0368hj
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflowjs as tfjs
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from PIL import Image
from skimage import transform

from sklearn.metrics import classification_report, confusion_matrix

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

PATH = 'data'

train_dir = os.path.join(PATH, 'training')
validation_dir = os.path.join(PATH, 'validation')
test_dir = os.path.join(PATH, 'testing');

categories = ['AddedLane','KeepRight','LaneEnds','Merge','PedestrianCrossing','School','SignalAhead','Stop','Yield']

training_dirs = []
validation_dirs = []
testing_dirs = []

total_train = 0;
total_val = 0;
total_test = 0;

for category in categories:
    training_dirs.append(os.path.join(os.path.join(PATH,'training'), category))
    validation_dirs.append(os.path.join(os.path.join(PATH,'validation'), category))
    testing_dirs.append(os.path.join(os.path.join(PATH,'testing'), category))
    
    print('total amount of training ',category,': ',len(os.listdir(os.path.join(os.path.join(PATH,'training'), category))))
    total_train += len(os.listdir(os.path.join(os.path.join(PATH,'training'), category)));
    print('total amount of validation ',category,': ',len(os.listdir(os.path.join(os.path.join(PATH,'validation'), category))))
    total_val += len(os.listdir(os.path.join(os.path.join(PATH,'validation'), category)));
    print('total amount of testing ',category,': ',len(os.listdir(os.path.join(os.path.join(PATH,'testing'), category))))
    total_test += len(os.listdir(os.path.join(os.path.join(PATH,'testing'), category)));
    


batch_size = 100
epochs = 20
IMG_HEIGHT = 37
IMG_WIDTH = 37

train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data
testing_image_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH))

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH))

test_data_gen = testing_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=test_dir,
                                                           shuffle=False,
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
    Conv2D(37, (3, 3), activation='relu', input_shape=(37, 37, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(74, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(74, (3, 3), activation='relu'),
    Flatten(input_shape=(IMG_HEIGHT,IMG_WIDTH, 3)),
    Dense(74,activation='relu'),
    Dropout(0.5),
    Dense(9,activation='softmax')
])
    
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

callback = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=3)

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    callbacks=[callback],
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

tfjs.converters.save_keras_model(model,'model')

score = model.evaluate_generator(test_data_gen, total_test//batch_size)

test_data_gen.reset()

#Confution Matrix and Classification Report
Y_pred = model.predict_generator(test_data_gen, total_test // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_data_gen.classes, y_pred))
print('Classification Report')
target_names = list(test_data_gen.class_indices.keys())
print(classification_report(test_data_gen.classes, y_pred, target_names=target_names))



print(score)

np_image = Image.open('yield.jpg')
np_image = np.array(np_image).astype('float32')/255
np_image = transform.resize(np_image, (37,37,3))
np_image = np.expand_dims(np_image, axis=0)

print(model.predict(np_image))


acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']



