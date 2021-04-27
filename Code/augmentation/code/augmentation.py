import csv
import numpy as np
import pandas as pd
import keras
from keras import backend as K
from keras.models import Sequential
from keras.callbacks import Callback
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.layers.core import Dense, Flatten, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.utils.vis_utils import plot_model
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt


train_path = '/home/ubuntu/air_quality/flowers-kaggle/processed_data/train/'
valid_path = '/home/ubuntu/air_quality/flowers-kaggle/processed_data/val/'
test_path = '/home/ubuntu/air_quality/flowers-kaggle/processed_data/test/'

# Image Datagenerator
train_batches = ImageDataGenerator().flow_from_directory(train_path,
                                                                   target_size=(224, 224),
                                                                   classes=['daisy','dandelion', 'rose', 'sunflower', 'tulip'],
                                                                   batch_size=32)

valid_batches = ImageDataGenerator().flow_from_directory(valid_path,
                                                        target_size=(224, 224),
                                                        classes=['daisy','dandelion', 'rose', 'sunflower', 'tulip'],
                                                        batch_size=32)

test_batches = ImageDataGenerator().flow_from_directory(test_path,
                                                        target_size=(224, 224),
                                                        classes=['daisy','dandelion', 'rose', 'sunflower', 'tulip'],
                                                        batch_size=32)

'''
# generates batches of normalized data.
train_batches = ImageDataGenerator(rescale = 1./255,
                                   rotation_range=10,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.15,
                                   zoom_range=0.1,
                                   channel_shift_range=10.,
                                   horizontal_flip=True,
                                   fill_mode='nearest').flow_from_directory(train_path,
                                                                   target_size=(224, 224),
                                                                   classes=['daisy','dandelion', 'rose', 'sunflower', 'tulip'],
                                                                   batch_size=32)

valid_batches = ImageDataGenerator(rescale = 1./255).flow_from_directory(valid_path,
                                                        target_size=(224, 224),
                                                        classes=['daisy','dandelion', 'rose', 'sunflower', 'tulip'],
                                                        batch_size=32)

test_batches = ImageDataGenerator(rescale = 1./255).flow_from_directory(test_path,
                                                        target_size=(224, 224),
                                                        classes=['daisy','dandelion', 'rose', 'sunflower', 'tulip'],
                                                        batch_size=32)


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dropout(rate = 1-0.5))
model.add(Dense(512, activation='relu'))
model.add(Dense(5, activation='softmax'))

model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit_generator(train_batches, steps_per_epoch=81, epochs=30, validation_data=valid_batches)
print(history.history)

with open('/home/ubuntu/air_quality/breed_replication/augmentation/results/augmentation.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
    w = csv.DictWriter(f, history.history.keys())
    w.writeheader()
    w.writerow(history.history)

# save model
model.summary()
model.save("/home/ubuntu/air_quality/breed_replication/augmentation/models/augmentation.h5")
plot_model(model, to_file='/home/ubuntu/air_quality/breed_replication/augmentation/images/augmentation_model.png',
                  show_shapes=True,
                  show_layer_names=True)

'''