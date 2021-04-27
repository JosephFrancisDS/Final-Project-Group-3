import os
import cv2
import numpy as np

#Encoding and Split data into Train/Test Sets
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#Tensorflow Keras CNN Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop

#Plot Images
import matplotlib.pyplot as plt


folder_dir = '/home/ubuntu/air_quality/breed_replication/flowers/'

data = []
label = []

SIZE = 128 #Crop the image to 128x128

for folder in os.listdir(folder_dir):
    for file in os.listdir(os.path.join(folder_dir, folder)):
        if file.endswith("jpg"):
            label.append(folder)
            img = cv2.imread(os.path.join(folder_dir, folder, file))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            im = cv2.resize(img_rgb, (SIZE,SIZE))
            data.append(im)
        else:
            continue

data_arr = np.array(data)
label_arr = np.array(label)

encoder = LabelEncoder()
y = encoder.fit_transform(label_arr)
y = to_categorical(y,5)
x = data_arr/255

SEED = 42
testSize=0.2
valSize=0.12

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=SEED, test_size=testSize, stratify=y)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, random_state=SEED, test_size=valSize, stratify=y_train)

np.save("/home/ubuntu/air_quality/breed_replication/pre_processing/numpy_files/x_train.npy", x_train);
np.save("/home/ubuntu/air_quality/breed_replication/pre_processing/numpy_files/y_train.npy", y_train);
np.save("/home/ubuntu/air_quality/breed_replication/pre_processing/numpy_files/x_val.npy", x_val);
np.save("/home/ubuntu/air_quality/breed_replication/pre_processing/numpy_files/y_val.npy", y_val);
np.save("/home/ubuntu/air_quality/breed_replication/pre_processing/numpy_files/x_test.npy", x_test);
np.save("/home/ubuntu/air_quality/breed_replication/pre_processing/numpy_files/y_test.npy", y_test);

