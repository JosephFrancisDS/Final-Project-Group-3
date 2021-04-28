import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from glob import glob
from tqdm import tqdm
tqdm().pandas()
from all_models_no_aug_tools import *
from sklearn import preprocessing

from tensorflow.keras.applications import InceptionV3, Xception, VGG16, VGG19, ResNet50V2, ResNet101V2, ResNet152V2, \
                                        InceptionResNetV2, MobileNetV2, DenseNet121, DenseNet169, DenseNet201, \
                                        EfficientNetB0, EfficientNetB4, EfficientNetB7, NASNetLarge

train_path = '/home/ubuntu/air_quality/flowers-kaggle/processed_data/train/'
valid_path = '/home/ubuntu/air_quality/flowers-kaggle/processed_data/val/'
test_path = '/home/ubuntu/air_quality/flowers-kaggle/processed_data/test/'

# save the path to the train data
data_dir = pathlib.Path(train_path)
test_dir = pathlib.Path(test_path)

# count the number of images in the train
image_count = len(list(data_dir.glob('*/*.jpg')))
print(f"The number of images contained in the train set is {image_count}")

# Train dataset
labels = os.listdir(train_path)
x = []
y = []
for label in labels:
    x_, y_ = prepare_dataset(os.path.join(train_path, label), label)
    x.extend(x_)
    y.extend(y_)
x = np.array(x)
y = np.array(y)

# Test dataset
labels = os.listdir(test_path)
x_test = []
y_test = []
for label in labels:
    x_, y_ = prepare_dataset(os.path.join(test_path, label), label)
    x_test.extend(x_)
    y_test.extend(y_)
x_test = np.array(x_test)
y_test = np.array(y_test)

# create a validation set
from sklearn.model_selection import train_test_split
train_x, valid_x, y_train, y_valid = train_test_split(x, y, random_state=42, stratify=y, test_size=0.2)

encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(y_train)
valid_y = encoder.transform(y_valid)
test_y  = encoder.transform(y_test)

print(len(train_x), len(valid_x))
print(len(train_y), len(valid_y))


# initialize values to store the results
dict_hist = {}
df_results = pd.DataFrame()

num_classes = 5

model = tf.keras.Sequential([
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.5),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.3),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

es = tf.keras.callbacks.EarlyStopping(patience=3)
history = model.fit(
  train_x,train_y,
  validation_data=(valid_x, valid_y), callbacks=[es], batch_size=32,
   epochs=30
)
dict_hist["from_scratch"] = history

plot_learning_curve(history, "image_classif_from_scratch")

y_pred = model.predict(x_test)

metrics = metrics_evaluation(test_y, y_pred, dic_score= score_metrics)
df_results = pd.DataFrame(metrics.values(), index=metrics.keys()).T
df_results["Name"] = "no_Augmentation"
print(df_results)
df_results.to_csv("metrics_no_augmentation.csv", sep=";", index = False)
dict_hist["no_augmentation"] = history


list_pretrained = [ InceptionV3, VGG16, VGG19, Xception, ResNet50V2, ResNet101V2, ResNet152V2, InceptionResNetV2,
                    MobileNetV2, DenseNet121, DenseNet169, DenseNet201, EfficientNetB0, EfficientNetB4 ]

for model in list_pretrained:
    name = model.__name__
    
    x = pretrained_model_classification(train_x, train_y, valid_x, valid_y, x_test, test_y,
                                                  _model=model(weights='imagenet', include_top=False,
                                                               input_shape=(150, 150, 3)),
                                                  batch_size=32, epochs_num=30, patience=1, num_classes=5)
    # x.history
    # # convert the history.history dict to a pandas DataFrame:
    # import pandas as pd
    # hist_df = pd.DataFrame(x.history)
    #
    # # or save to csv:
    # hist_csv_file = 'history.csv'
    # with open(hist_csv_file, mode='w') as f:
    #     hist_df.to_csv(f)
    #
    # x.historyto_csv("results_pretrained.csv", sep=";", index=False)

