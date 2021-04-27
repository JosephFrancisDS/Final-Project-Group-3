from keras.utils.vis_utils import plot_model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model, load_model
from keras.models import Model
from keras import optimizers
from keras.utils.vis_utils import plot_model
import numpy as np
import csv
import numpy as np
import pandas as pd
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


## other
img_width, img_height = 224, 224
nb_train_samples = 81
nb_validation_samples = 200
top_epochs = 30
fit_epochs = 30
batch_size = 32
nb_classes = 5
nb_epoch = 30

# %%------------------------------------------------------------------------------------------------------------
# Inception Model
# %%------------------------------------------------------------------------------------------------------------
#build CNN
print("-----TRAINING XCEPTION MODEL------")

model_InceptionV3_conv = InceptionV3(weights='imagenet', include_top=False)

input = Input(shape=(img_width,img_height, 3),name = 'image_input')

output_vgg16_conv = model_InceptionV3_conv(input)

for layer in model_InceptionV3_conv.layers[:15]:
    layer.trainable = False

model_InceptionV3_conv.summary()

x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(5, activation='softmax', name='predictis')(x)

inception_model = Model(inputs=input, outputs=x)
inception_model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
                        metrics=['accuracy']
)

history = inception_model.fit_generator(train_batches,steps_per_epoch=nb_train_samples, epochs=nb_epoch,
                                        validation_data=valid_batches)
print(history.history)


with open('/home/ubuntu/air_quality/breed_replication/augmentation/results/inception.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
    w = csv.DictWriter(f, history.history.keys())
    w.writeheader()
    w.writerow(history.history)

# Evaluating the model
pd.DataFrame(history.history).plot(figsize=(16, 10))
plt.gca().set_ylim(0, 1)
plt.title('Model Evaluation')
plt.savefig('/home/ubuntu/air_quality/breed_replication/augmentation/images/inception_model_evaluation.png')
plt.show()

def plot_curves():
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(1, len(acc) + 1)

  plt.plot(epochs, acc, 'bo', label='Training acc')
  plt.plot(epochs, val_acc, 'b', label='Validation acc')
  plt.title('Training and validation accuracy')
  plt.savefig('/home/ubuntu/air_quality/breed_replication/augmentation/images/inception_train&val_accuracy.png')
  plt.legend()

  plt.figure()

  plt.plot(epochs, loss, 'bo', label='Training loss')
  plt.plot(epochs, val_loss, 'b', label='Validation loss')
  plt.title('Training and validation loss')
  plt.savefig('/home/ubuntu/air_quality/breed_replication/augmentation/images/inception_train&val_loss.png')
  plt.legend()

  plt.show()

plot_curves()



# save model
inception_model.summary()
inception_model.save("/home/ubuntu/air_quality/breed_replication/augmentation/models/inception_augmentation.h5")
plot_model(inception_model, to_file='/home/ubuntu/air_quality/breed_replication/augmentation/images/inception_aug.png',
                  show_shapes=True,
                  show_layer_names=True)

# restore the model and do some test set evaluation.
model = load_model('/home/ubuntu/air_quality/breed_replication/augmentation/models/inception_augmentation.h5')

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")

results = model.evaluate(test_batches, batch_size=128)
print("test loss, test acc:", results)

# Save as csv
np.savetxt("/home/ubuntu/air_quality/breed_replication/augmentation/results/inception_test_loss&accu.csv", results, delimiter=",")

# %%------------------------------------------------------------------------------------------------------------
# Xception Model
# %%------------------------------------------------------------------------------------------------------------
#build CNN
print("-----TRAINING XCEPTION MODEL------")

model_Xception_conv = Xception(weights='imagenet', include_top=False)

input = Input(shape=(img_width, img_height, 3),name = 'image_input')

output_vgg16_conv = model_Xception_conv(input)

for layer in model_Xception_conv.layers[:15]:
    layer.trainable = False
model_Xception_conv.summary()

x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(5, activation='softmax', name='predictis')(x)

xception_model = Model(inputs=input, outputs=x)

xception_model.summary()

#Image preprocessing and image augmentation with keras
xception_model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy']
)

history = xception_model.fit_generator(train_batches,steps_per_epoch=nb_train_samples, epochs=nb_epoch,
                                        validation_data=valid_batches)
print(history.history)

with open('/home/ubuntu/air_quality/breed_replication/augmentation/results/Xception_augmentation.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
    w = csv.DictWriter(f, history.history.keys())
    w.writeheader()
    w.writerow(history.history)

# save model
xception_model.summary()
xception_model.save("/home/ubuntu/air_quality/breed_replication/augmentation/models/Xception_augmentation.h5")
plot_model(xception_model, to_file='/home/ubuntu/air_quality/breed_replication/augmentation/images/Xception_augmentation_model.png',
                  show_shapes=True,
                  show_layer_names=True)

# Evaluating the model
pd.DataFrame(history.history).plot(figsize=(16, 10))
plt.gca().set_ylim(0, 1)
plt.title('Model Evaluation')
plt.savefig('/home/ubuntu/air_quality/breed_replication/augmentation/images/Xception_model_evaluation.png')
plt.show()

def plot_curves():
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(1, len(acc) + 1)

  plt.plot(epochs, acc, 'bo', label='Training acc')
  plt.plot(epochs, val_acc, 'b', label='Validation acc')
  plt.title('Training and validation accuracy')
  plt.savefig('/home/ubuntu/air_quality/breed_replication/augmentation/images/Xception_train&val_accuracy.png')
  plt.legend()

  plt.figure()

  plt.plot(epochs, loss, 'bo', label='Training loss')
  plt.plot(epochs, val_loss, 'b', label='Validation loss')
  plt.title('Training and validation loss')
  plt.savefig('/home/ubuntu/air_quality/breed_replication/augmentation/images/Xception_train&val_loss.png')
  plt.legend()

  plt.show()

plot_curves()

# restore the model and do some test set evaluation.
model = load_model('/home/ubuntu/air_quality/breed_replication/augmentation/models/Xception_augmentation.h5')

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")

results = model.evaluate(test_batches, batch_size=128)
print("test loss, test acc:", results)

# Save as csv
np.savetxt("/home/ubuntu/air_quality/breed_replication/augmentation/results/Xception_test_loss&accu.csv", results, delimiter=",")

print("-----XCEPTION MODEL iS READY------")

# %%------------------------------------------------------------------------------------------------------------
# RESNET50 Model
# %%------------------------------------------------------------------------------------------------------------
print("-----TRAINING: RESNET MODEL------")

model_ResNet50_conv = ResNet50(weights='imagenet', include_top=False)

input = Input(shape=(img_width,img_height, 3),name = 'image_input')

output_vgg16_conv = model_ResNet50_conv(input)

for layer in model_ResNet50_conv.layers[:15]:
    layer.trainable = False
model_ResNet50_conv.summary()

x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(5, activation='softmax', name='predictis')(x)

resnet50_model = Model(inputs=input, outputs=x)

resnet50_model.summary()

#Image preprocessing and image augmentation with keras
resnet50_model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy']
)

history = resnet50_model.fit_generator(train_batches,steps_per_epoch=nb_train_samples, epochs=nb_epoch,
                             validation_data=valid_batches)
print(history.history)

with open('/home/ubuntu/air_quality/breed_replication/augmentation/results/Resnet50_augmentation.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
    w = csv.DictWriter(f, history.history.keys())
    w.writeheader()
    w.writerow(history.history)

# save model
resnet50_model.summary()
resnet50_model.save("/home/ubuntu/air_quality/breed_replication/augmentation/models/resnet50_augmentation.h5")
plot_model(resnet50_model, to_file='/home/ubuntu/air_quality/breed_replication/augmentation/images/resnet50_augmentation_model.png',
                  show_shapes=True,
                  show_layer_names=True)

# Evaluating the model
pd.DataFrame(history.history).plot(figsize=(16, 10))
plt.gca().set_ylim(0, 1)
plt.title('Model Evaluation')
plt.savefig('/home/ubuntu/air_quality/breed_replication/augmentation/images/resnet50_model_evaluation.png')
plt.show()

def plot_curves():
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(1, len(acc) + 1)

  plt.plot(epochs, acc, 'bo', label='Training acc')
  plt.plot(epochs, val_acc, 'b', label='Validation acc')
  plt.title('Training and validation accuracy')
  plt.savefig('/home/ubuntu/air_quality/breed_replication/augmentation/images/resnet50_train&val_accuracy.png')
  plt.legend()

  plt.figure()

  plt.plot(epochs, loss, 'bo', label='Training loss')
  plt.plot(epochs, val_loss, 'b', label='Validation loss')
  plt.title('Training and validation loss')
  plt.savefig('/home/ubuntu/air_quality/breed_replication/augmentation/images/resnet50_train&val_loss.png')
  plt.legend()

  plt.show()

plot_curves()

# restore the model and do some test set evaluation.
model = load_model('/home/ubuntu/air_quality/breed_replication/augmentation/models/resnet50_augmentation.h5')

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")

results = model.evaluate(test_batches, batch_size=128)
print("test loss, test acc:", results)

# Save as csv
np.savetxt("/home/ubuntu/air_quality/breed_replication/augmentation/results/resnet50_test_loss&accu.csv", results, delimiter=",")

print("-----RESNET50 MODEL iS READY------")