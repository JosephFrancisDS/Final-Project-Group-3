import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras.layers import (BatchNormalization, Conv2D, Conv2DTranspose, Dense,
                          Dropout, Flatten, Input, Reshape, UpSampling2D,
                          ZeroPadding2D)
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam

from PIL import Image, ImageDraw
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.advanced_activations import LeakyReLU

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Consistent results
np.random.seed(10)

# The dimension of z
noise_dim = 100

batch_size = 32
steps_per_epoch = 5000 # 60000 / 16
epochs = 10

save_path = '/home/ubuntu/air_quality/breed_replication/gan_augmentation/images/fc_gan/'

img_rows, img_cols, channels = 128, 128, 3

optimizer = Adam(0.0002, 0.5)

# Create path for saving images
if not os.path.isdir(save_path):
    os.mkdir(save_path)

# Load and process data
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
print(x_train.shape)

# Normalize to between -1 and 1
x_train = (x_train.astype(np.float32) - 127.5) / 127.5
x_train = x_train.reshape(-1, img_rows*img_cols*channels)

print(x_train.shape)


def create_generator():
    generator = Sequential()

    generator.add(Dense(256, input_dim=noise_dim))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(512))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(1024))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(img_rows * img_cols * channels, activation='tanh'))

    generator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return generator


def create_descriminator():
    discriminator = Sequential()

    discriminator.add(Dense(1024, input_dim=img_rows * img_cols * channels))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dense(512))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dense(1, activation='sigmoid'))

    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return discriminator


discriminator = create_descriminator()
generator = create_generator()

# Make the discriminator untrainable when we are training the generator.  This doesn't effect the discriminator by itself
discriminator.trainable = False

# Link the two models to create the GAN
gan_input = Input(shape=(noise_dim,))
fake_image = generator(gan_input)

gan_output = discriminator(fake_image)

gan = Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer=optimizer)


# Display images, and save them if the epoch number is specified
def show_images(noise, epoch=None):
    generated_images = generator.predict(noise)
    plt.figure(figsize=(10, 10))

    for i, image in enumerate(generated_images):
        plt.subplot(10, 10, i + 1)
        if channels == 1:
            plt.imshow(image.reshape((img_rows, img_cols)), cmap='plt.cm.binary')
        else:
            plt.imshow(image.reshape((img_rows, img_cols, channels)))
        plt.axis('off')

    plt.tight_layout()

    if epoch != None:
        plt.savefig(f'{save_path}/gan-images_epoch-{epoch}.png')

# Constant noise for viewing how the GAN progresses
static_noise = np.random.normal(0, 1, size=(100, noise_dim))
