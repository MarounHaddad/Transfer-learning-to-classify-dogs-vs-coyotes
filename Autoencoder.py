# Importing Libraries
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image
from keras import backend as K, optimizers
import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras import metrics
import os
import timeit

# VARIABLES............

# Start Tracking Training Time
start = timeit.default_timer()

# ==========================================
# ===============GPU SETUP==================
# ==========================================

# configuring GPUs and CPUs
config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 4})
sess = tf.Session(config=config)
K.set_session(sess)

# ==========================================
# ==================SETTING=================
# ==========================================

# Training Pictures Image Folder
trainPath = "D:/UNIVERSITY/Inf7370 - Apprentissage automatique/Projet/Data/dogsvscats/Original/train"
# Testing Pictures Image Folder
valPath = "D:/UNIVERSITY/Inf7370 - Apprentissage automatique/Projet/Data/dogsvscats/Original/validate"
# Testing Pictures Image Folder
testPath = "D:/UNIVERSITY/Inf7370 - Apprentissage automatique/Projet/Data/dogsvscats/Original/test"

# modelsPath
# with details in name: KerasAutoencoderModel.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5"
modelsPath = "Models/Autoencoder.hdf5"

# data loading setup
train_batch_size = 20000  # all 20000
val_batch_size = 5000  # all 5000
test_batch_size = 20  # all 12484

# images setup
image_scale = 128
image_channels = 1
images_color_mode = "grayscale"  # grayscale or rgb
image_shape = (image_scale, image_scale, image_channels)

# model fitting setup
fit_batch_size = 32
fit_epochs = 20

# ==========================================
# ===================MODEL==================
# ==========================================

activation = "relu"
# encoder
input_img = Input(shape=image_shape)
x = Conv2D(32, (3, 3), padding='same')(input_img)
x = Activation(activation)(x)
x = BatchNormalization()(x)
x = Conv2D(32, (3, 3), padding='same')(x)
x = Activation(activation)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(64, (3, 3), padding='same')(x)
x = Activation(activation)(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), padding='same')(x)
x = Activation(activation)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(128, (3, 3), padding='same')(x)
x = Activation(activation)(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), padding='same')(x)
x = Activation(activation)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(256, (3, 3), padding='same')(x)
x = Activation(activation)(x)
x = BatchNormalization()(x)
x = Conv2D(256, (3, 3), padding='same')(x)
x = Activation(activation)(x)
x = BatchNormalization()(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# decoder
x = UpSampling2D((2, 2))(encoded)
x = Conv2D(256, (3, 3), padding='same')(x)
x = Activation(activation)(x)
x = BatchNormalization()(x)
x = Conv2D(256, (3, 3), padding='same')(x)
x = Activation(activation)(x)
x = BatchNormalization()(x)

x = UpSampling2D((2, 2))(x)
x = Conv2D(128, (3, 3), padding='same')(x)
x = Activation(activation)(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), padding='same')(x)
x = Activation(activation)(x)
x = BatchNormalization()(x)

x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), padding='same')(x)
x = Activation(activation)(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), padding='same')(x)
x = Activation(activation)(x)
x = BatchNormalization()(x)

x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), padding='same')(x)
x = Activation(activation)(x)
x = BatchNormalization()(x)
x = Conv2D(32, (3, 3), padding='same')(x)
x = Activation(activation)(x)
x = BatchNormalization()(x)

x = Conv2D(image_channels, (3, 3), padding='same')(x)
decoded = Activation('sigmoid')(x)

# define autoencoder Model
autoencoder = Model(input_img, decoded)
# display autoencoder summary
autoencoder.summary()

# define optimizer
sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
autoencoder.compile(optimizer=sgd, loss='mean_squared_error')

# ==========================================
# ===============LOADING DATA===============
# ==========================================
train_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

# training data loader
train_itr = train_generator = train_datagen.flow_from_directory(
    trainPath,
    color_mode=images_color_mode,
    target_size=(image_scale, image_scale),
    batch_size=train_batch_size,
    shuffle=True,
    class_mode='binary')

# validation data loader
val_itr = val_generator = val_datagen.flow_from_directory(
    valPath,
    color_mode=images_color_mode,
    target_size=(image_scale, image_scale),
    batch_size=val_batch_size,
    shuffle=True,
    class_mode='binary')

# testing data loader
test_itr = test_generator = test_datagen.flow_from_directory(
    testPath,
    color_mode=images_color_mode,
    target_size=(image_scale, image_scale),
    batch_size=test_batch_size,
    shuffle=False,
    class_mode='binary')

(x_train, _) = train_itr.next()
(x_val, _) = val_itr.next()
(x_test, _) = test_itr.next()

max_value = float(x_train.max())
x_train = x_train.astype('float32') / max_value
x_val = x_val.astype('float32') / max_value
x_test = x_test.astype('float32') / max_value

# ======================================
# ===============TRAINING===============
# ======================================
# setting up early stopping
stopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')

# setting up model checkpoint to save the model
# every time the validation loss decreases
modelcheckpoint = ModelCheckpoint(filepath=modelsPath,
                                  monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

# autoencoder.load_weights(modelsPath)

# train the model
autoencoder_train = autoencoder.fit(x_train, x_train,
                                    epochs=fit_epochs,
                                    batch_size=fit_batch_size,
                                    shuffle=True,
                                    verbose=0,
                                    validation_data=(x_val, x_val),
                                    callbacks=[stopper, modelcheckpoint])

# ================================================
# ===============DISPLAYING RESULTS===============
# ================================================

decoded_imgs = autoencoder.predict(x_test)

# encoder used to display the images of the bottleneck
encoder = Model(input_img, encoded)

stop = timeit.default_timer()
print('Total Time:', stop - start)

# Plot validation and training losses
print(autoencoder_train.history.keys())
plt.plot(autoencoder_train.history['loss'])
plt.plot(autoencoder_train.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()

# Plot original, encoded and predicted image
images_show_start = 0
images_show_stop = 10
images_show_number = images_show_stop - images_show_start + 1

plt.figure(figsize=(30, 5))
for i in range(images_show_start, images_show_stop):
    # original image
    ax = plt.subplot(3, images_show_number, i + 1)
    image = x_test[i, :, :, 0]
    image_reshaped = np.reshape(image, [1, image_scale, image_scale, 1])
    plt.imshow(image, cmap='gray')

    # label
    image_label = os.path.dirname(test_generator.filenames[i])
    plt.title(image_label)

    # encoded image
    ax = plt.subplot(3, images_show_number, i + 1 + 1 * images_show_number)
    image_encoded = encoder.predict(image_reshaped)
    # adjust shape if the network parameters are adjusted
    image_encoded_reshaped = np.reshape(image_encoded[0, :, :, 0], [16, 16])
    plt.imshow(image_encoded_reshaped, cmap='gray')

    # predicted image
    ax = plt.subplot(3, images_show_number, i + 1 + 2 * images_show_number)
    image_pred = autoencoder.predict(image_reshaped)
    image_pred_reshaped = np.reshape(image_pred, [image_scale, image_scale])
    plt.imshow(image_pred_reshaped, cmap='gray')
plt.show()
