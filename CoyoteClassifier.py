# ==========================================
# The model in this Classifier was mainly based on the model presented in this blog post:
# https://towardsdatascience.com/image-classifier-cats-vs-dogs-with-convolutional-neural-networks-cnns-and-google-colabs-4e9af21ae7a8
# The model architecture and settings were altered to fit our workflow
# ==========================================


# loading libraries
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.engine.saving import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.optimizers import sgd
from keras.layers import Conv2D, MaxPooling2D, Input, BatchNormalization, UpSampling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.models import Model
import numpy as np
import tensorflow as tf
from keras import backend as K
import timeit

# ==========================================
# ===============GPU SETUP==================
# ==========================================

# start timer
start = timeit.default_timer()

# configuring GPUs and CPUs
config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 4})
sess = tf.Session(config=config)
K.set_session(sess)

# ==========================================
# ================SETTINGS==================
# ==========================================
DrivePath = "D:/UNIVERSITY/Inf7370 - Apprentissage automatique/Projet/"

mainDataPath = DrivePath + "Data/dogsvscoyotes/Images/"

# Training Pictures Image Folder
trainPath = mainDataPath + "train"
# Validation Pictures Image Folder
validationPath = mainDataPath + "validate"
# test Pictures Image Folder
testPath = mainDataPath + "test"

# modelsPath
# with details in name: KerasAutoencoderModel.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5"
modelsPath = DrivePath + "Results/Model.hdf5"
summaryPath = DrivePath + "Results/Summary.txt"
logPath = DrivePath + "Results/Logs.csv"
lossPlotPath = DrivePath + "Results/Loss.png"
accuracyPlotPath = DrivePath + "Results/Accuracy.png"
totalTimePath = DrivePath + "Results/Time.txt"

# data loading setup
train_batch_size = 4750  # all 20000
val_batch_size = 532  # all 5000

# images setup
image_scale = 150
image_channels = 3
images_color_mode = "rgb"  # grayscale or rgb
image_shape = (image_scale, image_scale, image_channels)

# model fitting setup
fit_batch_size = 32
fit_epochs = 50

# ==========================================
# ==================MODEL===================
# ==========================================

# input image setup
input_img = Input(shape=image_shape)

# activation type used in the encoder
activation = "relu"


# Encoder
def encoder():
    x = Conv2D(32, (3, 3), padding='same')(input_img)
    x = Activation(activation)(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = Activation(activation)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation(activation)(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation(activation)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = Activation(activation)(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = Activation(activation)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = Activation(activation)(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = Activation(activation)(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    return encoded


# Fully Connected Layer
def fc(encoded):
    x = Flatten(input_shape=image_shape)(encoded)
    x = Dense(256)(x)
    x = Activation(activation)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    return x


model = Model(input_img, fc(encoder()))

# ==========================================================
# this Part is enabled when using the autoencoder for
# the encoder part and later for tuning
# ==========================================================
encode = load_model("Models/ClassifierColor150.hdf5")
for l1, l2 in zip(model.layers[:20], encode.layers[0:20]):
    print(l1.name,l2.name)
    l1.set_weights(l2.get_weights())

for layer in model.layers[:20]:
    layer.trainable = False
    # print(layer.name)
# ==========================================================
# ==========================================================

with open(summaryPath, "w") as fh:
    model.summary(print_fn=lambda line: fh.write(line + "\n"))

model.summary()

model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0001), metrics=['accuracy'])

# ==========================================
# ==============LOADING DATA================
# ==========================================

# Data augmentation for training data
training_data_generator = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True)

validation_data_generator = ImageDataGenerator(rescale=1. / 255)

# training data generator
train_itr = training_generator = training_data_generator.flow_from_directory(
    trainPath,
    color_mode=images_color_mode,
    target_size=(image_scale, image_scale),
    batch_size=train_batch_size,
    class_mode="binary")

# validation data generator
val_itr = validation_generator = validation_data_generator.flow_from_directory(
    validationPath,
    color_mode=images_color_mode,
    target_size=(image_scale, image_scale),
    batch_size=val_batch_size,
    class_mode="binary")

# print the classes indices [0 cat 1 Dog]
print(training_generator.class_indices)
print(validation_generator.class_indices)

# load data
(x_train, y_train) = train_itr.next()
(x_val, y_val) = val_itr.next()

# Data Normalization
max_value = float(x_train.max())
x_train = x_train.astype('float32') / max_value
x_val = x_val.astype('float32') / max_value

# ==========================================
# =================TRAINING=================
# ==========================================

modelcheckpoint = ModelCheckpoint(filepath=modelsPath,
                                  monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
csvLog = CSVLogger(logPath,
                   append=False,
                   separator=",")

# setup early stopping in case the validation accuracy did not
# improve after 10 epochs
stopper = EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='auto')

# Train model
classifier = model.fit(x_train, y_train,
                       epochs=fit_epochs,
                       batch_size=fit_batch_size,
                       validation_data=(x_val, y_val),
                       verbose=0,
                       callbacks=[stopper, modelcheckpoint, csvLog],
                       shuffle=True)

# stop timer and display and save results
stop = timeit.default_timer()
print('Total Time:', stop - start)
with open(totalTimePath, 'w') as f:
    f.write(str(stop - start))

# ==========================================
# ============DISPLAY RESULTS===============
# ==========================================

# Plot loss over epochs
print(classifier.history.keys())
plt.plot(classifier.history['loss'])
plt.plot(classifier.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
fig = plt.gcf()
fig.savefig(lossPlotPath)
plt.show()

# Plot accuracy over epochs
print(classifier.history.keys())
plt.plot(classifier.history['acc'])
plt.plot(classifier.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
fig = plt.gcf()
fig.savefig(accuracyPlotPath)
plt.show()
