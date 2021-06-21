# ==========================================
# The idea was implemented according to the method provided in the blog post by Keras
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# https://gist.github.com/fchollet/f35fbc80e066a49d65f1688a7e99f069
# ==========================================

# loading Libraries
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.engine.saving import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Conv2D, MaxPooling2D, Input, BatchNormalization, UpSampling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.models import Model
import numpy as np
import tensorflow as tf
from keras import backend as K, applications
import timeit

DrivePath = "D:/UNIVERSITY/Inf7370 - Apprentissage automatique/Projet/"

start = timeit.default_timer()

# ==========================================
# ===============GPU SETUP==================
# ==========================================
config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 4})
sess = tf.Session(config=config)
K.set_session(sess)

# ==========================================
# ================SETTINGS==================
# ==========================================
# mainDataPath = DrivePath + "Data/Development/"
mainDataPath = DrivePath + "Data/dogsvscats/Original/"

# Training Pictures Image Folder
trainPath = mainDataPath + "train"
# Validation Pictures Image Folder
validationPath = mainDataPath + "validate"
# test Pictures Image Folder
testPath = mainDataPath + "test"

# modelsPath
# with details in name: KerasAutoencoderModel.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5"
modelsPath = "Results/Model.hdf5"
summaryPath = "Results/Summary.txt"
logPath = "Results/Logs.csv"
lossPlotPath = "Results/Loss.png"
accuracyPlotPath = "Results/Accuracy.png"
totalTimePath = "Results/Time.txt"

# data loading setup
train_batch_size = 20000  # all 20000
val_batch_size = 5000  # all 5000

# images setup
image_scale = 128
image_channels = 1
images_color_mode = "grayscale"  # grayscale or rgb
image_shape = (image_scale, image_scale, image_channels)

# model fitting setup
fit_batch_size = 32
fit_epochs = 50

# ===============VGG ENCODER==================
# loading the encoder from VGG16 trained on imagenet
# without the fully connected layer
encoder = applications.VGG16(include_top=False, weights='imagenet')
encoder.summary()
# ============================================

# ===============OUR ENCODER==================
# # this section was disabled as it was
# # used to test our encoder, it was not used
# # as it did not yield good results

# autoencoder = load_model("Models/Autoencoder.hdf5")
# encoder = Model(autoencoder.layers[0].input, autoencoder.layers[27].output)
# ============================================


# ==========================================
# ==============LOADING DATA================
# ==========================================

# training_data_generator = ImageDataGenerator(rescale=1., featurewise_center=True, shear_range=0.1,
#                                              zoom_range=0.1,
#                                              horizontal_flip=True)  # (rescale=1./255)
# training_data_generator.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape(1, 1, 3)
#
# validation_data_generator = ImageDataGenerator(rescale=1., featurewise_center=True)  # (rescale=1./255)
# validation_data_generator.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape(1, 1, 3)

# Data augmentation for training data
training_data_generator = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True)

validation_data_generator = ImageDataGenerator(rescale=1. / 255)

# Training Data Generator
train_itr = training_generator = training_data_generator.flow_from_directory(
    trainPath,
    color_mode=images_color_mode,
    target_size=(image_scale, image_scale),
    batch_size=train_batch_size,
    class_mode="binary",
    shuffle=False)

# Validation Data Generator
val_itr = validation_generator = validation_data_generator.flow_from_directory(
    validationPath,
    color_mode=images_color_mode,
    target_size=(image_scale, image_scale),
    batch_size=val_batch_size,
    class_mode="binary",
    shuffle=False)

# loading Data into memory (No normalization)
(x_train, y_train) = train_itr.next()
(x_val, y_val) = val_itr.next()

# ==========================================
# ==============ENCODING DATA===============
# ==========================================

# in this section the images are encoded with the VGG Model

bottleneck_features_train = encoder.predict(x_train, 32)
np.save(open('bottleneck_features_train.npy', 'wb'), bottleneck_features_train)

bottleneck_features_validation = encoder.predict(x_val, 32)
np.save(open('bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)

train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
train_labels = np.array([0] * 10000 + [1] * 10000)

validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
validation_labels = np.array([0] * 2500 + [1] * 2500)

print(len(train_data))
print(len(train_labels))
print(len(validation_data))
print(len(validation_labels))

# ==========================================
# ==================MODEL===================
# ==========================================
# The fully connected layer to be trained
model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=RMSprop(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# ==========================================
# ================TRAINING==================
# ==========================================

# checkpoint to save the model on every validation
# accuracy change
modelcheckpoint = ModelCheckpoint(filepath=modelsPath,
                                  monitor='val_acc', verbose=1, save_best_only=True, mode='auto')

# a csv file to save the logs of the validation loss/accuracy
# and training loss and accuracy
csvLog = CSVLogger(logPath,
                   append=False,
                   separator=",")

# setting up early stopping to stop the fitting if a validation loss did not improve
# after 20 epochs
stopper = EarlyStopping(monitor='val_acc', patience=20, verbose=1, mode='auto')

# training the model
classifier = model.fit(train_data, train_labels,
                       epochs=50,
                       batch_size=32,
                       validation_data=(validation_data, validation_labels),
                       verbose=0,
                       callbacks=[stopper, modelcheckpoint, csvLog],
                       shuffle=True)

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
# Save Plot Image
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
# Save Plot Image
fig.savefig(accuracyPlotPath)
plt.show()
