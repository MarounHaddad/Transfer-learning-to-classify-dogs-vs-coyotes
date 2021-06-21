# ==========================================
# The idea was implemented according to the method provided in the blog post by Keras
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# https://gist.github.com/fchollet/7eb39b44eb9e16e59632d25fb3119975
# ==========================================

# loading libraries
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.engine.saving import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import RMSprop, sgd
from keras.layers import Conv2D, MaxPooling2D, Input, BatchNormalization, UpSampling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.models import Model
import numpy as np
import tensorflow as tf
from keras import backend as K, applications
import timeit

# start timer
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

DrivePath = "D:/UNIVERSITY/Inf7370 - Apprentissage automatique/Projet/"

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
train_batch_size = 100  # all 20000
val_batch_size = 100  # all 5000

# images setup
image_scale = 200
image_channels = 3
images_color_mode = "rgb"  # grayscale or rgb
image_shape = (image_scale, image_scale, image_channels)

# model fitting setup
fit_batch_size = 32
fit_epochs = 5

# ===============OUR ENCODER==================
# # this section was disabled as it was
# # used to test our encoder, it was not used
# # as it did not yield good results

# autoencoder = load_model("Models/Autoencoder.hdf5")
# basemodel = Model(autoencoder.layers[0].input, autoencoder.layers[27].output)
# ============================================

# ===============VGG ENCODER==================
# loading the encoder from VGG16 trained on imagenet
# without the fully connected layer
basemodel = applications.VGG16(weights='imagenet', include_top=False, input_shape=(200, 200, 3))
# ============================================

# ==========================================
# ==============LOADING DATA================
# ==========================================
training_data_generator = ImageDataGenerator(rescale=1., featurewise_center=True, shear_range=0.1,
                                             zoom_range=0.1,
                                             horizontal_flip=True)  # (rescale=1./255)

training_data_generator.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape(1, 1, 3)

validation_data_generator = ImageDataGenerator(rescale=1., featurewise_center=True)  # (rescale=1./255)
validation_data_generator.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape(1, 1, 3)

# Data augmentation for training data
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

# loading Data
(x_train, y_train) = train_itr.next()
(x_val, y_val) = val_itr.next()

# max_value = float(x_train.max())
# x_train = x_train.astype('float32') / max_value
# x_val = x_val.astype('float32') / max_value

# ==========================================
# ==================MODEL===================
# ==========================================

# load fully connected layer model
fullyconnected = load_model("Models/Model.hdf5")

x = Flatten(input_shape=fullyconnected.output.shape)(basemodel.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(basemodel.input, output)

# disable all the first layers of VGG and only fine tuning the
# weights of the fully connected layer
for layer in model.layers[:28]:
    layer.trainable = False

model.summary()

# update the weights of the fully connected layer with
# weights of the saved model
print(model.layers[28].name, fullyconnected.layers[0].name)
model.layers[28].set_weights(fullyconnected.layers[0].get_weights())
model.layers[29].set_weights(fullyconnected.layers[1].get_weights())
model.layers[30].set_weights(fullyconnected.layers[2].get_weights())
model.layers[31].set_weights(fullyconnected.layers[3].get_weights())

model.compile(optimizer=sgd(lr=0.0001, momentum=0.9),
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
stopper = EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='auto')

# Training the model
classifier = model.fit(x_train, y_train,
                       epochs=fit_epochs,
                       batch_size=fit_batch_size,
                       validation_data=(x_val, y_val),
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
