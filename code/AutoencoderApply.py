# Loading Libraries
from PIL import Image
from keras import Model
from keras.models import load_model
from keras.preprocessing.image import img_to_array, ImageDataGenerator
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import os


def AutoencodeFolder(model_path,
                     images_path,
                     batch_size):
    """this function is used to encode the whole folder with all its images"""

    # images setup
    image_scale = 128
    encoded_image_scale = 16
    image_channels = 3
    images_color_mode = "rgb"

    # load Images
    datagen = ImageDataGenerator(rescale=1. / 255)

    images_itr = datagen.flow_from_directory(
        images_path,
        color_mode=images_color_mode,
        batch_size=batch_size,
        target_size=(image_scale, image_scale),
        shuffle=False)

    (images, _) = images_itr.next()

    # normalize images
    max_value = float(images.max())
    images = images.astype('float32') / max_value

    # Load the Autoencoder
    autoencoder = load_model(model_path)

    index = 0
    for image in images:
        image_reshaped = np.reshape(image, [1, image_scale, image_scale, image_channels])

        # taking the encoder part of the model
        encoder = Model(autoencoder.layers[0].input, autoencoder.layers[9].output)

        # encode image
        image_encoded = encoder.predict(image_reshaped)

        image_encoded_reshaped = np.reshape(image_encoded, [encoded_image_scale, encoded_image_scale, image_channels])

        # save encoded image
        scipy.misc.imsave(
            images_itr.filepaths[index].replace("train", "train_encoded_" + str(encoded_image_scale)
                                                ).replace("test", "test_encoded_" + str(encoded_image_scale)
                                                          ).replace("\\",
                                                                    "/"),
            image_encoded_reshaped)

        print("->" + images_itr.filenames[index])

        index += 1


# ================================================================
# ****disabled (only ran once to apply the encoding on images)****
# this method was not used as it gave poor results
# AutoencodeFolder("Models/ModelMSE16.hdf5",
#                  "D:/UNIVERSITY/Inf7370 - Apprentissage automatique/Projet/Data/live/dogsvscats/train",
#                  25000)
#
# AutoencodeFolder("Models/ModelMSE16.hdf5",
#                  "D:/UNIVERSITY/Inf7370 - Apprentissage automatique/Projet/Data/live/dogsvscats/test",
#                  12500)
# ================================================================

def AutoencodeSamples():
    """ this function is used to encode/decode only few samples and display the results"""
    # images setup
    test_batch_size = 20  # all 12484
    testPath = "D:/UNIVERSITY/Inf7370 - Apprentissage automatique/Projet/Data/dogsvscats/samples"
    image_scale = 128
    image_channels = 1
    images_color_mode = "grayscale"  # grayscale or rgb
    image_shape = (image_scale, image_scale, image_channels)

    # load data
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_itr = test_generator = test_datagen.flow_from_directory(
        testPath,
        color_mode=images_color_mode,
        target_size=(image_scale, image_scale),
        batch_size=test_batch_size,
        shuffle=False,
        class_mode='binary')
    (x_test, _) = test_itr.next()
    # normalize data
    max_value = float(x_test.max())
    x_test = x_test.astype('float32') / max_value

    # display the results
    images_show_start = 0
    images_show_stop = 5
    images_show_number = images_show_stop - images_show_start + 1
    plt.figure(figsize=(30, 5))

    # load model
    autoencoder = load_model("Models/Autoencoder.hdf5")

    # load encoder part
    encoder = Model(autoencoder.layers[0].input, autoencoder.layers[27].output)

    for i in range(images_show_start, images_show_stop):
        # original image
        ax = plt.subplot(3, images_show_number, i + 1)
        image = x_test[i, :, :, 0]
        image_reshaped = np.reshape(image, [1, image_scale, image_scale, 1])
        plt.imshow(image, cmap='gray')

        # label
        image_label = os.path.dirname(test_generator.filenames[i])
        plt.title(image_label)  # only OK if shuffle=false

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

    # show the images
    plt.show()


AutoencodeSamples()
