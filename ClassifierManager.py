# loading Libraries
import PIL
import numpy as np
from PIL import Image
from keras import Model
from keras.models import load_model


def ClassifyImage():
    model_path = "Models/ClassifierVGG200.hdf5"
    Classifier: Model = load_model(model_path)
    # if set to true the images will pe preprocessed same as the VGG model
    VGG = True

    imagepath = "D:/UNIVERSITY/Inf7370 - Apprentissage automatique/Projet/Data/dogsvscats/Original/test/cats/cat.4981.jpg"
    image_scale = 200
    images_color_mode = "rgb"  # grayscale or rgb
    dimensions = 3

    classes_titles = ["cats", "dogs"]

    # ==========================================
    # ============LOADING DATA==================
    # ==========================================

    # if VGG pre-process the images with the mean else rescale to  1/255

    size = image_scale, image_scale
    img = Image.open(imagepath)
    img = img.resize(size, PIL.Image.LANCZOS)
    x = np.array(img)
    max_value = float(x.max())
    x = x.astype('float32') / max_value

    if VGG:
        x = x * np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape(1, 1, dimensions)
    else:
        x = x * (1. / 255)

    x = x.reshape(1, image_scale, image_scale, dimensions)
    predicted_classes = Classifier.predict(x)
    predicted_classes_perc = np.round(predicted_classes.copy(), 4)
    predicted_classes = np.round(predicted_classes)

    if predicted_classes[0] == 0:
        predicted = "Cat"
        percentage = (1 - predicted_classes_perc[0]) * 100
    else:
        predicted = "Dog"
        percentage = (predicted_classes_perc[0]) * 100
    print("{}% {}".format(round(percentage[0]), predicted))
    return "{}% {}".format(round(percentage[0]), predicted)


ClassifyImage()
