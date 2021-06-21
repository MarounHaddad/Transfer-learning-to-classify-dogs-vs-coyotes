# loading Libraries
import itertools
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import Model
from keras import backend as K
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.metrics import auc

# ==========================================
# ===============GPU SETUP==================
# ==========================================
config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 4})
sess = tf.Session(config=config)
K.set_session(sess)

# ==========================================
# ==================MODEL===================
# ==========================================
model_path = "Models/ClassifierCoyotes150.hdf5"
Classifier: Model = load_model(model_path)
# if set to true the images will pe preprocessed same as the VGG model
VGG = False
Title = "Coyote vs Dogs 150x150"

# ==========================================
# ================SETTINGS==================
# ==========================================
DrivePath = "D:/UNIVERSITY/Inf7370 - Apprentissage automatique/Projet/"
mainDataPath = DrivePath + "Data/dogsvscoyotes/Images/"
testPath = mainDataPath + "test"
batchsize = 500

image_scale = 150
images_color_mode = "rgb"  # grayscale or rgb

classes_titles = ["coyotes", "dogs"]

# ==========================================
# ============LOADING DATA==================
# ==========================================

# if VGG pre-process the images with the mean else rescale to  1/255
if VGG:
    test_data_generator = ImageDataGenerator(rescale=1., featurewise_center=True, shear_range=0.1,
                                             zoom_range=0.1,
                                             horizontal_flip=True)

    test_data_generator.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape(1, 1, 3)
else:
    test_data_generator = ImageDataGenerator(rescale=1. / 255)

# Test Data
test_generator = test_data_generator.flow_from_directory(
    testPath,
    target_size=(image_scale, image_scale),
    class_mode="binary",
    shuffle=False,
    batch_size=1,
    color_mode=images_color_mode)

test_itr = test_data_generator.flow_from_directory(
    testPath,
    target_size=(image_scale, image_scale),
    class_mode="binary",
    shuffle=False,
    batch_size=batchsize,
    color_mode=images_color_mode)

(x, y_true) = test_itr.next()

# Normalize Data
max_value = float(x.max())
x = x.astype('float32') / max_value

# ==========================================
# ============TESTING MODEL=================
# ==========================================

# the real classes of the loaded data
y_true = np.array([0] * 250 + [1] * 250)

# results of the model evaluation
test_eval = Classifier.evaluate_generator(test_generator, verbose=1)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

# list of predicted classes
predicted_classes = Classifier.predict_generator(test_generator, verbose=1)
predicted_classes_perc = np.round(predicted_classes.copy(), 4)
predicted_classes = np.round(predicted_classes)

# this list will hold the correctly identified photos
correct = []
for i in range(0, len(predicted_classes) - 1):
    if predicted_classes[i] == y_true[i]:
        correct.append(i)

print(len(correct))
print("Found %d correct labels" % len(correct))

# Show the top 5 correct photos
topcorrect = correct[:5]
index = 0

for i in topcorrect:
    plt.subplot(5, 5, index + 1)
    plt.axis("off")

    if images_color_mode == "grayscale":
        image = x[i, :, :, 0]
        plt.imshow(image, cmap="gray")
    else:
        image = x[i, :, :, :]
        plt.imshow(image)

    index += 1
    if predicted_classes[i] == 0:
        predicted = "Coyote"
        percentage = (1 - predicted_classes_perc[i]) * 100
    else:
        predicted = "Dog"
        percentage = (predicted_classes_perc[i]) * 100

    if y_true[i] == 0:
        actual = "Coyote"
    else:
        actual = "Dog"
    plt.title("[{}%{}]".format(  round(percentage[0]),predicted))
plt.show()

# this list will hold the falsely identified photos
incorrect = []
for i in range(0, len(predicted_classes) - 1):
    if predicted_classes[i] != y_true[i]:
        incorrect.append(i)

print(len(incorrect))
print("Found %d incorrect labels" % len(incorrect))

# print first falsely identified photos
topincorrect = incorrect[:5]
index = 0
for i in topincorrect:
    plt.subplot(5, 5, index + 1)
    plt.axis("off")

    if images_color_mode == "grayscale":
        image = x[i, :, :, 0]
        plt.imshow(image, cmap="gray")
    else:
        image = x[i, :, :, :]
        plt.imshow(image)


    index += 1
    predicted = ""
    actual = ""
    if predicted_classes[i] == 0:
        predicted = "Coyote"
        percentage = (1 - predicted_classes_perc[i]) * 100
    else:
        predicted = "Dog"
        percentage = (predicted_classes_perc[i]) * 100

    if y_true[i] == 0:
        actual = "Coyote"
    else:
        actual = "Dog"
    plt.title("[{}%{}]".format( round(percentage[0]), predicted))
plt.show()

# print lst falsely identified photos
topincorrect = incorrect[-5:]
index = 0
for i in topincorrect:
    plt.subplot(5, 5, index + 1)

    plt.axis("off")

    if images_color_mode == "grayscale":
        image = x[i, :, :, 0]
        plt.imshow(image, cmap="gray")
    else:
        image = x[i, :, :, :]
        plt.imshow(image)

    index += 1
    predicted = ""
    actual = ""
    if predicted_classes[i] == 0:
        predicted = "Coyote"
        percentage = (1 - predicted_classes_perc[i]) * 100
    else:
        predicted = "Dog"
        percentage = (predicted_classes_perc[i]) * 100

    if y_true[i] == 0:
        actual = "Coyote"
    else:
        actual = "Dog"
    plt.title("[{}%{}]".format(round(percentage[0]),predicted))
plt.show()

y_pred = predicted_classes > 0.5
print(confusion_matrix(y_true, y_pred))

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)
plt.figure()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    This Function is taken from the example given in the following reference
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    global Y_test
    global Y_predicted

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


plot_confusion_matrix(cnf_matrix, normalize=False,
                      classes=classes_titles,
                      title='Confusion matrix', cmap=plt.cm.Blues)
plt.show()

fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_true, y_pred)
auc_keras = auc(fpr_keras, tpr_keras)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='(area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
