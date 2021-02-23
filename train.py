from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from simplenet import SimpleNet
from imutils import paths
import numpy as np
import random
import pickle
import cv2
import os

_dataset = './dataset'
_model = './model.h5'
_labelbin = 'mlb.pickle'

EPOCHS = 200
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (96, 96, 3)

print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(_dataset)))
random.seed(42)
random.shuffle(imagePaths)

data = []
labels = []

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = img_to_array(image)
    data.append(image)

    l = label = imagePath.split(os.path.sep)[-2]
    labels.append(l)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {} images ({:.2f}MB)".format(
    len(imagePaths), data.nbytes / (1024 * 1000.0)))

print("[INFO] class labels:")

mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data,
    labels, test_size=0.2, random_state=42)

print("[INFO] compiling model...")

model = SimpleNet.build(
    width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
    depth=IMAGE_DIMS[2], classes=len(mlb.classes_),
    finalAct="sigmoid")

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

model.compile(loss="binary_crossentropy", optimizer=opt,
    metrics=["accuracy"])

print("[INFO] training network...")
H = model.fit(
    trainX,
    trainY,
    batch_size=BS,
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS, verbose=1)

print("[INFO] serializing network...")
model.save(_model)

print("[INFO] serializing label binarizer...")
f = open(_labelbin, "wb")
f.write(pickle.dumps(mlb))
f.close()
