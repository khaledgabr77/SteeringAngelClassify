from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import cv2

_image = './test_1.jpg'
_model = './model.h5'
_labelbin = 'mlb.pickle'

image = cv2.imread(_image)

image = cv2.resize(image, (96, 96))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

print("[INFO] loading network...")
model = load_model(_model)
mlb = pickle.loads(open(_labelbin, "rb").read())

print("[INFO] classifying image...")
proba = model.predict(image)[0]
idxs = np.argsort(proba)[::-1][:2]

result = [None, -1]

for (label, p) in zip(mlb.classes_, proba):
    _p = float("{:.2f}".format(p * 100))
    if _p > result[1]:
        result = [int(label) - 3, _p]

print(result)
