from time import sleep
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import pickle
import cv2

def predict(image):
    global model; global mlb
    image = cv2.resize(image, (96, 96))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    proba = model.predict(image)[0]
    idxs = np.argsort(proba)[::-1][:2]
    result = [None, -1]
    for (label, p) in zip(mlb.classes_, proba):
        _p = float("{:.2f}".format(p * 100))
        if _p > result[1]:
            result = [int(label) - 3, _p]
    return result[0]

CAMERA = "touha.mp4"
MODEL = './model.h5'
LABELBIN = 'mlb.pickle'

model = load_model(MODEL)
mlb = pickle.loads(open(LABELBIN, "rb").read())
cap = cv2.VideoCapture(CAMERA)

while cap.isOpened():
    ret, frame = cap.read()
    steer = predict(frame)
    print(steer)
    cv2.imshow('FRAME',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
