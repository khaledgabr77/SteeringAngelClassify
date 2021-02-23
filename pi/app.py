import time
from picamera import PiCamera
from picamera.array import PiRGBArray
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
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

def steer(angle):
    print("Steering: {}".format(angle))

VIDEO = 0
MODEL = './model.h5'
LABELBIN = 'mlb.pickle'

model = load_model(MODEL)
mlb = pickle.loads(open(LABELBIN, "rb").read())

width = 320
height = 240
fps = 30
fcount = 0
camera = PiCamera()
camera.resolution = (width, height)
camera.framerate = fps
rawCapture = PiRGBArray(camera, size=(width, height))
time.sleep(1)

print("Capturing at {} fps!".format(fps))

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    if fcount % fps == 0:
        fcount = 0
        image = frame.array
        angle = predict(image)
        steer(angle)
    fcount += 1
