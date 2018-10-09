import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
from scipy.io import loadmat

from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Flatten, Dropout, Activation, Lambda, Permute, Reshape
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D

from keras import backend as K
import cv2

K.set_image_data_format('channels_last')  # WARNING : important for images and tensors dimensions ordering


def convblock(cdim, nb, bits=3):
    L = []
    for k in range(1, bits + 1):
        convname = 'conv' + str(nb) + '_' + str(k)
        L.append(Convolution2D(cdim, kernel_size=(3, 3), padding='same', activation='relu', name=convname))
    L.append(MaxPooling2D((2, 2), strides=(2, 2)))
    return L


def vgg_model():
    withDropOut = True 

    model = Sequential()

    model.add(Permute((1, 2, 3), input_shape=(224, 224, 3))) 

    for l in convblock(64, 1, bits=2):
        model.add(l)

    for l in convblock(128, 2, bits=2):
        model.add(l)

    for l in convblock(256, 3, bits=3):
        model.add(l)

    for l in convblock(512, 4, bits=3):
        model.add(l)

    for l in convblock(512, 5, bits=3):
        model.add(l)

    model.add(Convolution2D(4096, kernel_size=(7, 7), activation='relu', name='fc6'))
    if withDropOut:
        model.add(Dropout(0.5))
    model.add(Convolution2D(4096, kernel_size=(1, 1), activation='relu', name='fc7'))
    if withDropOut:
        model.add(Dropout(0.5))
    model.add(Convolution2D(2622, kernel_size=(1, 1), activation='relu', name='fc8'))
    model.add(Flatten())
    model.add(Activation('softmax'))

    return model


facemodel = vgg_model()

facemodel = load_model('facemodel.h5')
facemodel.summary()  # visual inspection of model architecture

data = loadmat('misc/vgg-face.mat', matlab_compatible=False, struct_as_record=False)
l = data['layers']
description = data['meta'][0, 0].classes[0, 0].description

def pred(kmodel, crpimg):

    imarr = np.array(crpimg).astype(np.float32)

    imarr = np.expand_dims(imarr, axis=0)
    out = kmodel.predict(imarr)

    best_index = np.argmax(out, axis=1)[0]
    best_name = description[best_index, 0]
    print(best_index, best_name[0], out[0, best_index], [np.min(out), np.max(out)])


# ************************************************************************************************************* #

imagePath = 'ak_ar.jpg'

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)
faces = faceCascade.detectMultiScale(gray, 1.2, 5)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

plt.imshow(image)
plt.show()

image = Image.open(imagePath)

for nb in np.arange(0, len(faces), 1):
    print("Image number : ", nb + 1)
    (x, y, w, h) = faces[nb]

    center_x = x + w / 2
    center_y = y + h / 2

    b_dim = min(max(w, h) * 1.2, image.width, image.height) 
    box = (center_x - b_dim / 2, center_y - b_dim / 2, center_x + b_dim / 2, center_y + b_dim / 2)

    # Crop Image
    cropim = image.crop(box).resize((224, 224))
    plt.imshow(np.asarray(cropim))
    plt.show()

    pred(facemodel, cropim)

facemodel.save('facemodel.h5')
