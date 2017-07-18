#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 17:26:29 2017

@author: joe
"""

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.layers import concatenate
from keras.optimizers import Adam
from keras.applications.inception_v3 import InceptionV3
import cv2
import numpy as np 


def combine_4_image(img1, img2, img3, img4):
    img1 = cv2.resize(img1, (112, 112))
    img2 = cv2.resize(img2, (112, 112))
    img3 = cv2.resize(img3, (112, 112))
    img4 = cv2.resize(img4, (112, 112))

    output = np.zeros((225, 225, 3))
    output[0:112, 0:112] = img1
    output[113:225, 0:112] = img2
    output[0:112, 113:225] = img3
    output[113:225, 113:225] = img4
    return output[0:224, 0:224]


def normalize(img):
    return img / 255


def numpy_minmax(x):
    xmin =  np.asarray([-38.03926279, -5.62667684, -27.69314801, -1.5502475])
    xmax = np.asarray([-0.03637159, -0.87553305, 23.91591131, 1.55643957])
    scaler = (x - xmin) / (xmax - xmin)
    scaler = scaler * .6
    return scaler + .2


def build_model(path_to_weights):
    images = Input(shape=(224, 224, 3), name='image_input')
    positions = Input(shape=(4, ), name='position_input')
    inception_v3_model = InceptionV3(weights='imagenet', include_top=False)
    for layer in inception_v3_model.layers:
        layer.trainable = False
    inception_v3_model.summary()
    inception_v3_model_conv = inception_v3_model(images)
    cnn = GlobalAveragePooling2D()(inception_v3_model_conv)
    final = concatenate([cnn, positions])
    final = Dense(1024, activation="relu")(final)
    #final = Dropout(0.5)(final)
    final = Dense(1024, activation="relu")(final)
    #final = Dropout(0.5)(final)
    final = Dense(8, activation='linear')(final)

    model = Model(input=[images, positions], output=final)
    model.load_weights(path_to_weights)
    adam = Adam(lr=0.001, clipvalue=1.5)
    model.compile(loss='mse', optimizer=adam)
    return model


class inference_runner():
    def __init__(self, path_to_weights):
        self.__trained_model = build_model(path_to_weights)

    def run_inference(self, noisy_positions, forward_last_img,
                      down_last_img, forward_current_img,
                      down_current_img):
        scaled_positions = np.asarray(numpy_minmax(noisy_positions))
        combined_img = combine_4_image(forward_last_img, down_last_img,
                                       forward_current_img,
                                       down_current_img)
        norm_img = normalize(combined_img)
        tensor_img = norm_img.reshape((1, 224, 224, 3))
        tensor_positions = scaled_positions.reshape(1, 4)
        return self.__trained_model.predict([tensor_img, tensor_positions])

# Tests
#img_path = "/home/joe/Dev/robot/auto pilot batch 1/{}"
#forward_last_img_path = img_path.format("autofwdB1.png")
#down_last_img_path = img_path.format("autofwdB1.png")
#forward_current_img_path = img_path.format("autofwdB2.png")
#down_current_img_path = img_path.format("autofwdB2.png")
#forward_last_img = cv2.imread('{}'.format(forward_last_img_path))
#down_last_img = cv2.imread('{}'.format(down_last_img_path))
#forward_current_img = cv2.imread('{}'.format(forward_current_img_path))
#down_current_img = cv2.imread('{}'.format(down_current_img_path))
#positions = [-4.9278298848, -1.045838346, -25.872343062,1.2358679902]
#model_path = '/home/joe/Dev/robot/models/InceptionV3_fc512_X2_pos-33.hdf5'
#position_predictor = inference_runner(model_path)
#output = position_predictor.run_inference(positions, forward_last_img,
#                      down_last_img, forward_current_img,
#                      down_current_img)
#
#print(output)