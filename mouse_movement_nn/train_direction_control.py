#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 10:45:11 2017

@author: joe
"""

import numpy as np
import pandas as pd
from pandas import *
import random

from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from keras.layers import concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.applications.inception_v3 import InceptionV3

import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


width_height = 224
BATCH_SIZE = 144
np.random.seed(42)

def load_image(filename1, filename2, filename3, filename4):
    img1 = cv2.imread('{}'.format(filename1))
    img2 = cv2.imread('{}'.format(filename2))
    img3 = cv2.imread('{}'.format(filename3))
    img4 = cv2.imread('{}'.format(filename4))
    img1 = cv2.resize(img1, (112, 112))
    img2 = cv2.resize(img2, (112, 112))
    img3 = cv2.resize(img3, (112, 112))
    img4 = cv2.resize(img4, (112, 112))

    output = np.zeros((225, 225, 3))
    output[0:112,0:112] = img1
    output[113:225,0:112] = img2
    output[0:112,113:225] = img3
    output[113:225,113:225] = img4
    return output[0:224,0:224]

def mean_normalize(img):
    return (img - img.mean()) / (img.max() - img.min())

def normalize(img):
    return img / 255

def numpy_minmax(x):
    xmin =  np.asarray([-38.03926279, -5.62667684, -27.69314801, -1.5502475])
    xmax = np.asarray([-0.03637159, -0.87553305, 23.91591131, 1.55643957])
    scaler = (x - xmin) / (xmax - xmin)
    scaler = scaler * .6
    return scaler + .2

def get_x_data(images_df, positions_df, num_rows, directory):
    """
    To Do redefine inputs
    """
    x_data = []
    for i in range(0, num_rows-1):
        scaled = numpy_minmax(np.asarray([positions_df[i][0],positions_df[i][1],
                              positions_df[i][2],positions_df[i][3]]))
        dict_images = {'filename1': directory + '/' + images_df[i][0], 'filename2': directory + '/' + images_df[i][1],
                       'filename3': directory + '/' + images_df[i+1][0], 'filename4': directory + '/' + images_df[i+1][1],
                       'position.x': scaled[0],
                       'position.y': scaled[1],
                       'position.z': scaled[2],
                       'rotation.y': scaled[3],
                       'pathmarker1Points': positions_df[i][4],
                       'buoy1Points': positions_df[i][5],
                       'buoy2Points': positions_df[i][6],
                       'buoy3Points': positions_df[i][7],
                       'pathmarker2Points': positions_df[i][8],
                       'maneuverPoints': positions_df[i][9],
                       'pathmarker3Points': positions_df[i][10],
                       'pathmarker4Points': positions_df[i][11],
                       'pathmarker5Points': positions_df[i][12],
                       'octagon1Points': positions_df[i][13],
                       'pathmarker6Points': positions_df[i][14],
                       'pathmarker7Points': positions_df[i][15],
                       'octagon2Points': positions_df[i][16]
                      }
        x_data.append(dict_images)
    df_x_data = pd.DataFrame(x_data, columns=['filename1', 'filename2', 'filename3', 'filename4',
                                                'position.x', 'position.y', 'position.z', 'rotation.y',
                                              'pathmarker1Points', 'buoy1Points', 'buoy2Points', 'buoy3Points',
                                              'pathmarker2Points', 'maneuverPoints', 'pathmarker3Points',
                                             'pathmarker4Points', 'pathmarker5Points', 'octagon1Points',
                                              'pathmarker6Points','pathmarker7Points','octagon2Points'])
    return df_x_data

def convert_y_wasd(input_value):
    if input_value == False:

        return 0
    if input_value == True:
        return 1
    return input_value

def get_y_data(positions_df, num_rows):
    """
    To do redefine inputs
    """
    y_data = []
    for i in range(0, num_rows-1):
        dict_images = {'mousex': positions_df[i+1][0], 'mousey': positions_df[i+1][1],
                       'W': convert_y_wasd(positions_df[i+1][0]),
                       'A': convert_y_wasd(positions_df[i+1][1]),
                       'S': convert_y_wasd(positions_df[i+1][2]),
                       'D': convert_y_wasd(positions_df[i+1][3])}
        y_data.append(dict_images)
    df_y_data = pd.DataFrame(y_data, columns=['mousex','mousey','W','A','S','D'])
    return df_y_data

def get_y_data_frame(data_frame):
    data_frame = data_frame.fillna(0)
    num_rows = len(data_frame.index)
    positions_df = data_frame[['mousex','mousey','W','A','S','D']].values
    return get_y_data(positions_df, num_rows)


def get_x_data_frame(data_frame, directory):
    data_frame = data_frame.fillna(0)
    num_rows = len(data_frame.index)
    images_df = data_frame[['filename1','filename2']].values
    non_image_df = data_frame[['position.x','position.y','position.z',
                               'rotation.y','pathmarker1Points', 'buoy1Points',
                               'buoy2Points', 'buoy3Points',
                               'pathmarker2Points', 'maneuverPoints',
                               'pathmarker3Points', 'pathmarker4Points',
                               'pathmarker5Points', 'octagon1Points',
                               'pathmarker6Points','pathmarker7Points',
                               'octagon2Points']].values
    return get_x_data(images_df, non_image_df, num_rows, directory)

def get_training_df(bat_num, csv_file_name):
    dir_path = '/home/joe/Dev/robot_direction_control/train_bat_'
    directory = '{0}{1}'.format(dir_path, str(bat_num))
    csv_file = '{0}{1}/{2}'.format(dir_path, str(bat_num),csv_file_name)
    return [pd.read_csv(csv_file), directory]
td1 = get_training_df(1, 'maneuversave.csv')
td2 = get_training_df(2, 'maneuversave.csv')
td3 = get_training_df(3, 'maneuversave.csv')
td4 = get_training_df(4, 'maneuversave.csv')
td5 = get_training_df(5, 'maneuversave.csv')
td6 = get_training_df(6, 'maneuversave.csv')
td7 = get_training_df(7, 'maneuversave.csv')
td8 = get_training_df(8, 'labels.csv')
td9 = get_training_df(9, 'labels.csv')
td10 = get_training_df(10, 'labels.csv')
td11 = get_training_df(11, 'labels.csv')
td12 = get_training_df(12, 'labels.csv')
td13 = get_training_df(13, 'labels.csv')
td14 = get_training_df(14, 'labels.csv')
td15 = get_training_df(15, 'labels.csv')
td16 = get_training_df(16, 'labels.csv')
td17 = get_training_df(17, 'labels.csv')
td18 = get_training_df(18, 'labels.csv')
td19 = get_training_df(19, 'labels.csv')
td20 = get_training_df(20, 'labels.csv')


X = get_x_data_frame(td1[0], td1[1])
X = X.append(get_x_data_frame(td2[0], td2[1]), ignore_index=True)
X = X.append(get_x_data_frame(td3[0], td3[1]), ignore_index=True)
X = X.append(get_x_data_frame(td4[0], td4[1]), ignore_index=True)
X = X.append(get_x_data_frame(td5[0], td5[1]), ignore_index=True)
X = X.append(get_x_data_frame(td6[0], td6[1]), ignore_index=True)
X = X.append(get_x_data_frame(td7[0], td7[1]), ignore_index=True)
X = X.append(get_x_data_frame(td8[0], td8[1]), ignore_index=True)
X = X.append(get_x_data_frame(td9[0], td9[1]), ignore_index=True)
X = X.append(get_x_data_frame(td10[0], td10[1]), ignore_index=True)
X = X.append(get_x_data_frame(td11[0], td11[1]), ignore_index=True)
X = X.append(get_x_data_frame(td12[0], td12[1]), ignore_index=True)
X = X.append(get_x_data_frame(td13[0], td13[1]), ignore_index=True)
X = X.append(get_x_data_frame(td14[0], td14[1]), ignore_index=True)
X = X.append(get_x_data_frame(td15[0], td15[1]), ignore_index=True)
X = X.append(get_x_data_frame(td16[0], td16[1]), ignore_index=True)
X = X.append(get_x_data_frame(td17[0], td17[1]), ignore_index=True)
X = X.append(get_x_data_frame(td18[0], td18[1]), ignore_index=True)
X = X.append(get_x_data_frame(td19[0], td19[1]), ignore_index=True)
X = X.append(get_x_data_frame(td20[0], td20[1]), ignore_index=True)
y = get_y_data_frame(td1[0])
y = y.append(get_y_data_frame(td2[0]), ignore_index=True)
y = y.append(get_y_data_frame(td3[0]), ignore_index=True)
y = y.append(get_y_data_frame(td4[0]), ignore_index=True)
y = y.append(get_y_data_frame(td5[0]), ignore_index=True)
y = y.append(get_y_data_frame(td6[0]), ignore_index=True)
y = y.append(get_y_data_frame(td7[0]), ignore_index=True)
y = y.append(get_y_data_frame(td8[0]), ignore_index=True)
y = y.append(get_y_data_frame(td9[0]), ignore_index=True)
y = y.append(get_y_data_frame(td10[0]), ignore_index=True)
y = y.append(get_y_data_frame(td11[0]), ignore_index=True)
y = y.append(get_y_data_frame(td12[0]), ignore_index=True)
y = y.append(get_y_data_frame(td13[0]), ignore_index=True)
y = y.append(get_y_data_frame(td14[0]), ignore_index=True)
y = y.append(get_y_data_frame(td15[0]), ignore_index=True)
y = y.append(get_y_data_frame(td16[0]), ignore_index=True)
y = y.append(get_y_data_frame(td17[0]), ignore_index=True)
y = y.append(get_y_data_frame(td18[0]), ignore_index=True)
y = y.append(get_y_data_frame(td19[0]), ignore_index=True)
y = y.append(get_y_data_frame(td20[0]), ignore_index=True)

def get_class_weights():
    data = get_y_data_frame(td1[0])
    max_val = data.max(axis=0)
    min_val = data.min(axis=0)
    delta = max_val - min_val
    class_weights = (1.0 / delta.values)/ 8
    return class_weights

n_features = 1
n_classes = y.shape[1]
print(y)

X, y = shuffle(X.values, y.values)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)

images = Input(shape=(224,224,3),name = 'image_input')
non_image_df = Input(shape=(14,),name = 'non_image_input')
inception_v3_model = InceptionV3(weights='imagenet', include_top=False)
for layer in inception_v3_model.layers:
    layer.trainable=False
inception_v3_model.summary()

#Use the generated model 
inception_v3_model_conv = inception_v3_model(images)
#forward = GlobalAveragePooling2D()(inception_v3_model_conv)
cnn = GlobalAveragePooling2D()(inception_v3_model_conv)
final = concatenate([cnn, non_image_df])    #forward
final = Dense(1024,activation="relu")(final)
final = Dropout(0.5)(final)
final = Dense(1024,activation="relu")(final)
final = Dropout(0.5)(final)
final = Dense(6, activation='linear')(final)

#Create model 
model = Model(input=[images, non_image_df], output=final)

model.summary()
adam = Adam(lr=0.001, clipvalue=1.5)
model.compile(loss='mse', optimizer=adam)

def generator(X, y, batch_size=BATCH_SIZE):
    X_copy, y_copy = X, y
    while True:
        for i in range(0, len(X_copy), batch_size):
            X_image, X_positions, y_result = [], [], []
            if i<len(X_copy):
                for x, y in zip(X_copy[i:i+batch_size], y_copy[i:i+batch_size]):
                    #print('x[0], x[1], x[2], x[3]', x[0], x[1], x[2], x[3])
                    try:
                        rx1, rx2, ry = [normalize(load_image(x[0], x[1], x[2], x[3]))], [x[4] + random.gauss(0, .08)
                                                                                         ,x[5] + random.gauss(0, .08)
                                                                                         ,x[6] + random.gauss(0, .08)
                                                                                         ,x[7] + random.gauss(0, .08)
                                                                                         ,x[8]
                                                                                         ,x[9]
                                                                                         ,x[10]
                                                                                         ,x[11]
                                                                                         ,x[12]
                                                                                         ,x[13]
                                                                                         ,x[14]
                                                                                         ,x[15]
                                                                                         ,x[16]
                                                                                         ,x[17]
                                                                                            ], [y]
                        #print(x[0], x[1], x[2], x[3])
                        y = np.array(ry)
                        X_image.append(rx1)
                        X_positions.append([rx2])
                        y_result.append(y)
                    except:
                        pass
                X_result_1, X_result_2, y_result = np.concatenate(X_image), np.concatenate(X_positions), np.concatenate(y_result)
                yield [X_result_1, X_result_2], y_result

EPOCHS = 60
PER_EPOCH = 115

class_w = get_class_weights()
filepath="/home/joe/Dev/robot_direction_control/models/InceptionV3_fc512_X2_pos-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True)
callbacks_list = [checkpoint]

history = model.fit_generator(
    generator(X_train, y_train),
    steps_per_epoch=PER_EPOCH,
    epochs=EPOCHS,
    validation_data=generator(X_valid, y_valid),
    #validation_steps=len(y_valid)//(BATCH_SIZE),
    class_weight = class_w,
    validation_steps=2,
    callbacks=callbacks_list
)