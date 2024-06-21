import numpy as np
import os
import PIL
import PIL.Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_curve

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import LSTM,Dropout,Dense,TimeDistributed,Conv1D,MaxPooling1D,Flatten

import tensorflow as tf

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display

import json
import math
import os
import cv2
from PIL import Image
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet201
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
import scipy
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import backend as K
import gc
from functools import partial
from sklearn import metrics
from collections import Counter
import json
import itertools
from keras.utils.vis_utils import plot_model

import keras
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D, SeparableConv1D, Conv2D, MaxPooling2D, SeparableConv2D, UpSampling2D, UpSampling1D
# from tensorflow.keras.layers import Conv1D, MaxPooling1D, SeparableConv1D, Conv2D, MaxPooling2D, SeparableConv2D, UpSampling2D, UpSampling1D
from keras.layers import concatenate
from keras.regularizers import l2
from keras.optimizers import SGD, RMSprop
from keras.utils import to_categorical
from tensorflow.keras.layers import BatchNormalization
from keras.utils.vis_utils import plot_model
from keras.layers import Input, GlobalAveragePooling2D
from keras import models
from keras.models import Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Embedding, Dense, Dropout
from tensorflow.keras.models import Sequential

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.python.keras.layers import Layer
"""
confusion metrics
"""
def confusion_metrics (conf_matrix):
# save confusion matrix and slice into four pieces
    TP = conf_matrix[1][1]
    TN = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]
    print('True Positives:', TP)
    print('True Negatives:', TN)
    print('False Positives:', FP)
    print('False Negatives:', FN)

    # calculate accuracy
    conf_accuracy = (float (TP+TN) / float(TP + TN + FP + FN))

    # calculate mis-classification
    conf_misclassification = 1- conf_accuracy

    # calculate the sensitivity
    conf_sensitivity = (TP / float(TP + FN))
    # calculate the specificity
    conf_specificity = (TN / float(TN + FP))

    # calculate precision
    conf_precision = (TN / float(TN + FP))
    # calculate f_1 score
    conf_f1 = 2 * ((conf_precision * conf_sensitivity) / (conf_precision + conf_sensitivity))
    print('-'*50)
    print(f'Accuracy: {(conf_accuracy,2)}')
    print(f'Mis-Classification: {(conf_misclassification,2)}')
    print(f'Sensitivity: {(conf_sensitivity,2)}')
    print(f'Specificity: {(conf_specificity,2)}')
    print(f'Precision: {(conf_precision,2)}')
    print(f'f_1 Score: {(conf_f1,2)}')


"""
for one gram, use this code to encode the data (One-Hot encoding)
"""
import numpy as np
import pandas as pd
from random import sample, seed

one_hot = {
  'A':np.array([1,0,0,0,0]).reshape(1,-1),
  'C':np.array([0,1,0,0,0]).reshape(1,-1),
  'G':np.array([0,0,1,0,0]).reshape(1,-1),
  'T':np.array([0,0,0,1,0]).reshape(1,-1),
  '$':np.array([0,0,0,0,1]).reshape(1,-1), # unknown nucleotides
  '_':np.array([0,0,0,0,0]).reshape(1,-1), # unknown nucleotides
}
one_hot

import numpy as np
import pandas as pd
from random import sample, seed

# this is the default random state across all files
RANDOM_STATE = 123432

def read_seq_data(file):
    with open(file, 'r') as f:
        all_lines = f.readlines()
        seq_data = [elt.replace('\n','').replace(' ','') for elt in list(filter(lambda line: False if '>' in line else True, all_lines))]
        seq_data = list(filter(None, seq_data))
        f.close()
    return seq_data

# if you are using acceptors, use atn_x and atp_x. if you are using donor, use dtn_x and dtp_x.

atn_x = read_seq_data(f'arabidopsis thaliana/Train/Acceptor_Train_Negative.txt')
atp_x = read_seq_data(f'arabidopsis thaliana/Train/Acceptor_Train_Positive.txt')
# dtn_x = read_seq_data(f'arabidopsis thaliana/Train/Donor_Train_Negative.txt')
# dtp_x = read_seq_data(f'arabidopsis thaliana/Train/Donor_Train_Positive.txt')


# if you are using acceptors, use atn_x_ts and atp_x_ts. if you are using donor, use dtn_x_ts and dtp_x_ts.

atn_x_ts = read_seq_data(f'arabidopsis thaliana/Test/Acceptor_Test_Negative.txt')
atp_x_ts = read_seq_data(f'arabidopsis thaliana/Test/Acceptor_Test_Positive.txt')
# dtn_x_ts = read_seq_data(f'arabidopsis thaliana/Test/Donor_Test_Negative.txt')
# dtp_x_ts = read_seq_data(f'arabidopsis thaliana/Test/Donor_Test_Positive.txt')

# if you are using acceptors, use atp_y and atn_y. if you are using donor, use dtp_y and dtn_y.

# acceptor and donor labels, positive is 1 negative is 0
atp_y = [1]*len(atp_x)
atn_y = [0]*len(atn_x)
# dtp_y = [1]*len(dtp_x)
# dtn_y = [0]*len(dtn_x)

# if you are using acceptors, use atp_y_ts and atn_y_ts. if you are using donor, use dtp_y_ts and dtn_y_ts.
atp_y_ts = [1]*len(atp_x_ts)
atn_y_ts = [0]*len(atn_x_ts)
# dtp_y_ts = [1]*len(dtp_x_ts)
# dtn_y_ts = [0]*len(dtn_x_ts)


x , y = atn_x+atp_x, atn_y+atp_y

x_ts , y_ts = atn_x_ts+atp_x_ts, atn_y_ts+atp_y_ts

train = {'sequence data':x , 'Labels':y }

test = {'Test sequence data':x_ts, 'Test Labels':y_ts}

train_df = pd.DataFrame(data=train)
test_df = pd.DataFrame(data=test)

# shuffle datasets
train_df = train_df.sample(frac=1, random_state=RANDOM_STATE)

test_df = test_df.sample(frac=1, random_state=RANDOM_STATE)
train_df


print(train_df.shape)
print(test_df.shape)

make_one_hot = lambda seq: np.vstack([one_hot[nucleo.upper()] if nucleo.upper() in 'ACGT' else one_hot['_'] for nucleo in list(seq)])
train_df['sequence data'] = train_df['sequence data'].apply(make_one_hot)
test_df['Test sequence data'] = test_df['Test sequence data'].apply(make_one_hot)

import tensorflow as tf

# categorically encode the data labels
y = tf.keras.utils.to_categorical(
    y=np.reshape(train_df['Labels'].to_numpy(), newshape=(-1,1)),
    num_classes=2,
    dtype='float32'
)

y_ts = tf.keras.utils.to_categorical(
    y=np.reshape(test_df['Test Labels'].to_numpy(), newshape=(-1,1)),
    num_classes=2,
    dtype='float32'
)

expand_1 = lambda seq_mat: seq_mat.reshape(1, seq_mat.shape[0], seq_mat.shape[1])
train_df['sequence data'] = train_df['sequence data'].apply(expand_1)
test_df['Test sequence data'] = test_df['Test sequence data'].apply(expand_1)
x = np.vstack(train_df['sequence data'].to_numpy())
y = np.vstack(train_df['Labels'].to_numpy())

x_ts = np.vstack(test_df['Test sequence data'].to_numpy())
y_ts = np.vstack(test_df['Test Labels'].to_numpy())



"""
for two and three gram
"""
# Define the path to the compressed archive file
input_file = 'Data path'

# Load the contents of the file into a dictionary
data = np.load(input_file)

# Retrieve the arrays from the dictionary
x = data['x']
y = data['y']
x_ts = data['x_ts']
y_ts = data['y_ts']

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=0.001,
            decay_steps=140,
            decay_rate=0.1,
            staircase=False)


"""
CNN 1: improved unet, use unet encoder layers
"""
#Unet 1 gram Acc

input_img = Input((602, 10))

conv1 = Conv1D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input_img)
pool1 = MaxPooling1D(pool_size=2)(conv1)

conv2 = Conv1D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
pool2 = MaxPooling1D(pool_size=2)(conv2)

conv3 = Conv1D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
pool3 = MaxPooling1D(pool_size=2)(conv3)

conv4 = Conv1D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
drop4 = Dropout(0.5)(conv4)
pool4 = MaxPooling1D(pool_size=2)(drop4)

conv5 = Conv1D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
drop5 = Dropout(0.5)(conv5)

flatten = Flatten()(drop5)
dense1 = Dense(1200, activation='relu')(flatten)
dense2 = Dense(600, activation='relu')(dense1)
dense3 = Dense(150, activation='relu')(dense2)
output = Dense(2, activation='softmax')(dense3)

model_unet = Model(inputs=input_img, outputs=output)
model_unet.summary()

"""
training unet
"""

opt = keras.optimizers.Adam(lr_schedule)
model_unet.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history_unet = model_unet.fit(x,y,
          epochs=50,
          verbose=1,
          batch_size = 32,
          callbacks=[
                    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=9, mode='min', min_delta=0.001),
                ],
          )

"""
DualConvNet: Inspired by the parallel CNN architecture of GoogleNet (also known
as InceptionNet)
"""

input_img = Input(shape=(602, 5))

# 1st layer
layer_1 = Conv1D(16, 3, padding='same', activation='relu')(input_img)
layer_1 = Conv1D(16, 3, padding='same', activation='relu')(layer_1)
layer_1 = MaxPooling1D(pool_size=2)(layer_1)
layer_1 = Dropout(0.2)(layer_1)

layer_2 = Conv1D(16, 5, padding='same', activation='relu')(input_img)
layer_2 = Conv1D(16, 5, padding='same', activation='relu')(layer_2)
layer_2 = MaxPooling1D(pool_size=2)(layer_2)
layer_2 = Dropout(0.2)(layer_2)

# Combine layer_1 and layer_2
mid_1 = concatenate([layer_1, layer_2], axis=2)

flatten = Flatten()(mid_1)

dense_1 = Dense(1024, activation='relu')(flatten)
dense_1 = BatchNormalization()(dense_1)
dense_1 = Dropout(0.3)(dense_1)

dense_2 = Dense(512, activation='relu')(dense_1)
dense_2 = BatchNormalization()(dense_2)
dense_2 = Dropout(0.3)(dense_2)

dense_3 = Dense(128, activation='relu')(dense_2)
dense_3 = BatchNormalization()(dense_3)
dense_3 = Dropout(0.3)(dense_3)

output = Dense(2, activation='softmax')(dense_3)

model_ins = Model(inputs=[input_img], outputs=output)
model_ins.summary()

"""
training DualConvNet
"""
opt = keras.optimizers.Adam(lr_schedule)
model_ins.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history_ins = model_ins.fit(x,y,
          epochs=50,
          verbose=1,
          batch_size = 32,
          callbacks=[
                    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=9, mode='min', min_delta=0.001),
                ],
          )

"""
AlexSppNet: Our deep learning model, called AlexSppNet, draws inspiration from
the architecture of AlexNet [42], a renowned convolutional neural
network for image classification. However, we have made significant
improvements to adapt it for our splice site prediction application. In
addition to the convolutional layers, ReLU activations, and fully connected
layers present in AlexNet, we have incorporated a Spatial Pyramid
Pooling (SppNet) layer.
"""


class SppnetLayer(Layer):
    def __init__(self, filters=[1], **kwargs):
        self.filters = filters
        super(SppnetLayer, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        length = 0
        for f_size in self.filters:
            length += (f_size * f_size)
        return (input_shape[0], length * input_shape[3])

    def get_config(self):
        config = {'filters': self.filters}
        base_config = super(SppnetLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# Create the model
model_Spp = Sequential()

# AlexNet Backbone
model_Spp.add(Conv1D(filters=96, input_shape=(602, 5), kernel_size=11, strides=4, padding='same', activation='relu'))
model_Spp.add(MaxPooling1D(pool_size=3, strides=2, padding='valid'))
model_Spp.add(Conv1D(filters=256, kernel_size=5, strides=1, padding='same', activation='relu'))
model_Spp.add(MaxPooling1D(pool_size=3, strides=2, padding='valid'))
model_Spp.add(Conv1D(filters=384, kernel_size=3, strides=1, padding='same', activation='relu'))
model_Spp.add(Conv1D(filters=384, kernel_size=3, strides=1, padding='same', activation='relu'))
model_Spp.add(Conv1D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'))

# SppnetLayer
model_Spp.add(Flatten())
model_Spp.add(SppnetLayer([1, 2, 4]))

# Additional Fully Connected Layers
model_Spp.add(Dense(4096, activation='relu'))
model_Spp.add(Dropout(0.2))
model_Spp.add(Dense(4096, activation='relu'))
model_Spp.add(Dropout(0.2))

# Output Layer
model_Spp.add(Dense(2, activation='softmax'))

model_Spp.summary()

"""
training AlexSppNet
"""

opt = keras.optimizers.Adam(lr_schedule)
model_Spp.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history_AlexSpp = model_Spp.fit(x,y,
          epochs=50,
          verbose=1,
          batch_size = 32,
          callbacks=[
                    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=9, mode='min', min_delta=0.001),
                ],
          )

"""
WaveNet
"""

#waveNet
class SppnetLayer(Layer):
    def __init__(self, filters=[1], **kwargs):
        self.filters = filters
        super(SppnetLayer, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        length = 0
        for f_size in self.filters:
            length += (f_size * f_size)
        return (input_shape[0], length * input_shape[2])

    def get_config(self):
        config = {'filters': self.filters}
        base_config = super(SppnetLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# Create the model
model_SppNet = Sequential()

# WaveNet Backbone
model_SppNet.add(Conv1D(filters=64, kernel_size=3, strides=1, padding='causal', activation='relu', input_shape=(602, 5)))
model_SppNet.add(Conv1D(filters=64, kernel_size=3, strides=1, padding='causal', activation='relu'))
model_SppNet.add(Conv1D(filters=64, kernel_size=3, strides=1, padding='causal', activation='relu'))
model_SppNet.add(Conv1D(filters=64, kernel_size=3, strides=1, padding='causal', activation='relu'))

# SppnetLayer
model_SppNet.add(Flatten())
model_SppNet.add(SppnetLayer([1, 2, 4]))

# Additional Fully Connected Layers
model_SppNet.add(Dense(4096, activation='relu'))
model_SppNet.add(Dropout(0.2))
model_SppNet.add(Dense(4096, activation='relu'))
model_SppNet.add(Dropout(0.2))

# Output Layer
model_SppNet.add(Dense(2, activation='softmax'))

model_SppNet.summary()


"""
training WaveNet
"""

opt = keras.optimizers.Adam(lr_schedule)
model_SppNet.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history_WaveSpp = model_SppNet.fit(x,y,
          epochs=50,
          verbose=1,
          batch_size = 32,
          callbacks=[
                    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=9, mode='min', min_delta=0.001),
                ],
          )