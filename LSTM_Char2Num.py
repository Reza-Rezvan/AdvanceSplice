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

X_train_Org = train_df['sequence data']
y_train_org = train_df['Labels']
X_test_Org = test_df['Test sequence data']
y_test_org = test_df['Test Labels']

"""
Character-to-numerical encoding
"""
def Char2int(xt):
  # Create character-to-integer mapping
  char_to_int = {char: i for i, char in enumerate(set(''.join(xt)))} #4 ==> {'G': 0, 'T': 1, 'C': 2, 'A': 3}

  # Convert input data to integer sequences
  X = np.zeros((len(xt), len(xt[0])), dtype=int)
  for i, seq in enumerate(xt):
    countchar={'G': 0, 'T': 0, 'C': 0, 'A': 0}
    for j, char in enumerate(seq):
      countchar[char]=countchar[char]+0
      X[i, j] = char_to_int[char]+countchar[char]
  return X

X_train=Char2int(X_train_Org)
X_test=Char2int(X_test_Org)
y_train_org=np.array(y_train_org)
y_test_org=np.array(y_test_org)

"""
ConvLSTM: The Convolutional LSTM (ConvLSTM) is a variant of the Long Short Term Memory [45] (LSTM)
model that incorporates convolutional layers.
LSTM is a type of Recurrent Neural Network (RNN) that excels at
capturing long-term dependencies and sequential patterns in data.
"""

n_signals = 1 #So far each instance is one signal. We will diversify them in next step
n_outputs = 1 #Binary Classification
#Build the model
verbose, epochs, batch_size = True, 75, 16
n_steps, n_length = 43, 14
X_train1 = X_train.reshape((X_train.shape[0], n_steps, n_length, n_signals))
# define model
model = Sequential()
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None,n_length,n_signals)))
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
model.add(TimeDistributed(Dropout(0.5)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(100))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train1, y_train_org, epochs=epochs, batch_size=batch_size, verbose=verbose)

