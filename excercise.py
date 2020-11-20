# -*- coding: utf-8 -*-
import os
from keras.layers import Input, Dense, Lambda, concatenate, LSTM, Activation, Flatten, MaxPooling2D, GlobalAveragePooling1D
from keras.layers.convolutional import Conv2D, Conv1D
from keras.models import Model
from keras import backend as K
from keras.layers.core import RepeatVector, Dropout
from keras.layers.wrappers import TimeDistributed
from keras import optimizers
from keras.losses import mse
from keras.utils import plot_model


class dcenet():
    def __init__(self):


        #################### MODEL CONSTRUCTION STARTS FROM HERE ####################

        # (1-1) Construct the dynamic map model for past time
        self.occu_in = Input(shape=[32,32,3], name='x_DMap_in')
        print(self.occu_in)
        self.occu_Conv1 = Conv2D(6, kernel_size=2, strides=1, padding='same', activation='relu', name='x_Map_Conv1')(
            self.occu_in)
        print(self.occu_Conv1)
        self.occu_MP1 = MaxPooling2D(pool_size=(2, 2), strides=1, padding='same', name='DMap_MP1')(self.occu_Conv1)
        print(self.occu_MP1)
        self.occu_DP = Dropout(0.2, name="occu_DP")(self.occu_MP1)
        print(self.occu_DP)
        self.occu_FT = Flatten(name='occu_FT')(self.occu_DP)
        print(self.occu_FT)
        self.occu_model = Model(self.occu_in, self.occu_FT)
        print(self.occu_model)

        # (1-2) Add the time axis
        self.occus_in = Input(shape=(7, 32, 32, 3),
                              name='x_DMaps_in')
        self.occus_layers = TimeDistributed(self.occu_model, name='occus_layers')(self.occus_in)
        print(self.occus_layers)

net = dcenet()
