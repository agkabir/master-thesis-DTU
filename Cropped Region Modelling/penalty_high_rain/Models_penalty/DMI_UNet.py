#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 21:46:50 2021

@author: alamgirkabir
"""

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import tensorflow as tf
import numpy as np

def conv_block(num_filter, filter_size,inputs, drop_out_conv = False):
    conv1 = Conv2D(num_filter, filter_size, padding='same', kernel_initializer='he_normal')(inputs)
    
    conv1 = Activation("relu")(conv1)
    conv2 = Conv2D(num_filter, filter_size, padding='same', kernel_initializer='he_normal')(conv1)
    outputs = Activation("relu")(conv2)
    if drop_out_conv:
        outputs = Dropout(0.5)(outputs)
    return outputs

def conv_down_sample(num_filter, filter_size,inputs, drop_out = False):
    skip = conv_block(num_filter, filter_size,inputs)
    #skip = BatchNormalization()(skip)
    outputs = MaxPooling2D(pool_size=(2, 2))(skip)
    if drop_out:
        outputs = Dropout(0.5)(outputs)
    return skip,outputs
    
    

def up_sample_and_concatenate_conv__(conv,skip,num_filter,filter_size):
    up_and_concate = concatenate([UpSampling2D(size=(2, 2))(conv), skip], axis=3)
    outputs = conv_block(num_filter, filter_size,up_and_concate)
    return outputs

def up_sample_and_concatenate_conv(conv,skip,num_filter,filter_size):
    up_and_concate = concatenate([Conv2DTranspose(num_filter,(2,2),strides=(2, 2),padding='same')(conv), skip])
    outputs = conv_block(num_filter, filter_size,up_and_concate)
    return outputs


def DMI_UNet(input_shape=()):

    inputs = Input(input_shape)
    
    # First layer convolution and down sampling
    first_skip,first_down = conv_down_sample(64, 3,inputs)
    # second layer convolution and down sampling
    second_skip,second_down = conv_down_sample(128, 3,first_down)
    # third layer convolution and down sampling
    third_skip,third_down = conv_down_sample(256, 3,second_down)
    # fourth layer convolution and down sampling
    fourth_skip,fourth_down = conv_down_sample(512, 3,third_down, drop_out= True)
    # fifth layer convolution
    fifth_conv = conv_block(1024, 3,fourth_down, drop_out_conv = True)
    
    # First upsampling and concatenations layers
    up_and_con_conv_sixth = up_sample_and_concatenate_conv(fifth_conv,fourth_skip,512,3)
    # Second upsampling and concatenations layers
    up_and_con_conv_seventh = up_sample_and_concatenate_conv(up_and_con_conv_sixth,third_skip,256,3)
    # Third upsampling and concatenations layers
    up_and_con_conv_eight = up_sample_and_concatenate_conv(up_and_con_conv_seventh,second_skip,128,3)
    # Fourth upsampling and concatenations layers
    up_and_con_conv_nine = up_sample_and_concatenate_conv(up_and_con_conv_eight,first_skip,64,3)
    
    # Predicted output
    outputs = Conv2D(6, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up_and_con_conv_nine)
    # FC convolution
    outputs = Conv2D(6, 1, activation='linear')(outputs)

    model = Model(inputs=inputs, outputs=outputs)

    return model