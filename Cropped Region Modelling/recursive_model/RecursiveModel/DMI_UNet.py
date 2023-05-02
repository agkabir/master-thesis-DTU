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
#gpus = tf.config.experimental.list_physical_devices('GPU') 
#tf.config.experimental.set_memory_growth(gpus[0], True)

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


def unet_output(inputs):
    # First layer convolution and down sampling
    first_skip,first_down = conv_down_sample(32, 3,inputs)
    # second layer convolution and down sampling
    second_skip,second_down = conv_down_sample(64, 3,first_down)
    # third layer convolution and down sampling
    third_skip,third_down = conv_down_sample(128, 3,second_down)
    # fourth layer convolution and down sampling
    fourth_skip,fourth_down = conv_down_sample(256, 3,third_down, drop_out= True)
    # fifth layer convolution
    fifth_conv = conv_block(512, 3,fourth_down, drop_out_conv = True)
    
    # First upsampling and concatenations layers
    up_and_con_conv_sixth = up_sample_and_concatenate_conv(fifth_conv,fourth_skip,256,3)
    # Second upsampling and concatenations layers
    up_and_con_conv_seventh = up_sample_and_concatenate_conv(up_and_con_conv_sixth,third_skip,128,3)
    # Third upsampling and concatenations layers
    up_and_con_conv_eight = up_sample_and_concatenate_conv(up_and_con_conv_seventh,second_skip,64,3)
    # Fourth upsampling and concatenations layers
    up_and_con_conv_nine = up_sample_and_concatenate_conv(up_and_con_conv_eight,first_skip,32,3)
    
    # Predicted output
    outputs = Conv2D(1, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up_and_con_conv_nine)
    # FC convolution
    #outputs = Conv2D(1, 1, activation='linear')(outputs)
    return outputs

def DMI_UNet(input_shape=()):
    original_inputs = Input(input_shape)
    ## first input output
    first_input = original_inputs[:,:,:,0:2]
    first_output = unet_output(first_input)
    ## second input output
    second_input = Concatenate(axis=3)([first_output, original_inputs[:,:,:,2:3]])
    second_output = unet_output(second_input)
    #final_outout = Concatenate(axis=3)([first_output, second_output])
    ## third input output
    third_input = Concatenate(axis=3)([second_output, original_inputs[:,:,:,3:4]])
    third_output = unet_output(third_input)
    #final_outout = Concatenate(axis=3)([final_outout, third_output])
    ###
    fourth_input = Concatenate(axis=3)([third_output, original_inputs[:,:,:,4:5]])
    fourth_output = unet_output(fourth_input)
    #final_outout = Concatenate(axis=3)([final_outout, fourth_output])
    
    ##
    fifth_input = Concatenate(axis=3)([fourth_output, original_inputs[:,:,:,5:6]])
    fifth_output = unet_output(fifth_input)
    #final_outout = Concatenate(axis=3)([final_outout, fifth_output])
    ##
    sixth_input = Concatenate(axis=3)([fifth_output, original_inputs[:,:,:,6:7]])
    sixth_output = unet_output(sixth_input)
    final_output = Concatenate(axis=3)([first_output,second_output,third_output,fourth_output,fifth_output, sixth_output])
    
    
    #print(final_output.shape)
    outputs = Conv2D(6, 1, activation='linear')(final_output)
    #print(outputs.shape)
    model = Model(inputs=original_inputs, outputs=outputs)
    return model
