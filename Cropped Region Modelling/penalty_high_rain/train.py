#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: alamgirkabir
"""
import tensorflow as tf
import tensorflow.keras.backend as kb
import pandas as pd
import numpy as np
import os, sys
from datetime import datetime
import csv
seed = 42


workfolder=os.path.dirname(os.path.realpath(__file__))

#####
sys.path.append(os.path.join(workfolder,'Models_penalty'))
sys.path.append(workfolder)

import necessary_functions as nf
import DMI_UNet as dmnet

csv_path = os.path.join(workfolder,'csv_files')
data_path = os.path.join(workfolder,'Data')

# Defining training parameters
image_height = 512
image_width = 512
batch_size = 10
num_epochs = 150

## custom loss function
total_num_pixels = float(batch_size*image_height*image_width*6)
def custom_loss(y_true, y_pred):
    loss = tf.square(y_true-y_pred)
    tol = tf.cast(1e-7,dtype=loss.dtype)
    value_counts_1 = kb.sum(tf.where(y_true <= 18.0, 1, 0))
    value_counts_1_frac = tf.cast(value_counts_1,dtype=loss.dtype)/total_num_pixels
    value_counts_1_frac_bool = kb.greater(value_counts_1_frac,tol)
    loss = tf.cond(value_counts_1_frac_bool, lambda: tf.where(y_true <= 18.0, loss/value_counts_1_frac,loss), lambda: loss)
    value_counts_2 = kb.sum(tf.where(((y_true > 18.0) & (y_true <= 23.0)), 1, 0))
    value_counts_2_frac = tf.cast(value_counts_2,dtype=loss.dtype)/total_num_pixels
    value_counts_2_frac_bool = kb.greater(value_counts_2_frac,tol)
    loss = tf.cond(value_counts_2_frac_bool, lambda: tf.where(((y_true > 18.0) & (y_true <= 23.0)), loss/value_counts_2_frac,loss), lambda: loss)
    value_counts_3 = kb.sum(tf.where(((y_true > 23.0) & (y_true <= 34.0)), 1, 0))
    value_counts_3_frac = tf.cast(value_counts_3,dtype=loss.dtype)/total_num_pixels
    value_counts_3_frac_bool = kb.greater(value_counts_3_frac,tol)
    loss = tf.cond(value_counts_3_frac_bool, lambda: tf.where(((y_true > 23.0) & (y_true <= 34.0)), loss/value_counts_3_frac,loss), lambda: loss)
    value_counts_4 = kb.sum(tf.where(y_true > 34.0, 1, 0))
    value_counts_4_frac = tf.cast(value_counts_4,dtype=loss.dtype)/total_num_pixels
    value_counts_4_frac_bool = kb.greater(value_counts_4_frac,tol)
    loss = tf.cond(value_counts_4_frac_bool, lambda: tf.where(y_true > 34.0, loss/value_counts_4_frac,loss), lambda: loss)
    return tf.reduce_mean(loss)

# get the training indexes
train_periods = ['2016-12','2017-01','2017-02','2017-03','2017-04','2017-05','2017-06','2017-07','2017-08','2017-09',
                 '2017-10','2017-11','2017-12','2018-01']
train_dimensions = list(map(lambda x: nf.calculate_dimension(x) ,train_periods))
train_df = pd.concat([pd.read_csv(os.path.join(csv_path,fname+'.csv'),delimiter=';') for fname in train_periods]).reset_index(drop=True)
#train_df.head()
train_indexes = nf.return_indexes(list(train_df['Rain']))
np.random.seed(seed)
np.random.shuffle(train_indexes)
# writing into csv file
header = ['Indexes']
with open(os.path.join(workfolder, datetime.now().strftime("%H:%M:%S").replace(':','_') +'.csv'), 'w',newline = '') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(train_indexes[18580:])

# creating train and validation datasets
train_dataset = nf.return_dataset(data_path,train_periods,train_dimensions,train_indexes[0:14450],image_height,image_width,batch_size=batch_size)
val_dataset = nf.return_dataset(data_path,train_periods,train_dimensions,train_indexes[14450:18580],image_height,image_width,batch_size=batch_size)

# creating model objects
mymodel = dmnet.DMI_UNet((image_height,image_width,12))
mymodel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),loss=custom_loss)

#check if training should be continued from last checkpoint
bepoch=0
if os.path.exists(os.path.join(workfolder,'dmi_checkpoints.h5')): 
    mymodel=tf.keras.models.load_model(os.path.join(workfolder,'dmi_checkpoints.h5'),custom_objects ={'custom_loss':custom_loss})
    if os.path.exists(os.path.join(workfolder,'model_history_log.csv')):
        f=open(os.path.join(workfolder,'model_history_log.csv'),'r')
        bepoch=sum([1 for row in f])-1
        f.close()
        num_epochs=num_epochs-bepoch
# defining callbacks
if num_epochs >1:
    checkpointer = tf.keras.callbacks.ModelCheckpoint(os.path.join(workfolder,'dmi_checkpoints.h5'), monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    csv_logger = tf.keras.callbacks.CSVLogger("model_history_log.csv", append=True)
    callbacks = [checkpointer,early_stop, csv_logger]
    history = mymodel.fit(train_dataset,steps_per_epoch=int(len(train_indexes[0:14450])/batch_size),validation_data = val_dataset,validation_steps=int(len(train_indexes[14450:18580])/batch_size), epochs = num_epochs,callbacks=callbacks)