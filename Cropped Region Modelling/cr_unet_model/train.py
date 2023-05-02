#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: alamgirkabir
"""
import tensorflow as tf
import pandas as pd
import numpy as np
import os, sys
from datetime import datetime
import csv
seed = 42


workfolder=os.path.dirname(os.path.realpath(__file__))

#####
sys.path.append(os.path.join(workfolder,'Model_UNet'))
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
mymodel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),loss=tf.keras.losses.MeanSquaredError(),metrics = [custom_loss])

#check if training should be continued from last checkpoint
bepoch=0
if os.path.exists(os.path.join(workfolder,'dmi_checkpoints.h5')): 
    mymodel=tf.keras.models.load_model(os.path.join(workfolder,'dmi_checkpoints.h5'))
    if os.path.exists(os.path.join(workfolder,'model_history_log.csv')):
        f=open(os.path.join(workfolder,'model_history_log.csv'),'r')
        bepoch=sum([1 for row in f])-1
        f.close()
        num_epochs=num_epochs-bepoch
# defining callbacks
if num_epochs >1:
    checkpointer = tf.keras.callbacks.ModelCheckpoint(os.path.join(workfolder,'dmi_checkpoints.h5'), monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    csv_logger = tf.keras.callbacks.CSVLogger("model_history_log.csv", append=True)
    callbacks = [checkpointer, csv_logger]
    history = mymodel.fit(train_dataset,steps_per_epoch=int(len(train_indexes[0:14450])/batch_size),validation_data = val_dataset,validation_steps=int(len(train_indexes[14450:18580])/batch_size), epochs = num_epochs,callbacks=callbacks)


