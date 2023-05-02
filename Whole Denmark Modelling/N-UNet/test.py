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


workfolder=os.path.dirname(os.path.realpath('__file__'))


#####
sys.path.append(os.path.join(workfolder,'Models'))
sys.path.append(workfolder)

import necessary_functions as nf

csv_path = os.path.join(workfolder,'csv_files')
data_path = os.path.join(workfolder,'Data')

# Defining training parameters
image_height = 1472
image_width = 1760
batch_size = 5
num_epochs = 150

# data periods
train_periods = ['2016-12','2017-01','2017-02','2017-03','2017-04']
train_dimensions = list(map(lambda x: nf.calculate_dimension(x) ,train_periods))
## read the test index files
index_df = pd.read_csv('./test.csv',delimiter='\n').reset_index(drop=True)
test_indexes = [tuple(np.array(item.split(',')).astype(int)) for item  in list(index_df['Indexes'])]


# importing mask array
mask_array = np.memmap( os.path.join(data_path,'mask_new.array'), dtype='uint8', mode='r', shape=(1,image_height,image_width))
mask = np.array(mask_array[0])
del mask_array

# creating test datasets
test_dataset = nf.return_dataset(data_path,train_periods,train_dimensions,test_indexes,mask,image_height,image_width,batch_size=1)

# loading trained model
mymodel=tf.keras.models.load_model(os.path.join(workfolder,'dmi_checkpoints.h5'))
X, Y,_ = next(test_dataset)
# prediction
prediction = mymodel.predict(X)
