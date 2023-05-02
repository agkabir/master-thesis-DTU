# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 10:15:16 2021

@author: ALAMGIR KABIR
"""
import os
import numpy as np
import datetime
import csv
import matplotlib.pyplot as plt
# Import functions
import helper_functions as hf
from tqdm import tqdm

# Set working directory
wd = os.path.abspath(YOUR_WORK_DIRECTORY)
os.chdir(wd)

# initialization of an array for missing datapoint
missing_data_point = np.ndarray(shape=(6,512,512), dtype='uint8')
missing_data_point [:,:] = 0

# path of the dataset
data_path = os.path.abspath(YOUR_DATA_PATH)

# Create processed data directory
processed_path = os.path.join(wd,'processed_nowcast')
if os.path.isdir(processed_path) == False:
    os.makedirs(processed_path)

periods = ['2017-04','2017-05','2017-06','2017-07','2017-08','2017-09','2017-10','2017-11','2017-12','2018-01']
#periods = ['2017-09']

# path of nowcast data
nowcast_path = os.path.join(data_path,'nowcast')

for period in periods:
    #import pdb
    #pdb.set_trace()
    print('processing',period)
    raw_path = os.path.join(nowcast_path,period)
    
    if os.path.isdir(raw_path)==False:
        print('The directory does not exist')
    
    else:
        all_nowcast = [x for x in os.listdir(raw_path)]
        
        # calculate memmap dimension and start_time based on period
        dimension, start_time = hf.calculate_dimension_and_starttime(period)
        mmf = np.memmap(processed_path+'\\' +period +'.array', dtype='uint8', mode='w+', shape=(dimension,6,512,512))
         
        for idx in tqdm(range(dimension)):
        #for idx in range(dimension):
            #sleep(0.1)
            timepoint = start_time + datetime.timedelta(minutes=10)*idx
            timepoint_str = timepoint.strftime("%Y%m%d%H%M")
            timepoint_nowcast_dir = 'nowcast.'+timepoint_str
            
            #print(timepoint_str)
            if timepoint_nowcast_dir in all_nowcast:
                nowcast_h5 = [x for x in os.listdir(os.path.join(raw_path,timepoint_nowcast_dir)) if '.h5' in x]
                Y = np.ndarray(shape=(6,512,512), dtype='uint8')
                for i,item in enumerate(nowcast_h5):
                    h5_filepath = os.path.join(raw_path,timepoint_nowcast_dir,item)
                    Y[i,:,:] = hf.processed_nowcast_to_dbz(h5_filepath)
                mmf[idx,:,:,:] = Y
            else:
                mmf[idx,:,:,:]=missing_data_point
        del mmf

