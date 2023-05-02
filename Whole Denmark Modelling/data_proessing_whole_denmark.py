# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 18:24:29 2021

@author: alamgirkabir
"""

import os
import numpy as np
import h5py
import datetime
import csv
from tqdm import tqdm
from time import sleep

# Set working directory
wd = os.path.abspath(YOUR_WORK_DIRECTORY)
os.chdir(wd)
# Import functions
import helper_functions as hf

# initialization of an array for missing datapoint
missing_data_point = np.ndarray(shape=(1472,1760), dtype='uint8')
missing_data_point [:,:] = 0

# path of the dataset
data_path = os.path.abspath(YOUR_DATA_PATH)

# Create processed data directory
processed_path = os.path.join(wd,'processed')
if os.path.isdir(processed_path) == False:
    os.makedirs(processed_path)

periods = ['2016-12','2017-01','2017-02','2017-03','2017-04','2017-05','2017-06','2017-07','2017-08','2017-09','2017-10','2017-11','2017-12','2018-01']
#periods = ['2017-09']

# path of radar observations
interpolated_path = os.path.join(data_path,'interpolated')

for period in periods:
    raw_path = os.path.join(interpolated_path,period)
    
    if os.path.isdir(raw_path)==False:
        print('The directory does not exist')
    
    else:
        all_h5 = [x for x in os.listdir(raw_path) if '.h5' in x]
        # initializing to track rain percentage
        rain_track = []
       
        
        # calculate memmap dimension and start_time based on period
        dimension, start_time = hf.calculate_dimension_and_starttime(period)
        mmf = np.memmap(processed_path+'\\' +period +'.array', dtype='uint8', mode='w+', shape=(dimension,1472,1760))
         
        for idx in tqdm(range(dimension)):
            timepoint = start_time + datetime.timedelta(minutes=10)*idx
            timepoint_str = timepoint.strftime("%Y%m%d%H%M")
            timepoint_h5_file = 'interpolated.'+timepoint_str+'.h5'
            if timepoint_h5_file in all_h5:
                h5_filepath = os.path.join(raw_path,timepoint_h5_file)
                nan_per, zero_per,rain_per = hf.write_to_memmap_rain_data(h5_filepath,mmf,idx)
            else:
                mmf[idx,:,:]=missing_data_point
                nan_per, zero_per,rain_per  = 100.0,0.0,0.0
            rain_track.append([nan_per,zero_per,rain_per,timepoint_h5_file])
        del mmf
        # writing into csv file
        header = ["NaN", "No Rain","Rain",'FileName']
        with open(processed_path+'\\' +period +'.csv', 'w',newline = '') as f:
            writer = csv.writer(f,quoting=csv.QUOTE_ALL,delimiter=';')
            writer.writerow(header)
            writer.writerows(rain_track)
