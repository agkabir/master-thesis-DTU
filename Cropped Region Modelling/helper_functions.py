# Python script with functions and an example that can:
# - Read a standard HDF5 radar file from DMI
# - Convert raw data to dBZ and rainfall intensity

import os
import numpy as np
import datetime
import h5py

# Function that reads the raw data from an HDF5 file
def read_raw_radardata(file_path):   
    
    with h5py.File(file_path, 'r') as file:
        raw_data = file["dataset1"]["data1"]["data"]
        raw_data_array = raw_data[()]
    
    #raw_data_array = raw_data_array[126:1598, 109:1869]
    raw_data_array = raw_data_array.astype('float32')
    return(raw_data_array)

# Function that converts raw data to dBZ values
def raw_radardata_to_dbz(raw_data_array):
    raw_data_array[raw_data_array == 255] = np.nan # values of 255 in raw data are actually NaN
    zero_values = raw_data_array == 0 # store where zero values are located in the grid, as these are changed by the transformation below

    gain = 0.5
    offset = -32
    dbz_data = offset + gain * raw_data_array # convert to dBZ
    
    dbz_data[zero_values] = 0 # insert NaN where there originally where zeros (if zeros are needed rather than NaN, change np.nan to 0 here)
    dbz_data = np.nan_to_num(dbz_data,nan = 255) # set nan values by 255
    dbz_data = np.where(dbz_data < 0, 0,dbz_data)
    dbz_data = np.ceil(dbz_data[835:1347,530:1042])
    #print('Min:',np.min(dbz_data))
    #print('Max:',np.max(dbz_data))
    return(dbz_data.astype('uint8'))

def processed_nowcast_to_dbz(file_path):
    raw_data = read_raw_radardata(file_path)
    dbz = raw_radardata_to_dbz(raw_data)
    dbz = np.where(dbz==255,0,dbz)
    return dbz

# Function that converts dBZ values to rainfall rates with constant Marshall-Palmer
def dbz_to_R_marshallpalmer(dbz_data):
    z_data = np.power(10, dbz_data/10)

    a = 200 # DMI recommended value
    b = 1.6 # DMI recommended value
    rain_data = np.power(z_data/a, 1/b)

    #rain_data_nozeros = rain_data
    #rain_data_nozeros[zero_values] = np.nan
    
    return(rain_data)



def calculate_dimension_and_starttime(period):
    '''
    This function calculates the dimensions of the memmap file for the given month and year
    also creates a start time point
    '''
    year_month = list(map(int,period.split('-')))
    if year_month[1] == 2 :
        if year_month[0]%4 == 0:
            dimension = 29*144
        else:
            dimension = 28*144
    elif year_month[1] in (4,6,9,11):
        dimension = 30*144
    else:
        dimension = 31*144
    start_datetime = datetime.datetime(year=year_month[0], month=year_month[1],day=1,hour=0,minute=0)
    return dimension, start_datetime

def write_to_memmap_rain_data(file_path,mf,idx):
    # path of the interpolated file   
    dbz_data = raw_radardata_to_dbz(read_raw_radardata(file_path))
    mf[idx,:,:]=dbz_data
    zero_per = round((dbz_data==0).sum()*100/(512*512),3)
    nan_per = round((dbz_data==255).sum()*100/(512*512),3)
    rain_per = round((100 - zero_per - nan_per),3)
    return nan_per,zero_per,rain_per
