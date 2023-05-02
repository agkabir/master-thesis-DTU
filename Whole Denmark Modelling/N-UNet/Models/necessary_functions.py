#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 08:16:45 2021

@author: alamgirkabir
"""
import numpy as np
import os
import h5py
def calculate_dimension(period):
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
    return dimension

def return_indexes(data,threshold=28.162):
    indexes =[]
    idx = 0
    while idx <=(len(data)-12):
        b = data[idx:idx+12]
        #print(b)
        result = [element == threshold for element in b]
        if all(result):
            indexes.append((idx,idx+12))
            idx+=1
        else:
            idx = idx + np.where(np.invert(result))[0][0]+1
    return indexes


def return_dataset(data_path,memamp_files,dimensions,indexes,mask,image_height,image_width,batch_size=2):
    begins_with = 0
    
    
    while True:#begins_with < len( dimensions):
        try:
            fp_obs = np.memmap(os.path.join(data_path,memamp_files[begins_with]+'.array'), dtype='uint8', mode='r', shape=(dimensions[begins_with],image_height,image_width))
            fp_obs_2 = np.memmap(os.path.join(data_path,memamp_files[begins_with+1]+'.array'), dtype='uint8', mode='r', shape=(dimensions[begins_with+1],image_height,image_width))
        except:
            fp_obs = np.memmap(os.path.join(data_path,memamp_files[begins_with]+'.array'), dtype='uint8', mode='r', shape=(dimensions[begins_with],image_height,image_width))
        for idx in range(int(len(indexes)/batch_size)):
            X = []
            Y = []
            current_dimension = np.sum(dimensions[0: begins_with+1])
            
            batch_idx_list = indexes[idx*batch_size : (idx+1)*batch_size]
            
            for start_idx,end_idx in batch_idx_list:
                #print('i am inside for', len(batch_idx_list))
                temp_list = []
                if end_idx <= current_dimension:
                    if start_idx < (current_dimension-dimensions[begins_with]):
                        current_dimension = 0
                        begins_with = 0
                        while current_dimension <= start_idx :
                            begins_with =  begins_with + 1
                            current_dimension = np.sum(dimensions[0: begins_with+1])

                        try:
                            fp_obs = np.memmap(os.path.join(data_path,memamp_files[begins_with]+'.array'), dtype='uint8', mode='r', shape=(dimensions[begins_with],image_height,image_width))
                            fp_obs_2 = np.memmap( os.path.join(data_path,memamp_files[begins_with+1]+'.array'), dtype='uint8', mode='r', shape=(dimensions[begins_with+1],image_height,image_width))
                        except:
                            fp_obs = np.memmap( os.path.join(data_path,memamp_files[begins_with]+'.array'), dtype='uint8', mode='r', shape=(dimensions[begins_with],image_height,image_width))
                            
                        if end_idx <= current_dimension:
                            #temp_list.append(mask)
                            temp_list.extend(np.array(fp_obs[start_idx-np.sum(dimensions[0: begins_with]).astype(int):end_idx-np.sum(dimensions[0: begins_with]).astype(int)]))
                        elif start_idx < current_dimension and end_idx > current_dimension and begins_with+1 <= len( dimensions):
                            #temp_list.append(mask)
                            temp_list.extend(np.array(fp_obs[start_idx-np.sum(dimensions[0: begins_with]).astype(int):]))
                            temp_list.extend(np.array(fp_obs_2[0:end_idx-current_dimension]))
                            
                    else :
                        #temp_list.append(mask)
                        temp_list.extend(np.array(fp_obs[start_idx-np.sum( dimensions[0: begins_with]).astype(int):end_idx-np.sum(dimensions[0: begins_with]).astype(int)]))
                    

                elif start_idx < current_dimension and end_idx > current_dimension and begins_with+1 <= len( dimensions):
                    #temp_list.append(mask)
                    temp_list.extend(np.array(fp_obs[start_idx-np.sum(dimensions[0: begins_with]).astype(int):]))
                    temp_list.extend(np.array(fp_obs_2[0:end_idx-current_dimension]))
                    
                else:
                    while current_dimension <= start_idx :
                        begins_with =  begins_with + 1
                        current_dimension = np.sum(dimensions[0: begins_with+1])
                    
                    try:
                        fp_obs = np.memmap(os.path.join(data_path,memamp_files[begins_with]+'.array'), dtype='uint8', mode='r', shape=(dimensions[begins_with],image_height,image_width))
                        fp_obs_2 = np.memmap( os.path.join(data_path,memamp_files[begins_with+1]+'.array'), dtype='uint8', mode='r', shape=(dimensions[begins_with+1],image_height,image_width))
                    except:
                        fp_obs = np.memmap( os.path.join(data_path,memamp_files[begins_with]+'.array'), dtype='uint8', mode='r', shape=(dimensions[begins_with],image_height,image_width))
                        del fp_obs_2
                    
                    if end_idx <= current_dimension:
                            #temp_list.append(mask)
                            temp_list.extend(np.array(fp_obs[start_idx-np.sum(dimensions[0: begins_with]).astype(int):end_idx-np.sum(dimensions[0: begins_with]).astype(int)]))
                    elif start_idx < current_dimension and end_idx > current_dimension and begins_with+1 <= len( dimensions):
                            #temp_list.append(mask)
                            temp_list.extend(np.array(fp_obs[start_idx-np.sum(dimensions[0: begins_with]).astype(int):]))
                            temp_list.extend(np.array(fp_obs_2[0:end_idx-current_dimension]))

                
                x = np.append([mask],np.array(temp_list[0:6])/93,axis=0)
                x = np.moveaxis(x, 0, -1)
                X.append(x)
                y = np.stack(temp_list[6:],0)
                y = np.moveaxis(y, 0, -1)
                Y.append(y)
            
            yield np.array(X),np.array(Y)
        del fp_obs
        begins_with = 0



def read_raw_radardata(file_path):   
    
    with h5py.File(file_path, 'r') as file:
        raw_data = file["dataset1"]["data1"]["data"]
        raw_data_array = raw_data[()]
    
    raw_data_array = raw_data_array[126:1598, 109:1869]
    raw_data_array = raw_data_array.astype('float16')
    return(raw_data_array)
# Function that converts raw data to dBZ values
def raw_radardata_to_dbz(raw_data_array):
    raw_data_array[raw_data_array == 255] = np.nan # values of 255 in raw data are actually NaN
    zero_values = raw_data_array == 0 # store where zero values are located in the grid, as these are changed by the transformation below

    gain = 0.5
    offset = -32
    dbz_data = offset + gain * raw_data_array # convert to dBZ
    
    dbz_data[zero_values] = 0 # insert NaN where there originally where zeros (if zeros are needed rather than NaN, change np.nan to 0 here)
    dbz_data = np.nan_to_num(dbz_data,nan = 0) # set nan values by 0
    dbz_data = np.where(dbz_data<0,0,dbz_data)
    dbz_data = np.ceil(dbz_data)
    return(dbz_data.astype('uint8'))