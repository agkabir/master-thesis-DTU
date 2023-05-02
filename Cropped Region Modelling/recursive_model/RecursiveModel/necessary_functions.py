#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 08:16:45 2021

@author: alamgirkabir
"""
import numpy as np
import os
import tensorflow as tf

def dbz_to_R_marshallpalmer(dbz_data):
    z_data = tf.pow(10.0, dbz_data/10.0)
    a = 200.0 # DMI recommended value
    b = 1.6 # DMI recommended value
    rain_data = tf.pow(z_data/a, 1/b)
    return tf.cast(rain_data, tf.float64)

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

def return_indexes(data,threshold=5.0):
    indexes =[]
    idx = 0
    while idx <=(len(data)-12):
        b = data[idx:idx+12]
        #print(b)
        result = [element >= threshold for element in b]
        if all(result):
            indexes.append((idx,idx+12))
            idx+=1
        else:
            idx = idx + np.where(np.invert(result))[0][0]+1
    return indexes


def return_dataset(data_path,memamp_files,dimensions,indexes,image_height,image_width,batch_size=2):
    begins_with = 0
    
    
    while True:#begins_with < len( dimensions):
        try:
            fp_obs = np.memmap(os.path.join(data_path,'observation',memamp_files[begins_with]+'.array'), dtype='uint8', mode='r', shape=(dimensions[begins_with],image_height,image_width))
            fp_obs_2 = np.memmap(os.path.join(data_path,'observation',memamp_files[begins_with+1]+'.array'), dtype='uint8', mode='r', shape=(dimensions[begins_with+1],image_height,image_width))
            fp_nc = np.memmap(os.path.join(data_path,'nowcast',memamp_files[begins_with]+'.array'), dtype='uint8', mode='r', shape=(dimensions[begins_with],6,image_height,image_width))
            fp_nc_2 = np.memmap(os.path.join(data_path,'nowcast',memamp_files[begins_with+1]+'.array'), dtype='uint8', mode='r', shape=(dimensions[begins_with+1],6,image_height,image_width))
        except:
            fp_obs = np.memmap(os.path.join(data_path,'observation',memamp_files[begins_with]+'.array'), dtype='uint8', mode='r', shape=(dimensions[begins_with],image_height,image_width))
            fp_nc = np.memmap(os.path.join(data_path,'nowcast',memamp_files[begins_with]+'.array'), dtype='uint8', mode='r', shape=(dimensions[begins_with],6,image_height,image_width))
        for idx in range(int(len(indexes)/batch_size)):
            X = []
            Y = []
            current_dimension = np.sum(dimensions[0: begins_with+1])
            
            batch_idx_list = indexes[idx*batch_size : (idx+1)*batch_size]
            
            for start_idx,end_idx in batch_idx_list:    
                temp_list = []
                nowcast_idx = start_idx
                if end_idx <= current_dimension:
                    if start_idx < (current_dimension-dimensions[begins_with]):
                        current_dimension = 0
                        begins_with = 0
                        while current_dimension <= start_idx :
                            begins_with =  begins_with + 1
                            current_dimension = np.sum(dimensions[0: begins_with+1])

                        try:
                            fp_obs = np.memmap(os.path.join(data_path,'observation',memamp_files[begins_with]+'.array'), dtype='uint8', mode='r', shape=(dimensions[begins_with],image_height,image_width))
                            fp_obs_2 = np.memmap( os.path.join(data_path,'observation',memamp_files[begins_with+1]+'.array'), dtype='uint8', mode='r', shape=(dimensions[begins_with+1],image_height,image_width))
                            fp_nc = np.memmap(os.path.join(data_path,'nowcast',memamp_files[begins_with]+'.array'), dtype='uint8', mode='r', shape=(dimensions[begins_with],6,image_height,image_width))
                            fp_nc_2 = np.memmap(os.path.join(data_path,'nowcast',memamp_files[begins_with+1]+'.array'), dtype='uint8', mode='r', shape=(dimensions[begins_with+1],6,image_height,image_width))
                        except:
                            fp_obs = np.memmap( os.path.join(data_path,'observation',memamp_files[begins_with]+'.array'), dtype='uint8', mode='r', shape=(dimensions[begins_with],image_height,image_width))
                            fp_nc = np.memmap(os.path.join(data_path,'nowcast',memamp_files[begins_with]+'.array'), dtype='uint8', mode='r', shape=(dimensions[begins_with],6,image_height,image_width))
                        
                        if end_idx <= current_dimension:
                            temp_list.extend(np.array(fp_obs[start_idx-np.sum(dimensions[0: begins_with]).astype(int):end_idx-np.sum(dimensions[0: begins_with]).astype(int)]))
                            nowcast_array = fp_nc[nowcast_idx-np.sum(dimensions[0: begins_with]).astype(int)]
                        elif start_idx < current_dimension and end_idx > current_dimension and begins_with+1 <= len( dimensions):
                            #temp_list.append(mask)
                            temp_list.extend(np.array(fp_obs[start_idx-np.sum(dimensions[0: begins_with]).astype(int):]))
                            temp_list.extend(np.array(fp_obs_2[0:end_idx-current_dimension]))
                            if nowcast_idx < current_dimension:
                                nowcast_array = fp_nc[nowcast_idx-np.sum(dimensions[0: begins_with]).astype(int)]
                            else:
                                nowcast_array = fp_nc_2[nowcast_idx- current_dimension]
                    else :
                        #temp_list.append(mask)
                        temp_list.extend(np.array(fp_obs[start_idx-np.sum( dimensions[0: begins_with]).astype(int):end_idx-np.sum(dimensions[0: begins_with]).astype(int)]))
                        nowcast_array = fp_nc[nowcast_idx-np.sum(dimensions[0: begins_with]).astype(int)]

                elif start_idx < current_dimension and end_idx > current_dimension and begins_with+1 <= len( dimensions):
                    #temp_list.append(mask)
                    temp_list.extend(np.array(fp_obs[start_idx-np.sum(dimensions[0: begins_with]).astype(int):]))
                    temp_list.extend(np.array(fp_obs_2[0:end_idx-current_dimension]))
                    if nowcast_idx < current_dimension:
                        nowcast_array = fp_nc[nowcast_idx-np.sum(dimensions[0: begins_with]).astype(int)]
                    else:
                        nowcast_array = fp_nc_2[nowcast_idx- current_dimension]
                             
                else:
                    while current_dimension <= start_idx :
                        begins_with =  begins_with + 1
                        current_dimension = np.sum(dimensions[0: begins_with+1])
                    
                    try:
                        fp_obs = np.memmap(os.path.join(data_path,'observation',memamp_files[begins_with]+'.array'), dtype='uint8', mode='r', shape=(dimensions[begins_with],image_height,image_width))
                        fp_obs_2 = np.memmap( os.path.join(data_path,'observation',memamp_files[begins_with+1]+'.array'), dtype='uint8', mode='r', shape=(dimensions[begins_with+1],image_height,image_width))
                        fp_nc = np.memmap(os.path.join(data_path,'nowcast',memamp_files[begins_with]+'.array'), dtype='uint8', mode='r', shape=(dimensions[begins_with],6,image_height,image_width))
                        fp_nc_2 = np.memmap(os.path.join(data_path,'nowcast',memamp_files[begins_with+1]+'.array'), dtype='uint8', mode='r', shape=(dimensions[begins_with+1],6,image_height,image_width))
                    except:
                        fp_obs = np.memmap( os.path.join(data_path,'observation',memamp_files[begins_with]+'.array'), dtype='uint8', mode='r', shape=(dimensions[begins_with],image_height,image_width))
                        fp_nc = np.memmap(os.path.join(data_path,'nowcast',memamp_files[begins_with]+'.array'), dtype='uint8', mode='r', shape=(dimensions[begins_with],6,image_height,image_width))
                        del fp_obs_2
                        del fp_nc_2
                    
                    if end_idx <= current_dimension:
                            temp_list.extend(np.array(fp_obs[start_idx-np.sum(dimensions[0: begins_with]).astype(int):end_idx-np.sum(dimensions[0: begins_with]).astype(int)]))
                            nowcast_array = fp_nc[nowcast_idx-np.sum(dimensions[0: begins_with]).astype(int)]
                    elif start_idx < current_dimension and end_idx > current_dimension and begins_with+1 <= len( dimensions):
                            #temp_list.append(mask)
                            temp_list.extend(np.array(fp_obs[start_idx-np.sum(dimensions[0: begins_with]).astype(int):]))
                            temp_list.extend(np.array(fp_obs_2[0:end_idx-current_dimension]))
                            if nowcast_idx < current_dimension:
                                nowcast_array = fp_nc[nowcast_idx-np.sum(dimensions[0: begins_with]).astype(int)]
                            else:
                                nowcast_array = fp_nc_2[nowcast_idx- current_dimension]
                
                x = np.stack(np.concatenate((temp_list[0:1],nowcast_array),axis=0),0)
                x = np.moveaxis(x, 0, -1)
                X.append(x)
                y = np.stack(temp_list[6:],0)
                y = np.moveaxis(y, 0, -1)
                Y.append(y)
            if batch_size==1:
                yield np.array(X)/95,np.array(Y), start_idx
            else:
                yield np.array(X)/95,np.array(Y)
        del fp_obs
        del fp_nc
        begins_with = 0
        
def nmser(x,y):
    z=0
    if len(x)==len(y):
        for k in range(len(x)):
            if x[k]!=0:
                z = z + (((x[k]-y[k])**2)/x[k])    
                z = z/(len(x))
    return z