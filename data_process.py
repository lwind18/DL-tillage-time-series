"""
Created on Mon May 23 14:51:01 2022
@author: Dong.Luo
This script is working on preparing data that 3d CNN need
It contains:
    (1) split train and test
    (2) normalization
    (3) train and validation
    (4) read input data for model training 
note:
    (1) mean and std just count pixels within polygon area
"""
import os 
import random 
import numpy as np
import pandas as pd
import pathlib
import math
import sys
import gc

import rasterio
import tensorflow as tf 

IMG_HEIGHT = 32
IMG_WIDTH  = 32
IMG_BANDS = 4

IMG_DEPTH  = 731
IMG_DEPTH  = 365

## https://towardsdatascience.com/understand-data-normalization-in-machine-learning-8ff3062101f0
def mean_std (file_dir):
    """function to get mean and std per each polygon (band 0-3)"""
    meanstd = np.loadtxt(open(file_dir, "rb"), dtype='<U30', delimiter=",", skiprows=1)  
    real_ms = meanstd[:,6:5854].astype(np.float32)     # linux will automatically ignore the first column
    blu_m = np.expand_dims(real_ms[:,0::8], axis=2)
    gre_m = np.expand_dims(real_ms[:,2::8], axis=2)
    red_m = np.expand_dims(real_ms[:,4::8], axis=2)
    nir_m = np.expand_dims(real_ms[:,6::8], axis=2)
    
    blu_s = np.expand_dims(real_ms[:,1::8], axis=2)
    gre_s = np.expand_dims(real_ms[:,3::8], axis=2)
    red_s = np.expand_dims(real_ms[:,5::8], axis=2)
    nir_s = np.expand_dims(real_ms[:,7::8], axis=2)
    
    x_mm = np.concatenate((blu_m, gre_m, red_m, nir_m),axis=2)    # mean shape (482 ,731, 4)
    x_ss = np.concatenate((blu_s, gre_s, red_s, nir_s),axis=2)    # std shape (482 ,731, 4)
    return x_mm, x_ss
    
def file_list (DIR, pattern=".npy"):
    file_list = list()
    for root, dirs, files in os.walk(DIR):
        for file in files:
            if pattern in file:                                                         
               file_list.append(os.path.join(root, file))                
    return sorted(file_list)

#########################fucntion: change width and height###############################################################
def slice_img (arr, m, n):
    """
    arr: 2d array shape (32,32)
    m, n: how many pixels (row and column) want to average
    """
    pl_list = []
    for x in range (0,arr.shape[0],m):
        for y in range(0,arr.shape[1],n):
            new = arr[x:x+m,y:y+n]
            new_mean = np.mean(new)
            pl_list.append(new_mean)
    pl_arr = np.array(pl_list)  
    pl_2d = pl_arr.reshape(int(arr.shape[0]/m), int(arr.shape[1]/n)) 
    return pl_2d

def get_new_hw (total_x, IN_DEPTH, m, n, IMG_WIDTH=32, IMG_HEIGHT=32, IMG_BANDS=4):
    total_n = total_x.shape[0]
    aa = np.full([total_n, int(IMG_WIDTH/m), int(IMG_HEIGHT/n), IN_DEPTH, IMG_BANDS], fill_value=0, dtype=np.float32)    
#    print(aa.shape)
    for i in range(total_n):
        arr_4d = total_x[i, :,:,:,:]
        for t in range(arr_4d.shape[2]):
            arr_3d = arr_4d[:,:,t,:]
            for b in range (arr_3d.shape[2]):
                arr_2d = arr_3d[:,:,b]
                arr_small = slice_img (arr_2d, m,n)
#                print(arr_small.shape)
                aa[i,:,:,t,b]=arr_small
    return aa    
##########################################################################################################################
## slice input data
def slice_x (total_x, num):
    """
    function to slice total_x (N, 32,32,365,4)
    num: 16, 32, 64, 128, 256, 365
    input: array with np.float32
    output: sliced array
    """
    if num ==16:
        step = round(total_x.shape[3]/num)
        indx_arr = np.arange(0, total_x.shape[3], step)
        slice_x = np.full([total_x.shape[0], total_x.shape[1], total_x.shape[2], num, total_x.shape[4]], fill_value=0, dtype=np.float32)
        for i, v in enumerate(indx_arr):
            slice_x[:,:,:,i,:] = total_x[:,:,:,v,:]
    elif num == 30:
        step = round(total_x.shape[3]/num)
        indx_arr = np.arange(0, total_x.shape[3], step) 
        indx_arr = np.delete(indx_arr,  0)
#        indx_arr = np.delete(indx_arr, -1)       
        slice_x = np.full([total_x.shape[0], total_x.shape[1], total_x.shape[2], num, total_x.shape[4]], fill_value=0, dtype=np.float32)
        for i, v in enumerate(indx_arr):
            slice_x[:,:,:,i,:] = total_x[:,:,:,v,:]
    elif num == 32:
        step = round(total_x.shape[3]/num)
        indx_arr = np.arange(0, total_x.shape[3], step)
        indx_arr = np.delete(indx_arr,  0)
        indx_arr = np.delete(indx_arr, -1)
        slice_x = np.full([total_x.shape[0], total_x.shape[1], total_x.shape[2], num, total_x.shape[4]], fill_value=0, dtype=np.float32)
        for i, v in enumerate(indx_arr):
            slice_x[:,:,:,i,:] = total_x[:,:,:,v,:]
    elif num == 64:  # left (-4), right (-5)
        step = int(total_x.shape[3]/num)
        indx_arr = np.arange(0, total_x.shape[3], step)
        indx_arr = np.delete(indx_arr, [0,1,2,3])
        indx_arr = np.delete(indx_arr, [-5,-4,-3,-2,-1])
        slice_x = np.full([total_x.shape[0], total_x.shape[1], total_x.shape[2], num, total_x.shape[4]], fill_value=0, dtype=np.float32)
        for i, v in enumerate(indx_arr):
            slice_x[:,:,:,i,:] = total_x[:,:,:,v,:]
    elif num == 128:    # left (-25), right (-30)
        step = int(total_x.shape[3]/num)
        indx_arr = np.arange(0, total_x.shape[3], step)
        indx_arr = np.delete(indx_arr, np.arange(0,24))
        indx_arr = np.delete(indx_arr, np.arange(128,159)) 
        slice_x = np.full([total_x.shape[0], total_x.shape[1], total_x.shape[2], num, total_x.shape[4]], fill_value=0, dtype=np.float32)
        for i, v in enumerate(indx_arr):
            slice_x[:,:,:,i,:] = total_x[:,:,:,v,:]
    elif num ==256:       # remove left (-50) and right (-59) based on the feature importance     
        indx_arr = np.arange(51,307)
        slice_x = np.full([total_x.shape[0], total_x.shape[1], total_x.shape[2], num, total_x.shape[4]], fill_value=0, dtype=np.float32)
        for i, v in enumerate(indx_arr):
            slice_x[:,:,:,i,:] = total_x[:,:,:,v,:]
    elif num ==236:       # remove left (-50) and right (-59) based on the feature importance     
        indx_arr = np.arange(51,287)
        slice_x = np.full([total_x.shape[0], total_x.shape[1], total_x.shape[2], num, total_x.shape[4]], fill_value=0, dtype=np.float32)
        for i, v in enumerate(indx_arr):
            slice_x[:,:,:,i,:] = total_x[:,:,:,v,:]
    elif num == 365:
        slice_x = total_x
    else:
        print("wrong number!")
    return slice_x
#######################################################################################################          
## this module works for list 
def read_xi_norm_till(x_dir_list, x_mm, x_ss, pn_list):
    """
    function to read normalize x but just use data from 09/01/2020 [244] to 08/31/2021 [608]
    x_dir_list: directory list each data has shape(n, 32, 32, 731, 6)
    x_mm: shape (482, 731, 4)
    x_ss: shape (482, 731, 4)
    return: normalized total x with shape (total_n, 32,32,365,4)
    """
    x_norm_list = []
    for i in pn_list:
        print(x_dir_list[i])
        xi = np.load(x_dir_list[i])
        x_normi = np.full([xi.shape[0], IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_BANDS], fill_value=-9999, dtype=np.float32)
        for bi in range(IMG_BANDS):
            x_normi[:,:,:,:,bi] = (xi[:,:,:,244:609,bi].astype(np.float32)-x_mm[i,244:609,bi])/x_ss[i,244:609,bi]      # chagne to np.float16 to save some memory                    
        x_norm_list.append(x_normi)
        del x_normi
        gc.collect()
    return np.vstack(x_norm_list)

def read_x_norm_till(x_dir_list, x_mm, x_ss, pn=482):
    """
    function to read normalize x but just use data from 09/01/2020 [244] to 08/31/2021 [608]
    x_dir_list: directory list each data has shape(n, 32, 32, 731, 6)
    x_mm: shape (482, 731, 4)
    x_ss: shape (482, 731, 4)
    return: normalized total x with shape (total_n, 32,32,365,4)
    """
    x_norm_list = []
    for i in range(pn):
#        print(x_dir_list[i])
        xi = np.load(x_dir_list[i])
        x_normi = np.full([xi.shape[0], IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_BANDS], fill_value=-9999, dtype=np.float32)
        for bi in range(IMG_BANDS):
            x_normi[:,:,:,:,bi] = (xi[:,:,:,244:609,bi].astype(np.float32)-x_mm[i,244:609,bi])/x_ss[i,244:609,bi]      # chagne to np.float16 to save some memory                    
        x_norm_list.append(x_normi)
        del x_normi
        gc.collect()
    return np.vstack(x_norm_list)
    
###############################################################################################################       
def random_split_train_test2 (fid, pecentage = 0.3):   
    """
    function to split train and test with solid pecentage all polygons but considering each tillage   
    KEEP 2 classes 
    return: training_index and test_index with bool list
    it needs './summary_tillage_sf_mean_std.csv' file 
    """
    poly = np.loadtxt(open('./summary_tillage_sf_mean_std.csv', "rb"), dtype='<U30', delimiter=",", skiprows=1)
    poly_id = poly[:,1]
    till_ty = poly[:,3]
    
    idx_ct = np.where(till_ty == 'conventional_till')   
    idx_nt = np.where(till_ty == 'no_till')                  

    file_index = "./split.total_n"+str(len(fid))+".for.train.test.txt"
    
    if os.path.isfile(file_index):
        print ('file already exist! ' + file_index)
        dat = np.loadtxt(open(file_index, "rb"), dtype='<U10', delimiter=",", skiprows=1)
        orders = dat.astype(np.int64)
    else:
        header = 'order'
        orders = []
        sample_ct = math.ceil(np.size(idx_ct)*pecentage)
        split_ct = math.ceil(np.size(idx_ct)/sample_ct)
        orde_ct = 0
        for i in range(split_ct):
            if i==0:
                orde_ct = np.repeat(i, sample_ct)
            else:
                orde_ct = np.concatenate((orde_ct, np.repeat(i, sample_ct)))
        orde_ct = orde_ct[range(np.size(idx_ct))]
        np.random.shuffle(orde_ct)
        ct_arr = np.vstack((np.squeeze(idx_ct, 0), orde_ct)).transpose()                         
        
        sample_nt = math.ceil(np.size(idx_nt)*pecentage)
        split_nt = math.ceil(np.size(idx_nt)/sample_nt)
        orde_nt = 0
        for i in range(split_nt):
            if i==0:
                orde_nt = np.repeat(i, sample_nt)
            else:
                orde_nt = np.concatenate((orde_nt, np.repeat(i, sample_nt)))
        orde_nt = orde_nt[range(np.size(idx_nt))]
        np.random.shuffle(orde_nt)
        nt_arr = np.vstack((np.squeeze(idx_nt, 0), orde_nt)).transpose() 
        
        till_arr = np.vstack((ct_arr, nt_arr))
        til_arr = till_arr[till_arr[:, 0].argsort()]
        
        for i in range(np.size(poly_id)):
            fid_n = np.sum(fid==poly_id[i])
            order = til_arr[i,1]
            porder = np.repeat(order, fid_n)
            orders.extend (porder)                      
        np.savetxt(file_index, orders, fmt="%s", header=header, delimiter=",")
    
    test_index = np.array(orders)==0
    train_index   = np.array(orders)!=0
    sum(test_index)
    sum(train_index  )
    return train_index,test_index    
         
# train_index, validation_index=random_split_train_test(fid, 0.3)    

def random_split_train_validation (total_n,pecentage = 0.04):
    # total_n = y_train.shape[0]
    sample_n = math.ceil(total_n*pecentage)
    split_n = math.ceil(total_n/sample_n)
    file_index = "./split.total_n"+str(total_n)+".for.training.validation.txt"
    
    if os.path.isfile(file_index):
        print ('file already exist! ' + file_index)
        dat = np.loadtxt(open(file_index, "rb"), dtype='<U10', delimiter=",", skiprows=1)
        orders = dat.astype(np.int64)
    else:
        header = 'order'
        orders = 0
        for i in range(split_n):
            if i==0:
                orders = np.repeat(i, sample_n)
            else:
                orders = np.concatenate((orders, np.repeat(i, sample_n)))
        
        orders = orders[range(total_n)]  
        np.random.shuffle(orders)
        np.savetxt(file_index, orders, fmt="%s", header=header, delimiter=",")
    
    validation_index = orders==0
    training_index   = orders!=0
    sum(validation_index)
    sum(training_index  )
    return training_index,validation_index
############################################################################################################################
## function to read train, validation and test data and normalization them
def read_planet_train_all (data_x, label_y, split=0, petg=0.04):      # train_x shape [train_n_new, 32, 32, 731, 4]
    """
    function to read train (training and validation) and test data
    args:
        data_x (for 3dcnn):   normalized x with shape (total_n,32,32,731,4) or (total_n,32,32,365,4)
        data_x (for cnnlstm): normalized x with shape (total_n,731,32,32,4) or (total_n,365, 32,32,4)
        label_y: read from the Y file with shape(total_n,2)        
        split: percentage of split train and test
        pecentage: percentage of split training and validation from train
    ourput:
        for 3dcnn:   train_x normalizated (n, 32,32,731,4),train_y, validate_x, validate_y, test_x, and test_y
        for cnnlstm: train_x normalizated (n, 731,32,32,4),train_y, validate_x, validate_y, test_x, and test_y
    """
    ## training x    
    train_n = data_x.shape[0]
    fid, till_y = label_y[:,0], label_y[:,1].astype(np.int8)
    ## training y    
    train_n_y = len(till_y)
    if (train_n_y!=train_n):
        print ('\n\n!!!!!!!!!!!!!!!!!!TOA and CLD mismatch!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n')
    
    ## split train and test 
    train_index   = np.full([train_n], fill_value=True , dtype=np.bool)
    test_index = np.full([train_n], fill_value=False, dtype=np.bool)
    if split>0: 
        train_index,test_index = random_split_train_test2 (fid,pecentage = split)
        
    train_n_new = train_index.sum()
    test_n_new = test_index.sum()
    out_arr = label_y[test_index,:]
    
    ## split training and validation from train
    training_index = np.full([train_n_new], fill_value=True , dtype=np.bool)
    validate_index = np.full([train_n_new], fill_value=True , dtype=np.bool)
    if petg>0:
        training_index, validate_index = random_split_train_validation(train_n_new,pecentage = petg)
    training_n_new = training_index.sum()
    validate_n_new = validate_index.sum()
    
    ## read training data          
    train_y = till_y[train_index][training_index] 
    train_x = data_x[train_index,:,:,:,:][training_index,:,:,:,:]
    train_x_sizeG = sys.getsizeof(train_x)/1024/1024/1024
    print ('Extract training '+str(training_n_new)+ ' patches in '+str(train_n) + ' from total 482 planet polygons ' + '{:5.2f}'.format(train_x_sizeG)+'G in memory')
    
    ## read validation data            
    validate_y = till_y[train_index][validate_index] 
    validate_x = data_x[train_index,:,:,:,:][validate_index,:,:,:,:]
    validate_x_sizeG = sys.getsizeof(validate_x)/1024/1024/1024
    print ('Extract validate '+str(validate_n_new)+ ' patches in '+str(train_n) + ' from total 482 planet polygons ' + '{:5.2f}'.format(validate_x_sizeG)+'G in memory')
                
    ## read test
    test_x=0; test_y = 0 
    if test_n_new>0:
#        test_x = np.full([train_n_new, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_BANDS], fill_value=-9999, dtype=np.float32)
                    
        test_y = till_y[test_index]
        test_x = data_x[test_index,:,:,:,:]
        test_x_sizeG = sys.getsizeof(test_x)/1024/1024/1024
        print ('Extract test '+str(test_n_new) + ' patches in '+str(train_n) + ' from total 482 planet polygons ' + '{:5.2f}'.format(test_x_sizeG)+'G in memory')                               
    return train_x, train_y, validate_x, validate_y, test_x, test_y, out_arr
        

