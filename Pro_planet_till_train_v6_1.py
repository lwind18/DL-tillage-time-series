# v6.1 Spt 02 2022 final model for 2dcnn with two classes (CT+RT, MT+NT)
"""
This code is for training tillage
note:
    (1) input shape (32,32,731,6) 6 means: blue, green, red, nir, and 2 QA layers
    (2) need to have './summary_tillage_sf_mean_std.csv'
"""
import os 
import pathlib
import random 
import numpy as np
import pandas as pd
import importlib
import socket
import logging
import gc 

import importlib
import tensorflow as tf 

import data_process
import cnn3d_model
import test_helper

version='v0_0'
if '__file__' in globals():
    print(os.path.basename(__file__))
    version=os.path.basename(__file__)[22:26] 
    print(version)    
##****************************************************************************************************
## check GPU device
import tensorflow as tf 
gpus = tf.config.list_physical_devices('GPU')

for gpu in gpus:
    print (gpu)
    tf.config.experimental.set_memory_growth(gpu, True)

if tf.test.gpu_device_name(): 
    print ('Default GPU Device has just been printed by invoking "if tf.test.gpu_device_name()"\n\n' )

    
IMG_HEIGHT = 32
IMG_WIDTH  = 32
IMG_BANDS = 4

IN_DEPTH  = 731
IN_DEPTH  = 365  
IN_DEPTH  = 16
#IN_DEPTH  = 32
#IN_DEPTH  = 64
#IN_DEPTH  = 128
#IN_DEPTH  = 256

print (IMG_BANDS)

import sys
if __name__ == "__main__":
    METHOD        =  int(sys.argv[1])
    EPOCHS         = int(sys.argv[2])    
    BATCH_SIZE    =  int(sys.argv[3])
    IN_DEPTH      =  int(sys.argv[4])        
    name_prefix = socket.gethostname()+'.'+version+'.epoch'+str(EPOCHS)+'.batch'+str(BATCH_SIZE)+'.M'+str(METHOD)
    name_prefix_no_soket = version+'.epoch'+str(EPOCHS)+'.batch'+str(BATCH_SIZE)+'.M'+str(METHOD)
    
    RESULT_DIR = './result.'+name_prefix_no_soket+'/'
    model_name = RESULT_DIR+name_prefix_no_soket+'.2dcnn.model' +'.'+ str(IN_DEPTH) + '.h5'
    if not os.path.isdir(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    
    logging.basicConfig(filename=RESULT_DIR+name_prefix+'.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
    print (name_prefix)
    ##****************************************************************************************************
    ## input file folder; read total_x, and total_y 
    XY_DIR  = "/scratch/" 
    step = 32
    x_dir = XY_DIR + 'PATCH.t95.c433.s' + str(step) +'/'
    y_dir = XY_DIR +'planet.14N.Y.433.s32.step' + str(step) + '.csv'
    
    label_y = np.loadtxt(open(y_dir, "rb"), dtype='<U30', delimiter=",", skiprows=1)
    importlib.reload(data_process)
    
    xs_list = data_process.file_list(x_dir)    
    x_mm, x_ss = data_process.mean_std('./summary_tillage_sf_mean_std.csv')     
    x_n = data_process.read_x_norm_till(xs_list, x_mm, x_ss, pn=433)    # out shape (total_n, 32,32,365, 4)  
    total_x = data_process.slice_x (x_n, IN_DEPTH)
    print(total_x.shape)
    ##*********************************************************************************
    ## prepare train and testing data    
    if not os.path.isfile(model_name):  
        train_x, train_y, validate_x, validate_y, test_x, test_y, out_arr = data_process.read_planet_train_all (total_x, label_y, split=0.3, petg=0.04)
        
        train_xx    =train_x.reshape(train_x.shape[0], train_x.shape[1],train_x.shape[2],train_x.shape[3]*train_x.shape[4]) 
        print(train_xx.shape) 
        validate_xx =validate_x.reshape(validate_x.shape[0], validate_x.shape[1],validate_x.shape[2],validate_x.shape[3]*validate_x.shape[4]) 
        test_xx     =test_x.reshape(test_x.shape[0], test_x.shape[1],test_x.shape[2],test_x.shape[3]*test_x.shape[4])           
    ##*********************************************************************************
    ## shuffle before batch -> 
    if not os.path.isfile(model_name):
        train_n = train_x.shape[0]          
        # fix bug that total number of data cannot evenly divided by batchsize by drop_remainder=True OR set batchsize as a number can divide number of data
        with tf.device("/cpu:0"): 
            train_dataset = tf.data.Dataset.from_tensor_slices( (train_xx, train_y)).shuffle(train_n+1).batch(BATCH_SIZE, drop_remainder=True)  
            validation_dataset = tf.data.Dataset.from_tensor_slices( (validate_xx, validate_y)).batch(BATCH_SIZE, drop_remainder=True)          
            del train_x, train_xx, validate_x, validate_xx 
            gc.collect()
    strategy = tf.distribute.MirroredStrategy()
    ##*********************************************************************************************************************
    ## build and train the model
    with strategy.scope():
        if os.path.isfile(model_name):
            print ('haha, this has been trained !!')
            model = tf.keras.models.load_model(model_name,compile=False)
        else:
            importlib.reload(cnn3d_model)            
            model=cnn3d_model.build_2dcnn0(IMG_HEIGHT, IMG_WIDTH, IN_DEPTH*IMG_BANDS, 2)
            logging.info(model.summary())       
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
            metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]             
            
            initial_learning_rate = 0.0001
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,
                                                                         decay_steps=100000,
                                                                         decay_rate=0.96,
                                                                         staircase=True)        
            momentum = 0.9
            if METHOD==0: 
                optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule, rho=momentum)
                model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
                model.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)          
            elif METHOD==1:                            
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=momentum)
                model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
                model.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)                        
                            
            model.save(model_name) 
                
    ##*********************************************************************************************************************
    ## predict test data 
    importlib.reload(test_helper)   
    accuracy,classes = test_helper.test_accuacy_2dcnn(model, test_xx, test_y, IN_DEPTH, BATCH_SIZE)
    print ('{:0.4f}'.format(accuracy) )
    test_out = np.concatenate((out_arr, np.expand_dims(classes, axis=1)), axis=1)
    out_test_file = './'+'planet.test.2dcnn.v6.1'+'.epoch'+str(EPOCHS)+'.batch'+str(BATCH_SIZE)+'.M'+str(METHOD) +'.step' + str(step) + '.'+str(IN_DEPTH) + '.csv'
    header = 'FID_1, till, predict'
    np.savetxt(out_test_file, test_out, fmt="%s", header=header, delimiter=",")
        
    

