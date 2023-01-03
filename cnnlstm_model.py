"""
Created on Thu Jun  9 14:35:37 2022
@author: Dong.Luo
This code is the cnn-lstm model for planet purpose
need to add attention moduel 
Input shape(B, 731, 32, 32, 4)
"""

import math 
import numpy as np
import tensorflow as tf 
from tensorflow import keras
import keras.backend as K
# from keras.layers import Layer
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers  
    
# this is the final cnnlstm model   (v6.3)         
def build_cnnlstm0(width=32, height=32, depth=731, band = 4, n_classes=4, nfilter=64):
    """
    build a cnn-lstm model with cnn to extract features and lstm learns time series. 2lstm
    input shape (B, depth, w, h, band)
    """
    inputs = keras.Input((depth, width, height, band))
    x = layers.TimeDistributed(layers.Conv2D(filters=nfilter, kernel_size=3, padding='same', kernel_initializer='he_normal', activation= 'relu'))(inputs)
    x = layers.TimeDistributed(layers.BatchNormalization())(x) 
    x = layers.TimeDistributed(layers.MaxPool2D(pool_size=2))(x)
    
    x = layers.TimeDistributed(layers.Conv2D(filters=nfilter*2, kernel_size=3, padding='same', kernel_initializer='he_normal', activation= 'relu'))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x) 
    x = layers.TimeDistributed(layers.MaxPool2D(pool_size=2))(x)
    
    x = layers.TimeDistributed(layers.Conv2D(filters=nfilter*4, kernel_size=3, padding='same', kernel_initializer='he_normal', activation= 'relu'))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x) 
    x = layers.TimeDistributed(layers.MaxPool2D(pool_size=2))(x)
    
    x = layers.TimeDistributed(layers.Conv2D(filters=nfilter*8, kernel_size=3, padding='same', kernel_initializer='he_normal', activation= 'relu'))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x) 
    x = layers.TimeDistributed(layers.MaxPool2D(pool_size=2))(x)
    
    x = layers.TimeDistributed(layers.Flatten())(x)
    
    x = layers.LSTM(units=2048, return_sequences=True)(x)
    x = layers.LSTM(units=1024, return_sequences=False)(x)
    
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    
    x = layers.Dense(n_classes)(x)
    outputs = layers.Softmax()(x)

    model = Model(inputs, outputs)
    return model

# this is the final attention cnn-lstm model (v6.6)
def build_att_cnnlstm0(width=32, height=32, depth=731, band = 4, n_classes=4, nfilter=64):
    """
    build a cnn-lstm model with cnn to extract features and lstm learns time series. 2lstm
    input shape (B, depth, w, h, band)
    """
    inputs = keras.Input((depth, width, height, band))
    x = layers.TimeDistributed(layers.Conv2D(filters=nfilter, kernel_size=3, padding='same', kernel_initializer='he_normal', activation= 'relu'))(inputs)
    x = layers.TimeDistributed(layers.BatchNormalization())(x) 
    x = layers.TimeDistributed(layers.MaxPool2D(pool_size=2))(x)
    
    x = layers.TimeDistributed(layers.Conv2D(filters=nfilter*2, kernel_size=3, padding='same', kernel_initializer='he_normal', activation= 'relu'))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x) 
    x = layers.TimeDistributed(layers.MaxPool2D(pool_size=2))(x)
    
    x = layers.TimeDistributed(layers.Conv2D(filters=nfilter*4, kernel_size=3, padding='same', kernel_initializer='he_normal', activation= 'relu'))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x) 
    x = layers.TimeDistributed(layers.MaxPool2D(pool_size=2))(x)
    
    x = layers.TimeDistributed(layers.Conv2D(filters=nfilter*8, kernel_size=3, padding='same', kernel_initializer='he_normal', activation= 'relu'))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x) 
    x = layers.TimeDistributed(layers.MaxPool2D(pool_size=2))(x)
    
    x = layers.TimeDistributed(layers.Flatten())(x)
    
    x0 = layers.LSTM(units=2048, return_sequences=True)(x)
    x1 = layers.LSTM(units=1024, return_sequences=False)(x0)
    
    xr = layers.Reshape ((1, 1024))(x1)
    context_vector, attention_weights =  layers.MultiHeadAttention(num_heads=8,key_dim=1024//8)(query=xr, key=x0, value=x0, return_attention_scores = True)
    context_vector1 = layers.Reshape ((1024,))(context_vector)
    att =  layers.Concatenate()([x1,context_vector1])
    
    x_att = layers.Dense(512, activation='relu')(att)
    x_att = layers.Dense(128, activation='relu')(x_att)
    x_att = layers.Dense(n_classes)(x_att)        
    outputs = layers.Softmax()(x_att)

    model = Model(inputs, outputs)
    return model


# model = build_cnnlstm()  
# print(model.summary())






