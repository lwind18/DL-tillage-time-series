"""
Created on Mon May 23 17:27:09 2022
@author: Dong.Luo
This code is for building 3D cnn model for planet purpose
Input shape(B, 32,32,731, 4)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

## final 1dcnn model
def build_1dcnn(depth=731, band = 4, n_classes=4, nfilter=64):
    """
    Build a 1D convolutional neural network model.
    add batchnormalization and dropout
    """
    inputs = keras.Input((depth, band))

    x = layers.Conv1D(filters=nfilter, kernel_size=3, padding = 'same', kernel_initializer='he_normal', activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(pool_size=2)(x)
    

    x = layers.Conv1D(filters=nfilter*2, kernel_size=3, padding = 'same', kernel_initializer='he_normal', activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(pool_size=2)(x)
    

    x = layers.Conv1D(filters=nfilter*4, kernel_size=3, padding = 'same', kernel_initializer='he_normal', activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(pool_size=2)(x)
    

    x = layers.Conv1D(filters=nfilter*8, kernel_size=3, padding = 'same', kernel_initializer='he_normal', activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(pool_size=2)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(units=nfilter*8, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(n_classes)(x)
    outputs = layers.Softmax()(x)
    model = Model(inputs, outputs)
    return model

## final 2dcnn model
def build_2dcnn0(width=32, height=32, band = 731*4, n_classes=4, nfilter=64):
    """
    Build a 2D convolutional neural network model.
    add batchnormalization and dropout
    """
    inputs = keras.Input((width, height, band))

    x = layers.Conv2D(filters=nfilter, kernel_size=3, padding = 'same', kernel_initializer='he_normal', activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=2)(x)
    
    x = layers.Conv2D(filters=nfilter*2, kernel_size=3, padding = 'same', kernel_initializer='he_normal', activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=2)(x)
    
    x = layers.Conv2D(filters=nfilter*4, kernel_size=3, padding = 'same', kernel_initializer='he_normal', activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=2)(x)    

    x = layers.Conv2D(filters=nfilter*8, kernel_size=3, padding = 'same', kernel_initializer='he_normal', activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=2)(x)    

    x = layers.Flatten()(x)
    x = layers.Dense(units=nfilter*8, activation="relu")(x)
#    x = layers.Dense(units=1024, activation="relu")(x)
#    x = layers.Dense(units=256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(n_classes)(x)
    outputs = layers.Softmax()(x)
    model = Model(inputs, outputs)
    return model

## final 3dcnn model                      
def build_3dcnn1(width=32, height=32, depth=731, band = 4, n_classes=4, nfilter=64):
    """
    Build a 3D convolutional neural network model.
    build_3dcnn1: add batchnormalization and dropout(0.5)
    """

    inputs = keras.Input((width, height, depth, band))

    x = layers.Conv3D(filters=nfilter, kernel_size=3, padding = 'same', kernel_initializer='he_normal', activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    

    x = layers.Conv3D(filters=nfilter*2, kernel_size=3, padding = 'same', kernel_initializer='he_normal', activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    

    x = layers.Conv3D(filters=nfilter*4, kernel_size=3, padding = 'same', kernel_initializer='he_normal', activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    

    x = layers.Conv3D(filters=nfilter*8, kernel_size=3, padding = 'same', kernel_initializer='he_normal', activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    

    x = layers.Flatten()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)         
    
    x = layers.Dense(n_classes)(x)
    outputs = layers.Softmax()(x)

    # Define the model.
    model = Model(inputs, outputs)
    return model

## final 4*4 model for v6.8
def build_3dcnn4(width=32, height=32, depth=731, band = 4, n_classes=4, nfilter=64):
    """
    Build a 3D convolutional neural network model (for input width and height =4).
    build_3dcnn1: add batchnormalization and dropout(0.5)
    build_3dcnn4: adapt to input as (batch, 4,4, depth, band)
    """

    inputs = keras.Input((width, height, depth, band))

    x = layers.Conv3D(filters=nfilter, kernel_size=1, padding = 'same', kernel_initializer='he_normal', activation="relu")(inputs)
    x = layers.BatchNormalization()(x)    

    x = layers.Conv3D(filters=nfilter*2, kernel_size=1, padding = 'same', kernel_initializer='he_normal', activation="relu")(x)
    x = layers.BatchNormalization()(x)    

    x = layers.Conv3D(filters=nfilter*4, kernel_size=1, padding = 'same', kernel_initializer='he_normal', activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2)(x)    

    x = layers.Conv3D(filters=nfilter*8, kernel_size=1, padding = 'same', kernel_initializer='he_normal', activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2)(x)    

    x = layers.Flatten()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)         
    
    x = layers.Dense(n_classes)(x)
    outputs = layers.Softmax()(x)
    # Define the model.
    model = Model(inputs, outputs)
    return model
##**************************************************************************************************************************************
    
# inputs = Input((32,32,731,4))
#model = build_3dcnn(32,32,731)
#print(model.summary())












