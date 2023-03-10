# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 14:33:38 2023

@author: Gaming
"""
from numpy import argmax
from numpy.random import seed as np_seed
from tensorflow.random import set_seed
from random import seed
from tensorflow import keras as kr

class Onehot2int():
    
    def __init__(self,model):
        
        self.model = model
    
    def predict(self,X):
        
        y_pred = self.model.predict(X,batch_size=1000)
        
        return argmax(y_pred,axis=1)
    
    
def random_seed(rand_seed):
    
    np_seed(rand_seed)
    set_seed(rand_seed)
    seed(rand_seed)
    
    
def single_layer_perceptron(optimizer='SGD',lr=0.001,initializer='Random Normal',
                            l1=0.0,l2=0.0,n_inputs=2):
    
    kr.backend.clear_session()
    loss=kr.losses.CategoricalCrossentropy()
    metrics = [kr.metrics.CategoricalAccuracy()]
    
    regul = kr.regularizers.L1L2(l1,l2)
    
    
    if optimizer == 'SGD':
        
        optimizer = kr.optimizers.SGD(lr)
        
    elif optimizer == 'Adam':
        
        optimizer = kr.optimizers.Adam(lr)
    
    elif optimizer == 'RMSprop':
        
        optimizer = kr.optimizers.RMSprop(lr)
    
    
    if initializer == 'Glorot Uniform':
        
        initializer = kr.initializers.GlorotUniform()
        
    elif initializer == 'Glorot Normal':
        
        initializer = kr.initializers.GlorotNormal()
    
    elif initializer == 'He Normal':
        
        initializer = kr.initializers.HeNormal()
    
    
    slp = kr.Sequential()
    
    slp.add(kr.layers.Dense(3,activation='softmax',
                            kernel_initializer=initializer,
                            bias_initializer=initializer,
                            kernel_regularizer=regul,
                            input_shape=(n_inputs,)))
    
    slp.compile(optimizer,loss,metrics)
    
    return slp

def multi_layer_perceptron(optimizer='SGD',lr=0.001,initializer='Random Normal',
                            l1=0.0,l2=0.0,n_inputs=2,neurons=100,hidden_layers=1,
                            activation='relu'):
    
    kr.backend.clear_session()
    loss=kr.losses.CategoricalCrossentropy()
    metrics = [kr.metrics.CategoricalAccuracy()]
    
    regul = kr.regularizers.L1L2(l1,l2)
    
    
    if optimizer == 'SGD':
        
        optimizer = kr.optimizers.SGD(lr)
        
    elif optimizer == 'Adam':
        
        optimizer = kr.optimizers.Adam(lr)
    
    elif optimizer == 'RMSprop':
        
        optimizer = kr.optimizers.RMSprop(lr)
    
    
    if initializer == 'Glorot Uniform':
        
        initializer = kr.initializers.GlorotUniform()
        
    elif initializer == 'Glorot Normal':
        
        initializer = kr.initializers.GlorotNormal()
    
    elif initializer == 'He Normal':
        
        initializer = kr.initializers.HeNormal()
    
    
    mlp = kr.Sequential()
    
    mlp.add(kr.layers.Dense(neurons,activation=activation,
                            kernel_initializer=initializer,
                            bias_initializer=initializer,
                            kernel_regularizer=regul,
                            input_shape=(n_inputs,)))
    
    for hl in range(hidden_layers):
        
        mlp.add(kr.layers.Dense(neurons,activation=activation,
                                kernel_initializer=initializer,
                                bias_initializer=initializer,
                                kernel_regularizer=regul))
    
    mlp.add(kr.layers.Dense(3,activation='softmax',
                            kernel_initializer=initializer,
                            bias_initializer=initializer,
                            kernel_regularizer=regul))
    
        
        
        
    
    mlp.compile(optimizer,loss,metrics)
    
    return mlp
    
    