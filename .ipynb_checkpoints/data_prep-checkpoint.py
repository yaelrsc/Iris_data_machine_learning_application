# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 14:39:39 2023

@author: Gaming
"""

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler,StandardScaler,RobustScaler
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from pandas import DataFrame

def create_feat_targ(df,features,target):
    
    encoder = OneHotEncoder(sparse=False)
    
    X = df[features].values
    y = encoder.fit_transform(df[target])
    
    return X,y
    


def create_data():
    
    iris_data = load_iris(as_frame=True)
    iris_df = iris_data.frame

    iris_df = iris_df.rename(columns={'target':'class'})

    for i in range(iris_df.shape[0]):
        
        class_ = iris_df['class'].iloc[i]
        
        iris_df['class'].iloc[i] = iris_data.target_names[class_]
        
    return iris_df , iris_data


def fit_pca(X):
    
    pca = PCA()

    pca.fit(X)
        
    X_pca = DataFrame(pca.transform(X),
                        columns=pca.get_feature_names_out())
    
    return pca , X_pca

def scaler_data(X,scaler='MinMax'):
    
    if scaler == 'MinMax':
        
        scaler = MinMaxScaler()
    
    elif scaler == 'Standar':
        
        scaler = StandardScaler()
        
    elif scaler == 'Robust':
        
        scaler == RobustScaler()
    
    scaler.fit(X)
        
    return scaler