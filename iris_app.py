# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 17:19:55 2022

@author: Gaming
"""

import streamlit as st


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame
import plotly.express as px
from mlxtend.plotting import plot_decision_regions
from models import *
from data_prep import *

iris_df, iris_data = create_data()

pca , X_pca = fit_pca(iris_data.data)

X_pca['class'] = iris_df['class']



st.title('Iris Machine Learning App')


container1 = st.container()

with container1:
    
    container1.header('Data')
    container1.dataframe(iris_df)
    

container2 = st.container()

with container2:
    
    cols = st.columns([1,2])
    
    
    x1  = cols[0].selectbox('Choose a variable',iris_data.feature_names)
    
    cols[0].dataframe(iris_data.data[[x1]].describe())
    
    cols[1].header('Density of a variable')
    
    fig = px.histogram(iris_df,x=x1,nbins=20,width=800,height=400)
    
    cols[1].plotly_chart(fig,use_container_width=True)

container3 = st.container()

with container3:
    
    container3.header('Features Scatter Plot')
    
    cols = st.columns(2)
    
    features = iris_data.feature_names.copy()
    
    x = cols[0].selectbox('X variable',features)
    
    y = cols[1].selectbox('Y variable',features)
    
    scat_plot = px.scatter(iris_df,x,y,color='class',trendline='ols',
                           trendline_scope='overall')
    container3.plotly_chart(scat_plot)

container4 = st.container()

with container4:
    
    container4.header('Principal Componet Analysis')
    
    
    
    tabs = st.tabs(['PC1 VS PC 2 Scatterplot','Acumulate Explained Variance Ratio Barplot'])
    
    
    with tabs[0]:
        
      
        fig = px.scatter(data_frame=X_pca,x='pca0',y='pca1',color='class')
        st.plotly_chart(fig)
        
    with tabs[1]:
        
        df = {'Principal Component':[0,1,2,3],
              'Acumulate Explained Variance Ratio':pca.explained_variance_ratio_.cumsum()}
        
        df = DataFrame(df)
        
        fig = px.bar(df,x='Principal Component',
                   y='Acumulate Explained Variance Ratio')
        
        st.plotly_chart(fig)

container5 = st.container()

with container5:
    
    container5.header('Neural Network Model')
    
    
    
    tabs = st.tabs(['Model','Results'])
    
    
    with tabs[0]:
    
        
        
        cols = st.columns(2)
        
        model = cols[0].radio('Choose a model',
                 ['Single Layer Perceptron','Multi-Layer Perceptron'])
        
        rand_seed = cols[0].number_input('Seed',1,1000,step=1)
        
        prepro = cols[0].selectbox('Data preprocesssing',
                                   ['MinMax','Standar','Robust'])
        
        
        train_data_split = cols[1].slider('Training data split',0.5,0.95,step=0.05)
        
        
        use_pca = cols[1].checkbox('Use PCA')
        
        var_features = cols[1].multiselect('Features',iris_data.feature_names.copy(),
                                       max_selections=4)
        
        
        
        if model == 'Single Layer Perceptron':
            
            cols1 = st.columns(3)
            cols3 = st.columns(4)
            
            Epochs = cols1[0].number_input('Epochs',1,1000,step=1)
            Batch_size = cols1[1].number_input('Batch Size',1,50,step=1)
            Learning_rate = cols1[2].number_input('Learning Rate',0.001,0.999,
                                                 step=0.001,format='%f')
            optimizer = cols3[0].selectbox('Optimizers', 
                                              ['SGD','Adam','RMSprop'])
            initializer = cols3[1].selectbox('Layer weight initializers', 
                                              ['Glorot Uniform','Glorot Normal','He Normal'])
            
            l1 = cols3[2].number_input('L1 regularization',0.0,0.99,step=0.01)
            l2 = cols3[3].number_input('L2 regularization',0.0,0.99,step=0.01)
            
        elif model == 'Multi-Layer Perceptron':
            
            cols1 = st.columns(3)
            cols2 = st.columns(3)
            cols3 = st.columns(4)
            
            Epochs = cols1[0].number_input('Epochs',1,1000,step=1)
            Batch_size = cols1[1].number_input('Batch size',1,50,step=1)
            Learning_rate = cols1[2].number_input('Learning rate',0.001,0.999,
                                                  step=0.001,format='%f')
            activation_function = cols2[0].selectbox('Activation function',
                                                    ['relu','sigmoid','tanh'])
            number_hidden_layers = cols2[1].number_input('Number of hidden layers',
                                                    1,100,step=1)
            number_neurons = cols2[2].number_input('Number of neurons',
                                                    1,1000,step=1)
            optimizer = cols3[0].selectbox('Optimizers', 
                                              ['SGD','Adam','RMSprop'])
            initializer = cols3[1].selectbox('Layer weight initializers', 
                                              ['Glorot Uniform','Glorot Normal','He Normal'])
            
            l1 = cols3[2].number_input('L1 regularization',0.0,0.99,step=0.01)
            l2 = cols3[3].number_input('L2 regularization',0.0,0.99,step=0.01)
        
        fit_model = st.button('Fit model')
            
            
    with tabs[1]:
        
        if fit_model:

            if use_pca:

                X, y = create_feat_targ(X_pca, ['pca0', 'pca1'], ['class'])

            else:

                X, y = create_feat_targ(iris_df, var_features, ['class'])

            X_train, X_test, y_train, y_test = train_test_split(X,
                                                                y,
                                                                train_size=train_data_split,
                                                                random_state=rand_seed)
            scaler = scaler_data(X_train)

            X_train = scaler.transform(X_train)

            X_test = scaler.transform(X_test)


            random_seed(rand_seed)
            
            if model == 'Single Layer Perceptron':
                
                my_model = single_layer_perceptron(optimizer,
                                                   Learning_rate,
                                                   initializer,
                                                   l1,
                                                   l2,
                                                   X.shape[1])
                
            
            elif model == 'Multi-Layer Perceptron':
                
                my_model = multi_layer_perceptron(optimizer,
                                                  Learning_rate,
                                                  initializer,
                                                  l1,
                                                  l2,
                                                  X.shape[1],
                                                  number_neurons,
                                                  number_hidden_layers,
                                                  activation_function)
            history = my_model.fit(X_train,
                                   y_train,
                                   batch_size=Batch_size,
                                   epochs=Epochs,
                                   validation_data=(X_test, y_test))
            
            
            history_data = DataFrame(history.history)
                
                
            plot_data = history_data[['loss', 'val_loss']]
            plot_data.set_axis(['Train data', 'Validation data'], axis='columns', inplace=True)
            loss_plot = px.line(plot_data,
                                labels={'index': 'Epochs'},
                                title='Loss Function')
            
            loss_plot.update_layout(yaxis_title='',
                                    legend_title_text='')
    
            st.plotly_chart(loss_plot)
    
            plot_data = history_data.iloc[:, [1, 3]]
            plot_data.set_axis(['Train data', 'Validation data'], axis='columns', inplace=True)
    
            acc_plot = px.line(plot_data,
                               labels={'index': 'Epochs'},
                               title='Accuracy')
            acc_plot.update_layout(yaxis_title='',
                                   legend_title_text='')
    
            st.plotly_chart(acc_plot)
            
            
            if X.shape[1] == 2:
                
                new_model = Onehot2int(my_model)
                
                fig_dr_plot, ax = plt.subplots(1,1,figsize=(10,8))
                ax = plot_decision_regions(X,argmax(y,axis=1),new_model)
                hand ,lab = ax.get_legend_handles_labels()
                ax.legend(hand,iris_data.target_names)
                ax.set_title('Decision Regions')
                
                
                st.pyplot(fig_dr_plot)
                
            y_test_pred = my_model.predict(X_test).argmax(axis=1)
            
            conf_mat = confusion_matrix(y_test.argmax(axis=1),y_test_pred)
            
            conf_mat = conf_mat/(conf_mat.sum(axis=1).reshape(-1,1))
            
            conf_mat = conf_mat.round(2)
            
            conf_mat = DataFrame(conf_mat,columns=iris_data.target_names,index=iris_data.target_names)
            
            conf_mat_plot = px.imshow(conf_mat,color_continuous_scale='blues',text_auto=True,zmax=1.0,
                                      zmin=0.0,title='Test data confusion matrix')
            
            st.plotly_chart(conf_mat_plot)
            
            
           



            
            
            
        


