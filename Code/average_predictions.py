#!/usr/bin/env python
# coding: utf-8

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from datetime import datetime
from time import time

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import SimpleRNN
from keras.layers import GRU

from tensorflow.keras.callbacks import Callback

def RNN_dataset(data, n_times, n_features):
    
    X = np.zeros((len(data)-n_times, n_times, n_features))
    Y = np.zeros(len(data)-n_times)

    for i in range(len(data) - n_times):

        X[i] = data[i:n_times+i, 0:n_features]
        Y[i] = data[n_times+i, -1]
        
    return X, Y

def RNN_dataset_pred(data, n_times, n_features):
    
    X = np.zeros((len(data)-n_times, n_times, n_features))

    for i in range(len(data) - n_times):

        X[i] = data[i:n_times+i, 0:n_features]
        
    return X
    
def preprocessing(data, n_times=24, test_size=0.2):
    
    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(data)

    data_f = scaled
    
    n_features = data.shape[-1] - 1

    X, Y = RNN_dataset(data_f, n_times, n_features)
    
    idxs = []

    for i in range(len(X)):

        if str(Y[i]) == 'nan':

            idxs.append(i)

        else:

            j = 0

            for item in X[i]:

                if str(item[0]) == 'nan' or str(item[1]) == 'nan':

                    #print("nan found")
                    idxs.append(i)

                    break

                j+= 1

        i += 1
        
    
    X_new = np.zeros((X.shape[0] - len(idxs), X.shape[1], X.shape[2]))
    Y_new = np.zeros(len(Y) - len(idxs))

    k = 0

    for i in range(len(X)):

        if i not in idxs:

            X_new[k] = X[i]
            Y_new[k] = Y[i]

            k += 1

    X_train, X_test, Y_train, Y_test = train_test_split(X_new, Y_new, test_size=test_size, random_state=4)
    
    print("Training set:", X_train.shape, "Test set:", X_test.shape)
    
    return X_train, X_test, Y_train, Y_test, scaler, X_new, Y_new

class PrintCrossPoint(Callback):
    
    def __init__(self):
        
        self.epoch_cross = ""
        self.epoch = 0
     
    def on_epoch_end(self, epoch, logs=None):
        
        self.epoch += 1
        
        logs = logs or {}
        
        current_train_loss = logs.get("loss")
        current_val_loss = logs.get("val_loss")
        
        if current_val_loss < current_train_loss:
            
            if self.epoch_cross == "":
                self.epoch_cross = self.epoch
                
            #self.model.stop_training = True
            
    def on_train_end(self, epoch, logs=None):
        
        print("Validation loss higher than training loss from epoch %s!" % self.epoch_cross)
        
class StopCrossPoint(Callback):
    
    def __init__(self):
    
        self.epoch = 0
     
    def on_epoch_end(self, epoch, logs=None):
        
        self.epoch += 1
        
        logs = logs or {}
        
        current_train_loss = logs.get("loss")
        current_val_loss = logs.get("val_loss")
        
        if current_val_loss < current_train_loss:
                
            #print("Validation loss higher than training loss from epoch %s!" % self.epoch)
                
            self.model.stop_training = True

def RNN(Nf=100, tol=0.8):

    final_train_loss_RNN = 1000.0

    train_errors = []
    val_errors = []
    N_epochs = []
    training_time = []

    slopes = []
    intercepts = []

    N = 0
    tol = 0.8

    while N < Nf:

        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2, min_delta=0.00001)
        callback_2 = StopCrossPoint()

        t0 = time()

        # design network
        model_RNN = Sequential()
        model_RNN.add(SimpleRNN(3, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])))
        model_RNN.add(Dense(1, activation='sigmoid'))
        model_RNN.compile(loss='mse', optimizer='adam')

        # fit network
        history_RNN = model_RNN.fit(
            X_train, 
            Y_train, 
            epochs=500,
            steps_per_epoch=10,
            #batch_size=100, 
            validation_data=(X_test, Y_test), 
            verbose=0, 
            shuffle=False,
            callbacks=[callback, callback_2]
            )

        time_elapsed_RNN =  time()-t0
        epochs_used_RNN = len(history_RNN.history['loss'])

        final_train_loss_RNN = (history_RNN.history['loss'][-1]*100)
        final_val_loss_RNN = (history_RNN.history['val_loss'][-1]*100)

        init_train_loss_RNN = (history_RNN.history['loss'][0]*100)

        y_pred_RNN = model_RNN.predict(X_to_predict)

        y_pred_noscale_RNN = (y_pred_RNN - scaler.min_[-1]) / scaler.scale_[-1]

        yhat_RNN = model_RNN.predict(X_scaled)

        y_pred_noscale_RNN[:, 0][df_final["PH"][window_size:].astype('str') != 'nan'] = df_final["PH"][window_size:][df_final["PH"][window_size:].astype('str') != 'nan'].values

        time_delta = df_final["Time"][window_size:] - df_final["Time"][window_size]

        time_years = np.array([(item / np.timedelta64(1, 'm')) / (60*24*365) for item in time_delta])

        Y_TREND = y_pred_noscale_RNN[y_pred_noscale_RNN.astype('str') != 'nan']
        X_TREND = time_years[y_pred_noscale_RNN[:, 0].astype('str') != 'nan']

        reg = LinearRegression().fit(X_TREND.reshape(-1,1), Y_TREND)

        slope_RNN = reg.coef_[0]
        intercept_RNN = reg.intercept_

        if final_train_loss_RNN < tol:

            print(N)

            N += 1

            slopes.append(slope_RNN)
            intercepts.append(intercept_RNN)

            train_errors.append(final_train_loss_RNN)
            val_errors.append(final_val_loss_RNN)

            N_epochs.append(epochs_used_RNN)
            training_time.append(time_elapsed_RNN)
            
    return slopes, intercepts, train_errors, val_errors, N_epochs, training_time

def LSTM_NN(Nf=100, tol=0.8):

    final_train_loss_LSTM = 1000.0

    train_errors = []
    val_errors = []
    N_epochs = []
    training_time = []

    slopes = []
    intercepts = []

    N = 0
    tol = 0.8

    while N < Nf:

        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2, min_delta=0.00001)
        callback_2 = StopCrossPoint()     

        t0 = time()

        # design network
        model_LSTM = Sequential()
        model_LSTM.add(LSTM(3, input_shape=(X_train.shape[1], X_train.shape[2])))
        model_LSTM.add(Dense(1, activation='sigmoid'))
        model_LSTM.compile(loss='mse', optimizer='adam')

        # fit network
        history_LSTM = model_LSTM.fit(
            X_train, 
            Y_train, 
            epochs=500,
            steps_per_epoch=10,
            #batch_size=100, 
            validation_data=(X_test, Y_test), 
            verbose=0, 
            shuffle=False,
            callbacks=[callback, callback_2]
            )

        time_elapsed_LSTM =  time()-t0
        epochs_used_LSTM = len(history_LSTM.history['loss'])

        final_train_loss_LSTM = (history_LSTM.history['loss'][-1]*100)
        final_val_loss_LSTM = (history_LSTM.history['val_loss'][-1]*100)

        init_train_loss_LSTM = (history_LSTM.history['loss'][0]*100)

        y_pred_LSTM = model_LSTM.predict(X_to_predict)

        y_pred_noscale_LSTM = (y_pred_LSTM - scaler.min_[-1]) / scaler.scale_[-1]

        yhat_LSTM = model_LSTM.predict(X_scaled)
        y_pred_noscale_LSTM[:, 0][df_final["PH"][window_size:].astype('str') != 'nan'] = df_final["PH"][window_size:][df_final["PH"][window_size:].astype('str') != 'nan'].values

        time_delta = df_final["Time"][window_size:] - df_final["Time"][window_size]

        time_years = np.array([(item / np.timedelta64(1, 'm')) / (60*24*365) for item in time_delta])

        Y_TREND = y_pred_noscale_LSTM[y_pred_noscale_LSTM.astype('str') != 'nan']
        X_TREND = time_years[y_pred_noscale_LSTM[:, 0].astype('str') != 'nan']

        reg = LinearRegression().fit(X_TREND.reshape(-1,1), Y_TREND)

        slope_LSTM = reg.coef_[0]
        intercept_LSTM = reg.intercept_

        if final_train_loss_LSTM < tol:

            print(N)

            N += 1

            slopes.append(slope_LSTM)
            intercepts.append(intercept_LSTM)

            train_errors.append(final_train_loss_LSTM)
            val_errors.append(final_val_loss_LSTM)

            N_epochs.append(epochs_used_LSTM)
            training_time.append(time_elapsed_LSTM)
            
    return slopes, intercepts, train_errors, val_errors, N_epochs, training_time

def BI_LSTM_NN(Nf=100, tol=0.8):

    final_train_loss_BI_LSTM = 1000.0

    train_errors = []
    val_errors = []
    N_epochs = []
    training_time = []

    slopes = []
    intercepts = []

    N = 0
    tol = 0.8

    while N < Nf:

        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2, min_delta=0.00001)
        callback_2 = StopCrossPoint()     

        t0 = time()

        # design network
        model_BI_LSTM = Sequential()
        model_BI_LSTM.add(Bidirectional(LSTM(3, activation='tanh',
                                      input_shape=(X_train.shape[1], X_train.shape[2]))))
        model_BI_LSTM.add(Dense(1, activation='sigmoid'))
        model_BI_LSTM.compile(loss='mse', optimizer='adam')

        # fit network
        history_BI_LSTM = model_BI_LSTM.fit(
            X_train, 
            Y_train, 
            epochs=500,
            steps_per_epoch=10,
            #batch_size=100, 
            validation_data=(X_test, Y_test), 
            verbose=0, 
            shuffle=False,
            callbacks=[callback, callback_2]
            )

        time_elapsed_BI_LSTM =  time()-t0
        epochs_used_BI_LSTM = len(history_BI_LSTM.history['loss'])

        final_train_loss_BI_LSTM = (history_BI_LSTM.history['loss'][-1]*100)
        final_val_loss_BI_LSTM = (history_BI_LSTM.history['val_loss'][-1]*100)

        init_train_loss_BI_LSTM = (history_BI_LSTM.history['loss'][0]*100)

        y_pred_BI_LSTM = model_BI_LSTM.predict(X_to_predict)

        y_pred_noscale_BI_LSTM = (y_pred_BI_LSTM - scaler.min_[-1]) / scaler.scale_[-1]

        yhat_BI_LSTM = model_BI_LSTM.predict(X_scaled)
        y_pred_noscale_BI_LSTM[:, 0][df_final["PH"][window_size:].astype('str') != 'nan'] = df_final["PH"][window_size:][df_final["PH"][window_size:].astype('str') != 'nan'].values

        time_delta = df_final["Time"][window_size:] - df_final["Time"][window_size]

        time_years = np.array([(item / np.timedelta64(1, 'm')) / (60*24*365) for item in time_delta])

        Y_TREND = y_pred_noscale_BI_LSTM[y_pred_noscale_BI_LSTM.astype('str') != 'nan']
        X_TREND = time_years[y_pred_noscale_BI_LSTM[:, 0].astype('str') != 'nan']

        reg = LinearRegression().fit(X_TREND.reshape(-1,1), Y_TREND)

        slope_BI_LSTM = reg.coef_[0]
        intercept_BI_LSTM = reg.intercept_

        if final_train_loss_BI_LSTM < tol:

            print(N)

            N += 1

            slopes.append(slope_BI_LSTM)
            intercepts.append(intercept_BI_LSTM)

            train_errors.append(final_train_loss_BI_LSTM)
            val_errors.append(final_val_loss_BI_LSTM)

            N_epochs.append(epochs_used_BI_LSTM)
            training_time.append(time_elapsed_BI_LSTM)
            
    return slopes, intercepts, train_errors, val_errors, N_epochs, training_time

def BI_GRU_NN(Nf=100, tol=0.8):

    final_train_loss_BI_GRU = 1000.0

    train_errors = []
    val_errors = []
    N_epochs = []
    training_time = []

    slopes = []
    intercepts = []

    N = 0
    tol = 0.8

    while N < Nf:

        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2, min_delta=0.00001)
        callback_2 = StopCrossPoint()     

        t0 = time()

        # design network
        model_BI_GRU = Sequential()
        model_BI_GRU.add(Bidirectional(GRU(1, activation='tanh',
                                      input_shape=(X_train.shape[1], X_train.shape[2]))))
        model_BI_GRU.add(Dense(1, activation='sigmoid'))
        model_BI_GRU.compile(loss='mse', optimizer='adam')

        # fit network
        history_BI_GRU = model_BI_GRU.fit(
            X_train, 
            Y_train, 
            epochs=500,
            steps_per_epoch=10,
            #batch_size=100, 
            validation_data=(X_test, Y_test), 
            verbose=0, 
            shuffle=False,
            callbacks=[callback, callback_2]
            )

        time_elapsed_BI_GRU =  time()-t0
        epochs_used_BI_GRU = len(history_BI_GRU.history['loss'])

        final_train_loss_BI_GRU = (history_BI_GRU.history['loss'][-1]*100)
        final_val_loss_BI_GRU = (history_BI_GRU.history['val_loss'][-1]*100)

        init_train_loss_BI_GRU = (history_BI_GRU.history['loss'][0]*100)

        y_pred_BI_GRU = model_BI_GRU.predict(X_to_predict)

        y_pred_noscale_BI_GRU = (y_pred_BI_GRU - scaler.min_[-1]) / scaler.scale_[-1]

        yhat_BI_GRU = model_BI_GRU.predict(X_scaled)
        y_pred_noscale_BI_GRU[:, 0][df_final["PH"][window_size:].astype('str') != 'nan'] = df_final["PH"][window_size:][df_final["PH"][window_size:].astype('str') != 'nan'].values

        time_delta = df_final["Time"][window_size:] - df_final["Time"][window_size]

        time_years = np.array([(item / np.timedelta64(1, 'm')) / (60*24*365) for item in time_delta])

        Y_TREND = y_pred_noscale_BI_GRU[y_pred_noscale_BI_GRU.astype('str') != 'nan']
        X_TREND = time_years[y_pred_noscale_BI_GRU[:, 0].astype('str') != 'nan']

        reg = LinearRegression().fit(X_TREND.reshape(-1,1), Y_TREND)

        slope_BI_GRU = reg.coef_[0]
        intercept_BI_GRU = reg.intercept_
        
        if final_train_loss_BI_GRU < tol:

            print(N)

            N += 1

            slopes.append(slope_BI_GRU)
            intercepts.append(intercept_BI_GRU)

            train_errors.append(final_train_loss_BI_GRU)
            val_errors.append(final_val_loss_BI_GRU)

            N_epochs.append(epochs_used_BI_GRU)
            training_time.append(time_elapsed_BI_GRU)
            
    return slopes, intercepts, train_errors, val_errors, N_epochs, training_time

###########################################################################################
#                                   PREPROCESSING                                         #
###########################################################################################

window_size = 6

#df_f = pd.read_csv("Datos pH Baleares/Palma_resampled_dayly.csv")
df_f = pd.read_csv("Datos pH Baleares/Corrected_dataset_Palma.csv")

df_f["Time"] = pd.to_datetime(df_f["Time"])

#Delete red points
df_final = df_f.drop(df_f[(df_f["Time"] > datetime(2019, 8, 15)) & (df_f["Time"] < datetime(2019, 9, 1))].index)

df_final = df_final.drop(df_final[(df_final["Time"] > datetime(2020, 6, 15)) & (df_final["Time"] < datetime(2020, 7, 1))].index)

#data_resampled = df_final[df_final["DO(umol kg-1)"].astype('str') != 'nan'][["Tempertaure (ºC)", "DO(umol kg-1)", "pHT"]].values
data_resampled = df_final[df_final["Oxygen"].astype('str') != 'nan'][["Temperature", "Oxygen", "Salinity", "PH"]].values

X_train, X_test, Y_train, Y_test, scaler, X_scaled, Y_scaled = preprocessing(data_resampled, n_times=window_size, test_size=0.1)

#data_new = df_final[["Tempertaure (ºC)", "DO(umol kg-1)"]].values
data_new = df_final[["Temperature", "Oxygen", "Salinity"]].values

scaled_new = scaler.min_[0:data_new.shape[-1]] + data_new * scaler.scale_[0:data_new.shape[-1]]

n_features = data_new.shape[-1]

X_to_predict = RNN_dataset_pred(scaled_new, window_size, n_features)

###########################################################################################
#                                   PREDICTION                                            #
###########################################################################################

Nf = 100

type_ = str(sys.argv[1])

number = int(sys.argv[2])

if type_ == "RNN":

    slopes, intercepts, train_errors, val_errors, N_epochs, training_time = RNN(Nf)
    
elif type_ == "LSTM":
    
    slopes, intercepts, train_errors, val_errors, N_epochs, training_time = LSTM_NN(Nf)
    
elif type_ == "BI_LSTM":
    
    slopes, intercepts, train_errors, val_errors, N_epochs, training_time = BI_LSTM_NN(Nf)
    
elif type_ == "BI_GRU":
    
    slopes, intercepts, train_errors, val_errors, N_epochs, training_time = BI_GRU_NN(Nf)
    
else:
    
    print("Undefined name")

header = "Slope Intercept Train_error Val_error N_epochs Training_time"

np.savetxt("%s_%s.txt" % (type_, number), 
           np.transpose([slopes, intercepts, train_errors, val_errors, N_epochs, training_time]),
           header=header)

