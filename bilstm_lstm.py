# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 15:15:02 2021

@author: IIT HYD
"""


import scipy
import numpy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers, callbacks
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt
from IPython.core.pylabtools import figsize

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import seaborn as sns





data = pd.read_excel("E:\\Objective_1\\objective_1-20201119T105917Z-003\\objective_1\\data_1\\Anomalies\\Gracesatellitedates\\Columnwise_data\\entiredata_column@1.xlsx")





features = ['CWSA','ET','LST','NDVI','BASEFLOW','SM','RAINF','AIR_TEMP','EVAP_CANOPY','EVAPORATION','RAINF','PRECP','Surface_runoff']





x = data.loc[:,features].values





y = data.loc[:,['TWSA']].values





x = StandardScaler().fit_transform(x)





from sklearn.decomposition import PCA


pca = PCA(n_components = 4)
principalcomponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalcomponents)
finalDF = pd.concat([principalDf, data[['TWSA']]],axis = 1)
pca.explained_variance_ratio_


features = finalDF




features.shape





# Set random seed for reproducibility
tf.random.set_seed(1234)




#file = 'TRAIL_SETUP.xlsx'
#raw_data = pd.read_excel(file, parse_dates = ['Date'], index_col = 'Date')
#
#df = raw_data.copy()








# Explore the first five rows
#df.head()
#
#
#
#
##Data description
#df.describe()
#
#
#
#
## Find number of rows and columns
#df.shape





# Define a function to draw time_series plot
#def timeseries (x_axis, y_axis, x_label, y_label):
#    plt.figure(figsize = (10, 6))
#    plt.plot(x_axis, y_axis, color ='black')
#    plt.xlabel(x_label, {'fontsize': 12})
#    plt.ylabel(y_label, {'fontsize': 12})
#    #plt.savefig('C:/Users/nious/Documents/Medium/LSTM&GRU/1.jpg', format='jpg', dpi=1000)
#
#timeseries(df.index, df['TWSA'], 'Time (Month)', 
#           'Terrestrial water storage ($m^3$/capita.day)')
#
#



#def plot_histogram(x):
#    plt.hist(x, bins = 19, alpha=0.8, color = 'gray', edgecolor = 'black')
#    plt.title("Histogram of '{var_name}'".format(var_name=x.name))
#    plt.xlabel("Value")
#    plt.ylabel("Frequency")
#
#plot_histogram(df['TWSA'])
#
#



#def plot_histogram(x):
#    plt.hist(x, bins = 19, alpha=0.8, color = 'gray', edgecolor = 'black')
#    plt.title("Histogram of '{var_name}'".format(var_name=x.name))
#    plt.xlabel("Value")
#    plt.ylabel("Frequency")
#
#plot_histogram(df['NDVI'])
#
#


train_size = int(len(finalDF)*0.8)
train_dataset, test_dataset = finalDF.iloc[:train_size], finalDF.iloc[train_size:]

# Plot train and test data
plt.figure(figsize = (10, 6))
plt.plot(train_dataset.TWSA)
plt.plot(test_dataset.TWSA)
plt.xlabel('Time (MONTH)')
plt.ylabel('Terrestrial Water storage anomalies ($m^3$/capita.day)')
plt.legend(['Train set', 'Test set'], loc='upper right')
#plt.savefig('C:/Users/nious/Documents/Medium/LSTM&GRU/2.jpg', format='jpg', dpi=1000)

print('Dimension of train data: ',train_dataset.shape)
print('Dimension of test data: ', test_dataset.shape)





train_dataset.head()





# Split train data to X and y
X_train = train_dataset.drop('TWSA', axis = 1)
y_train = train_dataset.loc[:,['TWSA']]

# Split test data to X and y
X_test = test_dataset.drop('TWSA', axis = 1)
y_test = test_dataset.loc[:,['TWSA']]



len(X_train)




# Transform X_train, y_train, X_test and y_test

# Different scaler for input and output
scaler_x = MinMaxScaler(feature_range = (0,1))
scaler_y = MinMaxScaler(feature_range = (0,1))

# Fit the scaler using available training data
input_scaler = scaler_x.fit(X_train)
output_scaler = scaler_y.fit(y_train)

# Apply the scaler to training data
train_y_norm = output_scaler.transform(y_train)
train_x_norm = input_scaler.transform(X_train)

# Apply the scaler to test data
test_y_norm = output_scaler.transform(y_test)
test_x_norm = input_scaler.transform(X_test)




def create_dataset (X, y, time_steps = 1):
    Xs, ys = [], []
    
    for i in range(len(X)-time_steps):
        v = X[i:i+time_steps, :]
        Xs.append(v)
        ys.append(y[i+time_steps])
        
    return np.array(Xs), np.array(ys)


TIME_STEPS = 1

X_test, y_test = create_dataset(test_x_norm, test_y_norm, TIME_STEPS)
X_train, y_train = create_dataset(train_x_norm, train_y_norm, TIME_STEPS)
print('X_train.shape: ', X_train.shape)
print('y_train.shape: ', y_train.shape)
print('X_test.shape: ', X_test.shape) 
print('y_test.shape: ', y_test.shape)




# Create BiLSTM model
def create_model_bilstm(units):
    model = Sequential()
    # First layer of BiLSTM
    model.add(Bidirectional(LSTM(units = units, return_sequences=True), 
                            input_shape=(X_train.shape[1], X_train.shape[2])))
    # Second layer of BiLSTM
    model.add(Bidirectional(LSTM(units = units)))
    model.add(Dense(1))
    #Compile model
    model.compile(loss='mse', optimizer='adam')
    return model


# Create LSTM or GRU model
def create_model(units, m):
    model = Sequential()
    # First layer of LSTM
    model.add(m (units = units, return_sequences = True, 
                 input_shape = [X_train.shape[1], X_train.shape[2]]))
    model.add(Dropout(0.2)) 
    # Second layer of LSTM
    model.add(m (units = units))                 
    model.add(Dropout(0.2))
    model.add(Dense(units = 1)) 
    #Compile model
    model.compile(loss='mse', optimizer='adam')
    return model


# BiLSTM
model_bilstm = create_model_bilstm(64)

# GRU and LSTM 
model_gru = create_model(64, GRU)
model_lstm = create_model(64, LSTM)





# Fit BiLSTM, LSTM and GRU
def fit_model(model):
    early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                               patience = 10)

    # shuffle = False because the order of the data matters
    history = model.fit(X_train, y_train, epochs = 1000, validation_split = 0.2,
                    batch_size = 32, shuffle = False, callbacks = [early_stop])
    return history

history_bilstm = fit_model(model_bilstm)
history_lstm = fit_model(model_lstm)
history_gru = fit_model(model_gru)





def plot_loss (history, model_name):
    plt.figure(figsize = (10, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Train vs Validation Loss for ' + model_name)
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['Train loss', 'Validation loss'], loc='upper right')
    #plt.savefig('C:/Users/nious/Documents/Medium/LSTM&GRU/loss_'+model_name+'.jpg', format='jpg', dpi=1000)

plot_loss (history_bilstm, 'BiLSTM')
plot_loss (history_lstm, 'LSTM')
plot_loss (history_gru, 'GRU')




# Note that I have to use scaler_y
y_test = scaler_y.inverse_transform(y_test)
y_train = scaler_y.inverse_transform(y_train)





#pd.DataFrame(y_test).to_excel("testingdataset.xlsx")




def prediction(model):
    prediction = model.predict(X_test)
    prediction = scaler_y.inverse_transform(prediction)
    return prediction

prediction_bilstm = prediction(model_bilstm)
prediction_lstm = prediction(model_lstm)
prediction_gru = prediction(model_gru)




#pd.DataFrame(prediction_bilstm).to_excel("bilstmpred.xlsx")


from sklearn.metrics import r2_score
cor_bilstm = r2_score(y_test,prediction_bilstm)
cor_lstm = r2_score(y_test,prediction_lstm)
cor_gru = r2_score(y_test,prediction_gru)




def plot_future(prediction, model_name, y_test):
    
    plt.figure(figsize=(10, 6))
    
    range_future = len(prediction)

    plt.plot(np.arange(range_future), np.array(y_test), label='True Future')
    plt.plot(np.arange(range_future), np.array(prediction),label='Prediction')

    plt.title('True future vs prediction for ' + model_name)
    plt.legend(loc='upper left')
    plt.xlabel('Time (day)')
    plt.ylabel('Terrestrial Water Storage ($m^3$/capita.day)')
    #plt.savefig('C:/Users/nious/Documents/Medium/LSTM&GRU/predic_'+model_name+'.jpg', format='jpg', dpi=1000)
    
    
plot_future(prediction_bilstm, 'BiLSTM', y_test)
plot_future(prediction_lstm, 'LSTM', y_test)
plot_future(prediction_gru, 'GRU', y_test)




# Define a function to calculate MAE and RMSE
def evaluate_prediction(predictions, actual, model_name):
    errors = predictions - actual
    mse = np.square(errors).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(errors).mean()

    print(model_name + ':')
    print('Mean Absolute Error: {:.4f}'.format(mae))
    print('Root Mean Square Error: {:.4f}'.format(rmse))
    print('')


evaluate_prediction(prediction_bilstm, y_test, 'Bidirectional LSTM')
evaluate_prediction(prediction_lstm, y_test, 'LSTM')
evaluate_prediction(prediction_gru, y_test, 'GRU')


#newinput = pd.read_excel('TRAIL_VALID.xlsx', parse_dates=['Date'], index_col = 'Date')

#X_new = newinput.loc['2002-01-01':'2017-12-01',:] 
#X_new


X_new = pd.read_excel("E:\\Objective_1\\objective_1-20201119T105917Z-003\\objective_1\\data_quarter\\Krishna\\Columnwisearrange\\Anomalies\\Dropna\\columnwise_data\\entiredata_column.xlsx")



X_new.shape



features = ['CWSA','ET','LST','NDVI','BASEFLOW','SM','RAINF','AIR_TEMP','EVAP_CANOPY','EVAPORATION','RAINF','PRECP','Surface_runoff']


X_n = X_new.loc[:,features].values
X_n =StandardScaler().fit_transform(X_n)

from sklearn.decomposition import PCA

pca1 = PCA(n_components = 4)
principalcomponents1 = pca1.fit_transform(X_n)

principalDf1 = pd.DataFrame(data = principalcomponents1)
finalDF1 = pd.concat([principalDf1, data[['TWSA']]],axis = 1)
pca1.explained_variance_ratio_


features = finalDF1

def plot_history_future(y_train, prediction, model_name):
    
    plt.figure(figsize=(10, 6))
    
    range_history = len(y_train)
    range_future = list(range(range_history, range_history + len(prediction)))

    plt.plot(np.arange(range_history), np.array(y_train), label='History')
    plt.plot(range_future, np.array(prediction),label='Prediction')

    plt.title('History and prediction for ' + model_name)
    plt.legend(loc='upper right')
    plt.xlabel('Data Length')
    plt.ylabel('Terrestrial Water Storage Anomalies (cm)')
    #plt.savefig('C:/Users/nious/Documents/Medium/LSTM&GRU/3.jpg', format='jpg', dpi=1000)

def forecast(X_input, time_steps):
    # Scale the unseen input with the scaler fitted on the training data
    X = input_scaler.transform(X_input)
    # Reshape unseen data to a 3D input
    Xs = []
    for i in range(len(X) - time_steps):
        v = X[i:i+time_steps, :]
        Xs.append(v)
        

    X_transformed = np.array(Xs)

    # Make prediction for unseen data using LSTM model 
    prediction = model_bilstm.predict(X_transformed)
    prediction_actual = scaler_y.inverse_transform(prediction)
    return prediction_actual
    

    prediction1 = model_lstm.predict(X_transformed)
    prediction_actual1 = scaler_y.inverse_transform(prediction1)
    return prediction_actual1

    prediction2 = model_gru.predict(X_transformed)
    prediction_actual2 = scaler_y.inverse_transform(prediction2)
    return prediction_actual2

prediction = forecast(principalDf1, TIME_STEPS)
plot_history_future(y_train, prediction,'BiLSTM')

prediction1 = forecast(principalDf1, TIME_STEPS)
plot_history_future(y_train, prediction1,'LSTM')

prediction2 = forecast(principalDf1, TIME_STEPS)
plot_history_future(y_train, prediction2,'GRU')


from sklearn.metrics import r2_score
cor = r2_score(y_test,prediction)

print(cor)


from sklearn.metrics import r2_score
cor1 = r2_score(y_test,prediction1)

print(cor1)

from sklearn.metrics import r2_score
cor2 = r2_score(y_test,prediction2)

print(cor2)

#pd.DataFrame(y_test).to_excel("y_test.xlsx")
#
#
#
#pd.DataFrame(prediction).to_excel("pred_Well.xlsx")

