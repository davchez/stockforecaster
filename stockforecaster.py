############################################################################################
# Stock Price Forecaster using LSTM neural network model.
# Uses time series to formulate minimized loss model that predicts stock movements 20 days
# in advance.  Univariate and does not consider any factors outside of movement patterns.
# Can be adapted to any stock but data considers Rivian Stock Price from IPO to
# November 24, 2023 closing price with 80% training data 20% testing data.
#
# Credit: Kite on Youtube
# Bugs: None discovered
# 
# @author David Sanchez (@davchez on GitHub)
############################################################################################

import pandas as pd
import numpy as np

#### SECTION 1: REFORMATTING DATA SO THAT NEURAL NETWORK CAN READ ####

RAW = pd.read_csv( 'Rivian_Stock_Data.csv' )
RAW = ( RAW[ [ 'date', 'close' ] ] ).drop( 'date', axis = 1 )
RAW = RAW.reset_index( drop = True )

data = RAW.astype( 'float32' )
data = np.reshape( data, (-1, 1) )

#### SECTION 2: CREATING METHOD WHICH CREATES SLIDING WINDOW ####

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler( feature_range = ( 0, 1 ) )
data = scaler.fit_transform( data )

DATA_LENGTH = len( data )

train_length = int( DATA_LENGTH * 0.80 ) 
test_length = int( DATA_LENGTH - train_length )

#Personal note: a:b, a, b are standard indexing; after , is column specificity

train_80 = data[ 0:train_length ] #research later
test_20 = data[ train_length:DATA_LENGTH ] #research later

def slidingWindow( input, window_size ):

    WINDOW, TARGET = [], []

    for i in range( ( len( input ) - 1 ) - window_size ):
        window = input[ i:( i + window_size ), 0 ]
        WINDOW.append( window )
        TARGET.append( input[ i + window_size, 0 ] )

    return np.array( WINDOW ), np.array( TARGET )

WINDOW_SIZE = 20

training_window, training_target = slidingWindow( train_80, WINDOW_SIZE )
testing_window, testing_target = slidingWindow( test_20, WINDOW_SIZE )

training_window = np.reshape( training_window, 
                             ( training_window.shape[ 0 ], 1, training_window.shape[ 1 ] ) )

testing_window = np.reshape ( testing_window,
                             ( testing_window.shape[ 0 ], 1, testing_window.shape[ 1 ] ) )

#### SECTION 3: CHECKING FOR DATA LEAKS BETWEEN THE TRAINING AND TESTING WINDOWS ####

data_shape = data.shape
training_shape = train_80.shape
testing_shape = test_20.shape

def dataLeakOccured( data_shape, training_shape, testing_shape ):
    return not( data_shape[0] == ( training_shape[ 0 ] + testing_shape[ 0 ] ) )

print( dataLeakOccured( data_shape, training_shape, testing_shape ) )

#### SECTION 4: PERFORMING NEURAL NETWORK PREDICTION ####

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import ModelCheckpoint

CONSTANT = 12
STD_DROPOUT_LAYER = 0.2

tf.random.set_seed( CONSTANT )
np.random.seed( CONSTANT )

model = Sequential()

model.add( LSTM ( units = 50,
                 activation = 'relu',
                 input_shape = ( training_window.shape[ 1 ], WINDOW_SIZE ) ) )

model.add( Dropout ( STD_DROPOUT_LAYER ) )
model.add( Dense( 1 ) )
model.compile( loss = 'mean_squared_error' )

filepath = 'saved_models/model_epoch_{epoch:02d}.hdf5'

checkpoint = ModelCheckpoint( filepath = filepath, 
                              monitor = 'val_loss', 
                              verbose = 1,
                              save_best_only = True,
                              mode = 'min' )

history = model.fit( training_window, training_target,
                     epochs = 100,
                     batch_size = 20,
                     validation_data = ( testing_window, testing_target ),
                     callbacks = [ checkpoint ],
                     verbose = 1,
                     shuffle = False )

#### SECTION 5: CHECKING FOR THE BEST EPOCH MINIMIZING LOSS ####

TRAINING_LOSS = history.history[ 'loss' ]
VALIDATION_LOSS = history.history[ 'val_loss' ]
optimal_epoch_index = np.argmin( VALIDATION_LOSS ) + 1
optimal_val_loss = VALIDATION_LOSS[ optimal_epoch_index ]

optimal_epoch_num = str( optimal_epoch_index - 1 ) 

#### SECTION 6: PERFORMING PREDICTIONS ####

from keras.models import load_model

optimal_model = load_model( 'saved_models/model_epoch_' + optimal_epoch_num + '.hdf5' )

training_prediction = optimal_model.predict( training_window )
training_prediction = scaler.inverse_transform( training_prediction )
training_prediction = np.reshape( training_prediction, newshape = training_prediction.shape[ 0 ] )

testing_prediction = optimal_model.predict( testing_window )
testing_prediction = scaler.inverse_transform( testing_prediction )
testing_prediction = np.reshape( testing_prediction, newshape = testing_prediction.shape[ 0 ] )

testing_target = scaler.inverse_transform( [ testing_target ] )
testing_target = np.reshape( testing_target, newshape = testing_prediction.shape[ 0 ] )

training_target = scaler.inverse_transform( [ training_target] )
training_target = np.reshape( training_target, newshape = training_prediction.shape[ 0 ] )

#### SECTION 7: CALCULATING LOSS ####

from sklearn.metrics import mean_squared_error

training_RMSE = np.sqrt( mean_squared_error( training_target, training_prediction ) )
testing_RMSE = np.sqrt( mean_squared_error( testing_target, testing_prediction ) )

print('Training RMSE is: ' + str( training_RMSE ) + '\n')
print('Testing RMSE is: ' + str( testing_RMSE ) + '\n')

#### SECTION 8: PREPARING FOR FUTURE PREDICTION OF 20 DAY INCREMENT ####

last_window = data[ -WINDOW_SIZE: ].reshape( ( 1, 1, WINDOW_SIZE ) )

future = []
current_window = last_window

DAYS_AHEAD = 20

for _ in range( DAYS_AHEAD ):

    next_price = model.predict( current_window )
    future.append( next_price[ 0, 0 ] )

    current_window = np.roll( current_window, -1, axis = 2 )
    current_window[ 0, 0, -1 ] = next_price

future = np.array( future ).reshape( -1, 1 )
future = scaler.inverse_transform( future )

#### SECTION 9: PLOTTING THE RESULTS OF MODELING ####

import matplotlib.pyplot as plt

stock_prices = np.concatenate( ( training_target, testing_target ), axis = 0 )
model_prices = np.concatenate( ( training_prediction, testing_prediction ), axis = 0 )
forecasted_prices = np.concatenate( ( model_prices, future.squeeze() ), axis = 0 )

STOCK_DAYS = range( len( stock_prices ) )
FORECAST_DAYS = range( len( stock_prices ) + DAYS_AHEAD )

plt.figure( figsize = ( 12, 6 ) )
plt.plot( STOCK_DAYS, stock_prices, label = 'Stock Prices in USD', color = 'green' )
plt.plot( FORECAST_DAYS, forecasted_prices, label = 'Forecasted Prices in USD', color = 'red' )

plt.title( 'Stock Price Prediction using Long Short-Term Memory Neural Network Model' )
plt.xlabel( 'Days' )
plt.ylabel( 'Stock Price in USD ($)' )
plt.figtext( 0.5, 0.01, 'Training loss USD (RMSE): ' + str( training_RMSE.round(3) ) + 
             '. Testing loss USD (RMSE): ' + str( testing_RMSE.round(3) ) + '.', 
             ha = 'center', fontsize = 10 )
plt.legend()
plt.show()