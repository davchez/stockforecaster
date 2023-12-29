# Title: Stock Price Predictor
# Author: David Sanchez
# GitHub: https://github.com/davchez
# 
# Desc: Machine learning Python project that takes advantage of neural network capabilities 
# of the Tensorflow and Keras packages. Formats historic stock prices into time series data 
# to train long short-term memory model. Predicts stock price movement 
# x amount of days into the future; generally works best with 20 days predictions.
#
# Bugs: None discovered
#
# Credit: Caelan from Kite on Youtube
# Assistance in developing code for neural network model

import pandas as pd
import numpy as np

RAW = pd.read_csv( 'stockdata/AAPL_Stock_Data.csv' )
RAW = ( RAW[ [ 'date', 'close' ] ] ).drop( 'date', axis = 1 )
RAW = RAW.reset_index( drop = True )

data = RAW.astype( 'float32' )
data = np.reshape( data, (-1, 1 ) )

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler( feature_range = ( 0, 1 ) )
data = scaler.fit_transform( data )

DATA_LENGTH = len( data )

train_length = int( DATA_LENGTH * 0.80 ) 
test_length = int( DATA_LENGTH - train_length )

train_80 = data[ 0:train_length ]
test_20 = data[ train_length:DATA_LENGTH ]

def slidingWindow( input, window_size ):

    WINDOW, TARGET = [], []

    for i in range( ( len( input ) - 1 ) - window_size ):
        window = input[ i:( i + window_size ), 0 ]
        WINDOW.append( window )
        TARGET.append( input[ i + window_size, 0 ] )

    return np.array( WINDOW ), np.array( TARGET )

WINDOW_SIZE = 20

train_window, train_target = slidingWindow( train_80, WINDOW_SIZE )
test_window, test_target = slidingWindow( test_20, WINDOW_SIZE )

train_window = np.reshape( train_window, 
                            ( train_window.shape[ 0 ], 1, train_window.shape[ 1 ] ) )

test_window = np.reshape( test_window,
                           ( test_window.shape[ 0 ], 1, test_window.shape[ 1 ] ) )

data_shape = data.shape
train_shape = train_80.shape
test_shape = test_20.shape

def dataLeakOccured( merge_shape, shape_1, shape_2 ):
    return not( merge_shape[0] == ( shape_1[ 0 ] + shape_2[ 0 ] ) )

print( dataLeakOccured( data_shape, train_shape, test_shape ) )

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
                 input_shape = ( train_window.shape[ 1 ], WINDOW_SIZE ) ) )

model.add( Dropout ( STD_DROPOUT_LAYER ) )
model.add( Dense( 1 ) )
model.compile( loss = 'mean_squared_error' )

filepath = 'saved_models/model_epoch_{epoch:02d}.hdf5'

checkpoint = ModelCheckpoint( filepath = filepath, 
                              monitor = 'val_loss', 
                              verbose = 1,
                              save_best_only = True,
                              mode = 'min' )

history = model.fit( train_window, train_target,
                     epochs = 100,
                     batch_size = 20,
                     validation_data = ( test_window, test_target ),
                     callbacks = [ checkpoint ],
                     verbose = 1,
                     shuffle = False )

train_loss = history.history[ 'loss' ]
val_loss = history.history[ 'val_loss' ]
optimal_epoch_index = val_loss.index( min( val_loss ) ) + 1
optimal_val_loss = val_loss[ optimal_epoch_index - 1 ]
min_val_loss = [ optimal_epoch_index, optimal_val_loss ]

if optimal_epoch_index < 10:
    optimal_epoch_num = "0" + str( optimal_epoch_index ) 
else:
    optimal_epoch_num = str( optimal_epoch_index ) 

from keras.models import load_model

optimal_model = load_model( 'saved_models/model_epoch_' + optimal_epoch_num + '.hdf5' )

train_prediction = optimal_model.predict( train_window )
train_prediction = scaler.inverse_transform( train_prediction )
train_prediction = np.reshape( train_prediction, newshape = train_prediction.shape[ 0 ] )

test_prediction = optimal_model.predict( test_window )
test_prediction = scaler.inverse_transform( test_prediction )
test_prediction = np.reshape( test_prediction, newshape = test_prediction.shape[ 0 ] )

test_target = scaler.inverse_transform( [ test_target ] )
test_target = np.reshape( test_target, newshape = test_prediction.shape[ 0 ] )

train_target = scaler.inverse_transform( [ train_target ] )
train_target = np.reshape( train_target, newshape = train_prediction.shape[ 0 ] )

from sklearn.metrics import mean_squared_error

train_RMSE = np.sqrt( mean_squared_error( train_target, train_prediction ) )
test_RMSE = np.sqrt( mean_squared_error( test_target, test_prediction ) )

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

stock_prices = np.concatenate( ( train_target, test_target ), axis = 0 )
model_prices = np.concatenate( ( train_prediction, test_prediction ), axis = 0 )
forecasted_prices = np.concatenate( ( model_prices, future.squeeze( ) ), axis = 0 )

STOCK_DAYS = range( len( stock_prices ) )
FORECAST_DAYS = range( len( stock_prices ) + DAYS_AHEAD )
EPOCHS_RANGE = range( 1, len( train_loss ) + 1 )

import matplotlib.pyplot as plt

fig, ( stock_plt, loss_plt ) = plt.subplots( 1, 2, figsize = ( 18, 6 ) )

stock_plt.plot( STOCK_DAYS, stock_prices, label = 'Stock Prices in USD', color = 'green' )
stock_plt.plot( FORECAST_DAYS, forecasted_prices, label = 'Forecasted Prices in USD', color = 'red' )
stock_plt.set_title( str( DAYS_AHEAD ) + '-Day Stock Price Prediction' )
stock_plt.set_xlabel( 'Days' )
stock_plt.set_ylabel( 'Stock Price in USD' )
stock_plt.legend( )

loss_plt.plot( EPOCHS_RANGE, train_loss, label = 'Training Loss', color = 'orange' )
loss_plt.plot( EPOCHS_RANGE, val_loss, label = 'Validation Loss', color = 'blue' )
loss_plt.scatter( *min_val_loss, 
                  label = 'Optimal Epoch', marker = 'o', color = 'red' )
loss_plt.set_title( 'Training and Validation Loss' )
loss_plt.set_xlabel( 'Epochs' )
loss_plt.set_ylabel( 'Loss' )
loss_plt.legend( )

plt.figtext( 0.5, 0.01, 'Training loss USD (RMSE): ' + str( train_RMSE.round( 3 ) ) + 
             '. Testing loss USD (RMSE): ' + str( test_RMSE.round( 3 ) ) +
             '. Optimal epoch val loss: ' + str( round( min_val_loss[ 1 ], 5 ) ), 
             ha = 'center', fontsize = 8 )
plt.show( )