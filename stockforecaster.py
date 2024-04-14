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

# Imports data, reads a comma separated value sheet
RAW = pd.read_csv( 'stockdata/NVDA_Stock_Data.csv' )

# Isolated date and close columns, but drops the date column (axis = 1)
RAW = ( RAW[ [ 'date', 'close' ] ] ).drop( 'date', axis = 1 )

# Removes original indices so that only the raw indices for the prices remain (just prices)
RAW = RAW.reset_index( drop = True )

# Converts all price data to 32 digit float values
data = RAW.astype( 'float32' )

# Reshapes the data variable so that it allows as many rows as possible but formats it into one column
data = np.reshape( data, (-1, 1 ) )

from sklearn.preprocessing import MinMaxScaler

# Reshapes data into variable "scaler" so that it shrinks all values of data into range between 0 and 1
scaler = MinMaxScaler( feature_range = ( 0, 1 ) )

# Calculates the minimum and maximum values in data, then scales it into a range between 0 and 1
data = scaler.fit_transform( data )

DATA_LENGTH = len( data )

# Separating the training data of the program into the first 80% of the price movement
train_length = int( DATA_LENGTH * 0.80 ) 

# Separating the validation/testing data of the program into the second 20% of the price movement
test_length = int( DATA_LENGTH - train_length )

# Implementing length
train_80 = data[ 0:train_length ]
test_20 = data[ train_length:DATA_LENGTH ]

# Separating data such that it creates time-series data which will be processed later on
def slidingWindow( input, window_size ):

    WINDOW, TARGET = [], []

    for i in range( ( len( input ) - 1 ) - window_size ):

        # Creates windows of size "WINDOW_SIZE"
        window = input[ i:( i + window_size ), 0 ]

        # Appends windows into WINDOW array
        WINDOW.append( window )
        
        # Appends a "target" data point which will be the 21st point (relatively) of each window
        TARGET.append( input[ i + window_size, 0 ] )

    # Creating numpy arrays WITHOUT INDICES which contain necessary data points (thus, returns tuples)
    return np.array( WINDOW ), np.array( TARGET )

WINDOW_SIZE = 20

# The first 80% of the data is parsed into 20 data points with an associated target point; this is repeated throughout the first 80% of the dataset
train_window, train_target = slidingWindow( train_80, WINDOW_SIZE )

# The same is made for the second 20% of the data
test_window, test_target = slidingWindow( test_20, WINDOW_SIZE )

# Reformatting train_window into time-series data: Each 20-day sliding window training sample is being classified into a single time step
train_window = np.reshape( train_window, 
                            ( train_window.shape[ 0 ], 1, train_window.shape[ 1 ] ) )

# Same idea
test_window = np.reshape( test_window,
                           ( test_window.shape[ 0 ], 1, test_window.shape[ 1 ] ) )

# Shape of the data variable, which SHOULD be the sum of the samples (rows) of the training + test division (80% + 20%)
data_shape = data.shape

# First 80% of the data
train_shape = train_80.shape

# Last 20% of the data
test_shape = test_20.shape

# Checks if any entry was repeated or missing as a result of the splitting (or, alternatively, an accidental merge): data leak
def dataLeakOccured( merge_shape, shape_1, shape_2 ):
    return not( merge_shape[0] == ( shape_1[ 0 ] + shape_2[ 0 ] ) )

# Prints false if data leak did not occur; true if otherwise
print( dataLeakOccured( data_shape, train_shape, test_shape ) )

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import ModelCheckpoint

# Constant seed for tensorflow and numpy model in order to reproduce results
CONSTANT = 12

# Dropped out 20% of neurons to prevent overfitting during each training phase
STD_DROPOUT_LAYER = 0.2

tf.random.set_seed( CONSTANT )
np.random.seed( CONSTANT )

model = Sequential()

# Model is a Long Short-Term Memory model using the ReLU Activation model to introduce non-linearity concept to training
model.add( LSTM ( units = 50,
                  activation = 'relu',
                  input_shape = ( train_window.shape[ 1 ], WINDOW_SIZE ) ) )

model.add( Dropout ( STD_DROPOUT_LAYER ) )
model.add( Dense( 1 ) )
model.compile( loss = 'mean_squared_error' )

filepath = 'saved_models/model_epoch_{epoch:02d}.hdf5'

# Model only remembers best training points and forgets lossy training data
checkpoint = ModelCheckpoint( filepath = filepath, 
                              monitor = 'val_loss', 
                              verbose = 1,
                              save_best_only = True,
                              mode = 'min' )

# History of all models which will be compiled into 100 generationally-learning epochs
history = model.fit( train_window, train_target,
                     epochs = 100,
                     batch_size = 20,
                     validation_data = ( test_window, test_target ),
                     callbacks = [ checkpoint ],
                     verbose = 1,
                     shuffle = False )

train_loss = history.history[ 'loss' ]
val_loss = history.history[ 'val_loss' ]

# Defined "optimal epoch" as the epoch that experienced the MINIMUM AMOUNT OF TRAINING LOSS (and only training loss, not validation loss)
# Also, + 1 as epochs are listed with 1-indexing
optimal_epoch_num = val_loss.index( min( val_loss ) ) + 1

# Kept for record-keeping: epoch index with minimum validation loss
optimal_val_loss = val_loss[ optimal_epoch_num - 1 ]

# Kept for record-keeping: minimum validation loss value
min_val_loss = [ optimal_epoch_num, optimal_val_loss ]

# Avoiding key errors
if optimal_epoch_num < 10:
    optimal_epoch_num = "0" + str( optimal_epoch_num ) 
else:
    optimal_epoch_num = str( optimal_epoch_num ) 

from keras.models import load_model

optimal_model = load_model( 'saved_models/model_epoch_' + optimal_epoch_num + '.hdf5' )

# Training phase movement of first 80% of data is predicted using optimal epoch
train_prediction = optimal_model.predict( train_window )

# Removes the standardizing scaler of [0, 1] to return prediction to original scaling [min, max]
train_prediction = scaler.inverse_transform( train_prediction )
train_prediction = np.reshape( train_prediction, newshape = train_prediction.shape[ 0 ] )

# Testing phase movement of remaining 20% of data is predicted using optimal epoch
test_prediction = optimal_model.predict( test_window )

# Removes the standardizing scale of [0, 1] to return prediction to original scaling [min, max]
test_prediction = scaler.inverse_transform( test_prediction )
test_prediction = np.reshape( test_prediction, newshape = test_prediction.shape[ 0 ] )

# Actual training data to be compared against, i.e. first 80% of original data set
# Removes the standardizing scaler of [0, 1] to return prediction to original scaling [min, max]
train_target = scaler.inverse_transform( [ train_target ] )
train_target = np.reshape( train_target, newshape = train_prediction.shape[ 0 ] )

# Actual validation data to be compared against, i.e. last 20% of original data set
# Removes the standardizing scaler of [0, 1] to return prediction to original scaling [min, max]
test_target = scaler.inverse_transform( [ test_target ] )
test_target = np.reshape( test_target, newshape = test_prediction.shape[ 0 ] )

from sklearn.metrics import mean_squared_error

# Root mean square error showing loss between actual training data and training prediction from optimal epoch
train_RMSE = np.sqrt( mean_squared_error( train_target, train_prediction ) )

# Root mean square error showing loss between actual validation data and validation prediction from optimal epoch
test_RMSE = np.sqrt( mean_squared_error( test_target, test_prediction ) )

# Moving into the future; no original data to compare with, attempting to predict movement 20 days in advance
# First comparing with the last 20 days of model prediction window (final 20 days of data set comparison) to begin modeling
last_window = data[ -WINDOW_SIZE: ].reshape( ( 1, 1, WINDOW_SIZE ) )

future = []

current_window = last_window

DAYS_AHEAD = 20

# Actual model prediction: Looking 20 days into the future and continunally predicting using a sliding window
# There is no training/validation phase here: This is all epoch prediction

for _ in range( DAYS_AHEAD ):

    # Predicting next price using the current 20-day window
    next_price = model.predict( current_window )

    # Prediction price is appended to the future list for storage
    future.append( next_price[ 0, 0 ] )

    # Current window is shifted one day in the future
    # i.e. if start index was k and end index was n, then new window is [k+1, n+1]
    current_window = np.roll( current_window, -1, axis = 2 )

    # Current window now contains model predicted price for "21st" day
    current_window[ 0, 0, -1 ] = next_price

future = np.array( future ).reshape( -1, 1 )

# Resizing future list so that it removes standardizing scale of [0, 1] to original size [min, max]
future = scaler.inverse_transform( future )

# Actual prices concatenated into one list
stock_prices = np.concatenate( ( train_target, test_target ), axis = 0 )

# Model training prices concatenated into one list
model_prices = np.concatenate( ( train_prediction, test_prediction ), axis = 0 )

# Predicted prices added onto model training prices to create +20 day prediction
forecasted_prices = np.concatenate( ( model_prices, future.squeeze( ) ), axis = 0 )

STOCK_DAYS = range( len( stock_prices ) )
FORECAST_DAYS = range( len( stock_prices ) + DAYS_AHEAD )
EPOCHS_RANGE = range( 1, len( train_loss ) + 1 )

import matplotlib.pyplot as plt

# Preparing data visualization for performance metrics and model prediction
fig, ( stock_plt, loss_plt ) = plt.subplots( 1, 2, figsize = ( 18, 6 ) )

stock_plt.plot( STOCK_DAYS, stock_prices, label = 'Stock Prices in USD', color = 'green' )
stock_plt.plot( FORECAST_DAYS, forecasted_prices, label = 'Forecasted Prices in USD', color = 'red' )
stock_plt.set_title( str( DAYS_AHEAD ) + '-Day Stock Price Prediction' )
stock_plt.set_xlabel( 'Days' )
stock_plt.set_ylabel( 'Stock Price in USD' )
stock_plt.legend( )

loss_plt.plot( EPOCHS_RANGE, train_loss, label = 'Training Loss', color = 'orange' )
loss_plt.plot( EPOCHS_RANGE, val_loss, label = 'Validation Loss', color = 'blue' )
loss_plt.axvline( x = min_val_loss[ 0 ], color = 'red', alpha = 0.2,
                  label = 'Optimal Epoch (epoch ' + str( optimal_epoch_num ) + ')' )
loss_plt.set_title( 'Training and Validation Loss' )
loss_plt.set_xlabel( 'Epochs' )
loss_plt.set_ylabel( 'Loss' )
loss_plt.legend( )

plt.figtext( 0.5, 0.01, 'Training loss USD (RMSE): ' + str( train_RMSE.round( 3 ) ) + 
             '. Testing loss USD (RMSE): ' + str( test_RMSE.round( 3 ) ) +
             '. Optimal epoch val loss: ' + str( round( min_val_loss[ 1 ], 5 ) ), 
             ha = 'center', fontsize = 8 )
plt.show( )