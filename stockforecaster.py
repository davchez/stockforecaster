"""
Title: Stock Price Predictor
Author: David Sanchez
GitHub: https://github.com/davchez

Description:
    This machine learning Python project leverages the neural network capabilities of the TensorFlow
    and Keras packages.  It formats historic stock prices into time series data to train a long short-term
    memory (LSTM) model.  The model predicts stock price movements x amount of days into the future;
    it generally works best with predictions spanning 20 days.

    The model allows 15 epochs of the entire data set, which, according to scientific literature
    found in https://www.geeksforgeeks.org, is enough to train the model and to prevent
    overfitting.

Bugs: Investpy and Investiny packages are deprecated, and are unreliable.

Credits:
    Caelan from Kite on YouTube.  Assistance in developing code for the neural network model.

Citation:
    Alvaro Bartolome del Canto. investpy - Financial Data Extraction from Investing.com with Python.
    2018-2021. GitHub Repository. Available at: https://github.com/alvarobartt/investpy
"""

import pandas as pd
import numpy as np
import investiny
import investpy
import glob
import os

DIRECTORY = 'saved_models/'
PATH = DIRECTORY + 'model_epoch_*.hdf5'
epochs = glob.glob( PATH )

for epoch in epochs:
    os.remove( epoch )
 
symbol = input( "Enter stock symbol: " )
exchange = input( "Enter stock exchange: " )
default = False

try:
    # Searches Investing.com database to see if query exists
    search_result = investiny.search_assets( query = symbol, 
                                            limit = 1, 
                                            type = 'Stock',
                                            exchange = exchange )
    investing_id = int( search_result[ 0 ][ 'ticker' ] )
    print( f"\ninvestiny Search successful.\n" )
    start_date = input( "Enter start date (mm/dd/yyyy): " )
    end_date = input( "Enter end date (mm/dd/yyyy): " )
    historical_data = investiny.historical_data( investing_id = investing_id, 
                                                from_date = start_date, 
                                                to_date = end_date )
    historical_data = pd.DataFrame( historical_data )
except Exception as e:
    print( "Search failed:", e, '\nTrying again with investpy package.' )
    try:
        country = input( "Enter country of company: ")
        search_result = investpy.search_quotes( text = symbol,
                                               products = [ 'stocks' ],
                                               countries = [ country ],
                                               n_results = 1 )
        print( f"\ninvestpy Search successful.\n" )
        start_date = input( "Enter start date (dd/mm/yyyy): " )
        end_date = input( "Enter end date (dd/mm/yyyy): " )
        historical_data = search_result.retrieve_historical_data( from_date = start_date,
                                                                 to_date = end_date )
    except Exception as e:
        print( "Search failed:", e, '\nResorting to default data.' )
        default = True

if default == False:
    print( f"""\nHistorical data for { symbol }:
          START: { start_date }
          END: { end_date }\n""" )
    df = historical_data.reset_index()
else:
    df = pd.read_csv( 'stockdata/AAPL_Stock_Data.csv' )

# Clean data and prepare for analysis
df.columns = df.columns.str.lower()
df[ 'date' ] = pd.to_datetime( df[ 'date' ], errors = 'coerce' )
df = df.sort_values( by = 'date', ascending = True )

# Isolated date and close columns, but drops the date column (axis = 1)
df = ( df[ [ 'date', 'close' ] ] ).drop( 'date', axis = 1 )

# Removes original indices so that only the raw indices for the prices remain (just prices)
df = df.reset_index( drop = True )

# Converts all price data to 32 digit float values
prices_df = df.astype( 'float32' )

# Reshapes the data variable so that it allows as many rows as possible but formats it into one column
prices_df = np.reshape( prices_df, ( -1, 1 ) )

from sklearn.preprocessing import MinMaxScaler

# Reshapes data into variable "scaler" so that it shrinks all values of data into range between 0 and 1
scaler = MinMaxScaler( feature_range = ( 0, 1 ) )

# Calculates the minimum and maximum values in data, then scales it into a range between 0 and 1
prices_df = scaler.fit_transform( prices_df )

DATA_LENGTH = len( prices_df )

# Separating the training data of the program into the first 80% of the price movement
training_phase = int( DATA_LENGTH * 0.80 ) 

# Separating the validation/testing data of the program into the second 20% of the price movement
val_phase = int( DATA_LENGTH - training_phase )

# Implementing length
training_df = prices_df[ 0:training_phase ]
val_df = prices_df[ training_phase:DATA_LENGTH ]

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
training_window, training_target = slidingWindow( training_df, WINDOW_SIZE )

# The same is made for the second 20% of the data
val_window, val_target = slidingWindow( val_df, WINDOW_SIZE )

# Reformatting train_window into time-series data: Each 20-day sliding window training sample is being classified into a single time step
training_window = np.reshape( training_window, 
                            ( training_window.shape[ 0 ], 1, training_window.shape[ 1 ] ) )

# Same idea
val_window = np.reshape( val_window,
                           ( val_window.shape[ 0 ], 1, val_window.shape[ 1 ] ) )

# Shape of the data variable, which SHOULD be the sum of the samples (rows) of the training + test division (80% + 20%)
prices_dimensions = prices_df.shape

# First 80% of the data
training_df_dimensions = training_df.shape

# Last 20% of the data
val_df_dimensions = val_df.shape

# Checks if any entry was repeated or missing as a result of the splitting (or, alternatively, an accidental merge): data leak
def dataLeakOccured( merge_shape, shape_1, shape_2 ):
    return not( merge_shape[0] == ( shape_1[ 0 ] + shape_2[ 0 ] ) )

# Prints false if data leak did not occur; true if otherwise
dataLeak = dataLeakOccured( prices_dimensions, training_df_dimensions, val_df_dimensions )
if dataLeak:
    print("Data leak HAS occurred.\nPlease check data as final result will likely be affected.\n")
else:
    print("Data leak has NOT occurred.\nTraining and validation phases were partitioned correctly.\n")

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
                  input_shape = ( training_window.shape[ 1 ], WINDOW_SIZE ) ) )

model.add( Dropout ( STD_DROPOUT_LAYER ) )
model.add( Dense( 1 ) )
model.compile( loss = 'mean_squared_error' )

filepath = 'saved_models/model_epoch_{epoch:02d}.hdf5'

# Model only remembers best training points and forgets lossy training data
checkpoint = ModelCheckpoint( filepath = filepath, 
                              # monitor = 'val_loss', ################## COMMENTED OUT TO TEST OUT OTHER MONITOR METHODS, WE'RE GOING TO FOCUS ON ACCURACY
                              verbose = 1,
                              # save_best_only = True, ################## COMMENTED OUT TO TEST OUT OTHER MONITOR METHODS
                              mode = 'min' )

# History of all models which will be compiled into 100 generationally-learning epochs
history = model.fit( training_window, training_target,
                     epochs = 15,
                     batch_size = 20,
                     validation_data = ( val_window, val_target ),
                     callbacks = [ checkpoint ],
                     verbose = 1,
                     shuffle = False )

# Recording the training losses of all epochs
training_loss_list = history.history[ 'loss' ]

# Recording the validation losses of all epochs
val_loss_list = history.history[ 'val_loss' ]

import re
from keras.models import load_model
from sklearn.metrics import mean_squared_error

ALPHA = 0.2
epochs = glob.glob( PATH )

training_target = scaler.inverse_transform( [ training_target ] )
val_target = scaler.inverse_transform( [ val_target ] )

scores = []

for epoch in epochs:
    model = load_model( epoch )
    regex = re.search( r'model_epoch_(\d+).hdf5', epoch )
    index = int( regex.group( 1 ) ) - 1

    # Training phase movement of first 80% of data is predicted using optimal epoch
    # Removes the standardizing scaler of [0, 1] to return prediction to original scaling [min, max]
    training_prediction_temp = scaler.inverse_transform( model.predict( training_window ) )
    training_prediction_temp = np.reshape( training_prediction_temp, training_prediction_temp.shape[ 0 ] )
    
    # Testing phase movement of remaining 20% of data is predicted using optimal epoch
    # Removes the standardizing scale of [0, 1] to return prediction to original scaling [min, max]
    val_prediction_temp = scaler.inverse_transform( model.predict( val_window ) )
    val_prediction_temp = np.reshape( val_prediction_temp, val_prediction_temp.shape[ 0 ] )

    # Actual training data to be compared against, i.e. first 80% of original data set
    # Removes the standardizing scaler of [0, 1] to return prediction to original scaling [min, max]
    training_target_temp = np.reshape( training_target, training_prediction_temp.shape[ 0 ] )
    val_target_temp = np.reshape( val_target, val_prediction_temp.shape[ 0 ] )

    # Actual validation data to be compared against, i.e. first 80% of original data set
    # Removes the standardizing scaler of [0, 1] to return prediction to original scaling [min, max]
    training_MSE = mean_squared_error( training_target_temp, training_prediction_temp )

    # Actual validation data to be compared against, i.e. last 20% of original data set
    # Removes the standardizing scaler of [0, 1] to return prediction to original scaling [min, max]
    val_MSE = mean_squared_error( val_target_temp, val_prediction_temp )

    # Root mean square error showing loss between actual training data and training prediction from epoch
    training_RMSE = np.sqrt( training_MSE )

    # Root mean square error showing loss between actual validation data and validation prediction from epoch
    val_RMSE = np.sqrt( val_MSE )

    score = ALPHA * ( training_MSE + training_RMSE ) + ( 1 - ALPHA ) * ( val_MSE + val_RMSE )
    
    scores.append( ( str( index + 1 ), training_MSE, training_RMSE, val_MSE, val_RMSE, score ) )

scores_df = pd.DataFrame( scores )
scores_df.columns = [ 'epoch', 'training MSE', 'training RMSE', 'validation MSE', 'validation RMSE', 'score' ]

print( scores_df )

optimal_index = scores_df[ 'score' ].idxmin()
optimal_epoch_num = scores_df.iloc[ optimal_index, 0 ]

if int( optimal_epoch_num ) < 10:
    optimal_epoch_num = '0' + optimal_epoch_num

optimal_model = load_model( 'saved_models/model_epoch_' + str( optimal_epoch_num ) + '.hdf5' )
optimal_training_RMSE = scores_df.iloc[ optimal_index, 2 ]
optimal_val_RMSE = scores_df.iloc[ optimal_index, 4 ]

# Training phase movement of first 80% of data is predicted using optimal epoch
# Removes the standardizing scaler of [0, 1] to return prediction to original scaling [min, max]
training_prediction = scaler.inverse_transform( model.predict( training_window ) )
training_prediction = np.reshape( training_prediction, training_prediction.shape[ 0 ] )
    
# Testing phase movement of remaining 20% of data is predicted using optimal epoch
# Removes the standardizing scale of [0, 1] to return prediction to original scaling [min, max]
val_prediction = scaler.inverse_transform( model.predict( val_window ) )
val_prediction = np.reshape( val_prediction, val_prediction.shape[ 0 ] )

training_target = np.reshape( training_target, ( -1, 1 ) )
val_target = np.reshape( val_target, ( -1, 1 ) )

# Moving into the future; no original data to compare with, attempting to predict movement 20 days in advance
# First comparing with the last 20 days of model prediction window (final 20 days of data set comparison) to begin modeling
last_window = prices_df[ -WINDOW_SIZE: ].reshape( ( 1, 1, WINDOW_SIZE ) )

# Preparing model forecast list for 20 days of prediction into the future
forecast = []

# Using final 20-day window from validation phase as a starting point for prediction
current_window = last_window

DAYS_AHEAD = 20

# Actual model prediction: Looking 20 days into the future and continunally predicting using a sliding window
# There is no training/validation phase here: This is all epoch prediction

for _ in range( DAYS_AHEAD ):

    # Predicting next price using the current 20-day window
    next_price = model.predict( current_window )

    # Prediction price is appended to the future list for storage
    forecast.append( next_price[ 0, 0 ] )

    # Current window is shifted one day in the future
    # i.e. if start index was k and end index was n, then new window is [k+1, n+1]
    current_window = np.roll( current_window, -1, axis = 2 )

    # Current window now contains model predicted price for "21st" day
    current_window[ 0, 0, -1 ] = next_price

forecast = np.array( forecast ).reshape( -1, 1 )

# Resizing future list so that it removes standardizing scale of [0, 1] to original size [min, max]
forecast = scaler.inverse_transform( forecast )

# Actual prices concatenated into one list
prices_actual = np.concatenate( ( training_target, val_target ), axis = 0 )

# Model training prices concatenated into one list
prices_model = np.concatenate( ( training_prediction, val_prediction ), axis = 0 )

# Predicted prices added onto model training prices to create +20 day prediction
forecasted_prices = np.concatenate( ( prices_model, forecast.squeeze( ) ), axis = 0 )

STOCK_DAYS = range( len( prices_actual ) )
FORECAST_DAYS = range( len( prices_actual ) + DAYS_AHEAD )
EPOCHS_RANGE = range( 1, len( training_loss_list ) + 1 )

import matplotlib.pyplot as plt

# Preparing data visualization for performance metrics and model prediction
fig, ( stock_plt, loss_plt ) = plt.subplots( 1, 2, figsize = ( 18, 6 ) )

stock_plt.plot( STOCK_DAYS, prices_actual, label = 'Stock Prices in USD', color = 'green' )
stock_plt.plot( FORECAST_DAYS, forecasted_prices, label = 'Forecasted Prices in USD', color = 'red' )
stock_plt.set_title( str( DAYS_AHEAD ) + '-Day Stock Price Prediction' )
stock_plt.set_xlabel( 'Days' )
stock_plt.set_ylabel( 'Stock Price in USD' )
stock_plt.legend( )

loss_plt.plot( EPOCHS_RANGE, training_loss_list, label = 'Training Loss', color = 'orange' )
loss_plt.plot( EPOCHS_RANGE, val_loss_list, label = 'Validation Loss', color = 'blue' )
loss_plt.axvline( x = int( optimal_epoch_num ), color = 'red', alpha = 0.2,
                  label = 'Optimal Epoch (epoch ' + str( optimal_epoch_num ) + ')' )
loss_plt.set_title( 'Training and Validation Loss' )
loss_plt.set_xlabel( 'Epochs' )
loss_plt.set_ylabel( 'Loss' )
loss_plt.legend( )

plt.figtext( 0.5, 0.01, 'Training loss USD (RMSE): ' + str( optimal_training_RMSE.round( 3 ) ) + 
             '. Testing loss USD (RMSE): ' + str( optimal_val_RMSE.round( 3 ) ) +
             '. Optimal epoch score: ' + str( round( scores_df[ 'score' ].min(), 5 ) ), 
             ha = 'center', fontsize = 8 )
plt.show( )