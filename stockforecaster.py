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

Bugs: 
    Investpy and Investiny packages are not entirely reliable.
    Length of necessary stock data must be longer than 5 months for program to work.

Credits:
    Caelan from Kite on YouTube.  Assistance in developing code for the neural network model.

Citation:
    Alvaro Bartolome del Canto. investpy - Financial Data Extraction from Investing.com with Python.
    2018-2021. GitHub Repository. Available at: https://github.com/alvarobartt/investpy

Notes:
    Small observation made is that the model seems to work well with stock data just over one year.
"""
import investiny # Necessary for scraping Investing.com
import investpy # Necessary for scraping Investing.com
import glob 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler # Necesssary to constrict range to improve runtime

DAYS_AHEAD = 20 # Predicting 20 days ahead
WINDOW_SIZE = 20 # Size of window (20 days)
DIRECTORY = 'saved_models/' 
PATH = DIRECTORY + 'model_epoch_*.keras'
UNUSED_MSG = 'N/A' # If investiny and investpy fail

date_format, start_date, end_date = UNUSED_MSG, UNUSED_MSG, UNUSED_MSG
epochs = glob.glob( PATH ) # List of old model epochs

# Wipes out saved models folder before proceeding
for epoch in epochs: 
    os.remove( epoch ) # Removes all old model epochs, if they exist

symbol = input( "Enter stock symbol: " ) # Symbol for stock ticker ( i.e. if Apple, enter AAPL )
exchange = input( "Enter stock exchange: " ) # Exchange that stock is publicly traded on ( i.e. NASDAQ )
default = False # Default CSV file in stockdata will load if investiny/investpy fail

try:
    search_result = investiny.search_assets( query = symbol, # Searches Investing.com database to see if query exists
                                            limit = 1, 
                                            type = 'Stock',
                                            exchange = exchange )
    investing_id = int( search_result[ 0 ][ 'ticker' ] ) # Specific stock ID ticker collected from search_assets
    print( f"\ninvestiny Search successful.\n" )
    date_format = 'mm/dd/yyyy'
    start_date = input( f"Enter start date ({ date_format }): " ) # Beginning date of stock history for analysis
    end_date = input( f"Enter end date ({ date_format }): " ) # End date of stock history for analysis 
    historical_data = investiny.historical_data( investing_id = investing_id, # Investiny scrapes Investing.com for the specific stock and its data if found
                                                from_date = start_date, 
                                                to_date = end_date )
    historical_data = pd.DataFrame( historical_data ) # Converted to DataFrame for data analysis purposes
except Exception as e: # Investiny package failed due to erroneous user input or failure to connect to Investing.com
    print( "Search failed:", e, '\nTrying again with investpy package.' )
    try:
        country = input( "Enter country of company: ") # Country of origin of company
        search_result = investpy.search_quotes( text = symbol, # Using investpy search function to look for company
                                               products = [ 'stocks' ],
                                               countries = [ country ],
                                               n_results = 1 )
        print( f"\ninvestpy Search successful.\n" )
        date_format = 'dd/mm/yyyy'
        start_date = input( f"Enter start date ({ date_format }): " ) # Beginning date of stock history for analysis
        end_date = input( f"Enter end date ({ date_format }): " ) # End date of stock history for analysis
        historical_data = search_result.retrieve_historical_data( from_date = start_date, # DataFrame of historical stock data from time range
                                                                 to_date = end_date )
    except Exception as e: # Investiny and investpy failed
        print( "Search failed:", e, '\nResorting to default data.' ) # Default data being CSV stock data
        default = True # Resort to default, use preloaded CSV file

if default == False: # If investiny/investpy worked
    print( f"""\nHistorical data for { symbol }:
          START: { start_date }
          END: { end_date }\n""" )
    df = historical_data.reset_index() # Ensuring that indices are formatted correctly before analysis/tidying
else: # IF investiny/investpy failed
    symbol = 'AAPL' # Default stock to be used
    df = pd.read_csv( f'stockdata/{ symbol }_Stock_Data.csv' )
    

# Clean data and prepare for analysis
df.columns = df.columns.str.lower() # Lowercasing all column labels in the DataFrame
df[ 'date' ] = pd.to_datetime( df[ 'date' ], errors = 'coerce' ) # Convert all date entries to datetime
df = df.sort_values( by = 'date', ascending = True ) # Such that the newest stock data point is last
df = ( df[ [ 'date', 'close' ] ] ).drop( 'date', axis = 1 ) # Isolated date and close columns, but drops the date column
df = df.reset_index( drop = True ) # Removes original indices so that only the raw indices for the prices remain (just prices)

prices_df = df.astype( 'float32' ) # Converts all price data to 32 digit float values
prices_df = np.reshape( prices_df, ( -1, 1 ) ) # Reshapes data into necessary rows and one column

scaler = MinMaxScaler( feature_range = ( 0, 1 ) ) # Reshapes data into variable "scaler" so that it shrinks all values of data into range between 0 and 1

prices_df = scaler.fit_transform( prices_df ) # Calculates the minimum and maximum values in data, then scales it into a range between 0 and 1

DATA_LENGTH = len( prices_df )

training_phase = int( DATA_LENGTH * 0.80 ) # Separating the training data of the program into the first 80% of the price movement

val_phase = int( DATA_LENGTH - training_phase ) # Separating the validation/testing data of the program into the second 20% of the price movement

training_df = prices_df[ 0:training_phase ] # First 80% of stock data
val_df = prices_df[ training_phase:DATA_LENGTH ] # Last 20% of stock data

"""
Generates windows used to partition and prepare data for LSTM predictive 
training.  The model needs to learn to predict a value based on a sequence
of previous values.

This function uses a times series data array and window size as parameters 
for partitioning time series data sets, and constructs an array of sliding 
windows on the training data with corresponding target values.  Each target
immediately follows its corresponding window.

Parameters
----------
input : ndarray
    A 2D NumPy array of shape ( n, 1 ) such that each row represents a
    sequential and chronological value from the time series stock data.
window_size : int
    Length of the window and the number of data points to include in each
    window.  Therefore there will be n - window_size windows generated.

Returns
-------
tuple of ndarray
    Tuple containing two elements, the first being a 3D NumPy array of
    shape ( n_windows, 1, window_size ) where each element is a window
    of 'window_size' consecutive samples from the input array.  
    The second element is a 1D array of length `n_windows` where each 
    element is the target value for the corresponding window.
"""
def slidingWindow( input, window_size ):

    WINDOW, TARGET = [], [] 

    for i in range( ( len( input ) - 1 ) - window_size ): # Number of windows to be generated

        window = input[ i:( i + window_size ), 0 ] # Creates windows of size "WINDOW_SIZE"
        WINDOW.append( window ) # Appends windows into WINDOW array
        TARGET.append( input[ i + window_size, 0 ] ) # Appends a "target" data point which will be the 21st point (relatively) of each window

    return np.array( WINDOW ), np.array( TARGET ) # Creating numpy arrays WITHOUT INDICES which contain necessary data points (thus, returns tuples)

training_window, training_target = slidingWindow( training_df, WINDOW_SIZE ) # Training array of 20-day windows with corresponding prediction target 
val_window, val_target = slidingWindow( val_df, WINDOW_SIZE ) # Validation array of 20-day windows with corresponding prediction target

# Reformatting train_window into time-series data: Each 20-day sliding window training sample is being classified into a single time step
training_window = np.reshape( training_window, 
                            ( training_window.shape[ 0 ], 1, training_window.shape[ 1 ] ) )

# Reformatting val_window into time-series data: Each 20-day sliding window training sample is being classified into a single time step
val_window = np.reshape( val_window,
                           ( val_window.shape[ 0 ], 1, val_window.shape[ 1 ] ) )

prices_dimensions = prices_df.shape # Variable to check if data leak occurred (i.e. dimension concatenation error)
training_df_dimensions = training_df.shape # First 80% of the data
val_df_dimensions = val_df.shape # Remaining 20% of the data

# Checks if any entry was repeated or missing as a result of the splitting (or, alternatively, an accidental merge): data leak
def dataLeakOccured( merge_shape, shape_1, shape_2 ):
    return not( merge_shape[0] == ( shape_1[ 0 ] + shape_2[ 0 ] ) )

dataLeak = dataLeakOccured( prices_dimensions, training_df_dimensions, val_df_dimensions )
if dataLeak:
    print("Data leak HAS occurred.\nPlease check data as final result will likely be affected.\n")
else:
    print("Data leak has NOT occurred.\nTraining and validation phases were partitioned correctly.\n")

CONSTANT = 12 # Arbitrary constant seed such that Tensorflow and numPy can reproduce results
STD_DROPOUT_LAYER = 0.20 # Dropped out 20% of neurons to prevent overfitting

tf.random.set_seed( CONSTANT )
np.random.seed( CONSTANT )

model = Sequential() # Important for time series data, which is sequential

# Model is a Long Short-Term Memory model using the ReLU Activation model to introduce non-linearity concept to training
model.add( LSTM ( units = 50,
                  activation = 'relu',
                  input_shape = ( training_window.shape[ 1 ], WINDOW_SIZE ) ) )

model.add( Dropout ( STD_DROPOUT_LAYER ) ) # Dropping out 20% of neurons (random selection)
model.add( Dense( 1 ) ) 
model.compile( loss = 'mean_squared_error' ) # Calculating mean squared error of training and validation

filepath = 'saved_models/model_epoch_{epoch:02d}.keras'

# Model only remembers best training points and forgets lossy training data
checkpoint = ModelCheckpoint( filepath = filepath, 
                              # monitor = 'val_loss', # Unnecessary as we have scoring formula
                              verbose = 1,
                              # save_best_only = True, # We need all epochs unfortunately
                              mode = 'min' )

# History of all model epochs (15)
history = model.fit( training_window, training_target,
                     epochs = 15,
                     batch_size = 64,
                     validation_data = ( val_window, val_target ),
                     callbacks = [ checkpoint ],
                     verbose = 1,
                     shuffle = False )

training_loss_list = history.history[ 'loss' ] # Training MSE of model epochs
val_loss_list = history.history[ 'val_loss' ] # Validation MSE of model epochs

ALPHA = 0.25 # Scoring/Accuracy formula: ALPHA * training RMSE + (1 - ALPHA) * validation RMSE
epochs = glob.glob( PATH ) # Returns list of all model epochs generated

training_target = scaler.inverse_transform( [ training_target ] ) # Removes scaler for plotting purposes
val_target = scaler.inverse_transform( [ val_target ] ) # Removes scaler for plotting purposes

scores = [] # Scores list meant to store statistics of each model epoch

for i in range( len( epochs ) ):

    training_RMSE = np.sqrt( training_loss_list[ i ] ) # Calculating RMSE of training phase
    val_RMSE = np.sqrt( val_loss_list[ i ] ) # Calculating RMSE of validation phase
    score = 1 / ( ALPHA * ( training_RMSE ) + ( 1 - ALPHA ) * ( val_RMSE ) ) # Scoring formula to evaluate model epoch
    scores.append( ( str( i + 1 ), training_RMSE, val_RMSE, score ) ) # Appending statistics to score list

scores_df = pd.DataFrame( scores ) # Converting to dataframe for further ease of analysis
scores_df.columns = [ 'epoch', 'training RMSE', 'validation RMSE', 'score' ] # Changing column names

print( scores_df.head() ) # Brief check

optimal_index = scores_df[ 'score' ].idxmax() # Highest score means least los
optimal_epoch_num = scores_df.iloc[ optimal_index, 0 ] # For string use, since epochs are 1-indexed

if int( optimal_epoch_num ) < 10: # Preventing index errors
    optimal_epoch_num = '0' + optimal_epoch_num

optimal_model = load_model( 'saved_models/model_epoch_' + str( optimal_epoch_num ) + '.keras' ) # Epoch that had the highest score
optimal_training_RMSE = scores_df.iloc[ optimal_index, 1 ] # Training RMSE of highest score epoch
optimal_val_RMSE = scores_df.iloc[ optimal_index, 2 ] # Validation RMSE of highest score epoch

training_prediction = scaler.inverse_transform( model.predict( training_window ) ) # training phase, forecasting 20-day window predictions
training_prediction = np.reshape( training_prediction, training_prediction.shape[ 0 ] ) # Reshaping for concatenation with validation predictions, visualization

val_prediction = scaler.inverse_transform( model.predict( val_window ) ) # validation phase, forecasting 20-day window predictions
val_prediction = np.reshape( val_prediction, val_prediction.shape[ 0 ] ) # Reshaping for concatenation with training predictions, visualization

training_target = np.reshape( training_target, training_prediction.shape[ 0 ] ) # Reshaping for concatenation with validation target data
val_target = np.reshape( val_target, val_prediction.shape[ 0 ] ) # Reshaping for concatenation with training validation target data

last_window = prices_df[ -WINDOW_SIZE: ].reshape( ( 1, 1, WINDOW_SIZE ) ) # Future predictions begin with last 20-day window of stock history data
forecast = [] # Model epoch future forecast list of 20-day movement of price
current_window = last_window # Using final 20-day window from validation phase as a starting point for prediction

# Actual model prediction: Looking 20 days into the future and continunally predicting using a sliding window
# There is no training/validation phase here: This is all epoch prediction
for _ in range( DAYS_AHEAD ):

    next_price = model.predict( current_window ) # Predicting next price using the current 20-day window
    forecast.append( next_price[ 0, 0 ] ) # Prediction price is appended to the future list for storage

    # Current window is shifted one day in the future
    # i.e. if start index was k and end index was n, then new window is [k+1, n+1]
    current_window = np.roll( current_window, -1, axis = 2 )
    current_window[ 0, 0, -1 ] = next_price # Current window now contains model predicted price for "21st" day

forecast = np.array( forecast ).reshape( -1, 1 )
forecast = scaler.inverse_transform( forecast ) # Removes scaler of [0, 1] to original interval

prices_actual = np.concatenate( ( training_target, val_target ), axis = 0 ) # Actual prices concatenated into one list
prices_model = np.concatenate( ( training_prediction, val_prediction ), axis = 0 ) # Model training + validation predictions concatenated into one list
forecasted_prices = np.concatenate( ( prices_model, forecast.squeeze( ) ), axis = 0 ) # Model 20-day future prediction movements concatenated into one list

STOCK_DAYS = range( len( prices_actual ) )
FORECAST_DAYS = range( len( prices_actual ) + DAYS_AHEAD )
EPOCHS_RANGE = range( 1, len( training_loss_list ) + 1 )
FINAL_PRICE = str( round( forecasted_prices[ -1 ], 2 ) )

# Preparing data visualization for performance metrics and model prediction
fig, ( stock_plt, loss_plt ) = plt.subplots( 1, 2, figsize = ( 18, 6 ) )

stock_plt.plot( STOCK_DAYS, prices_actual, label = 'Stock Prices in USD', color = 'green' )
stock_plt.plot( FORECAST_DAYS, forecasted_prices, label = 'Forecasted Prices in USD', color = 'red' )
stock_plt.plot( FORECAST_DAYS[ -1 ], forecasted_prices[ -1 ], marker = 'o', markersize = '5', markerfacecolor = 'blue',
               alpha = 0.2, label = f'20-day price: ${ FINAL_PRICE }' )
stock_plt.set_title( f'{ str( DAYS_AHEAD ) }-Day Stock Price Prediction for { symbol.upper() }' )
stock_plt.set_xlabel( 'Days' )
stock_plt.set_ylabel( 'Stock Price in USD' )
stock_plt.legend( )

loss_plt.plot( EPOCHS_RANGE, scores_df[ 'training RMSE' ], label = 'Training RMSE', color = 'orange' )
loss_plt.plot( EPOCHS_RANGE, scores_df[ 'validation RMSE' ], label = 'Validation RMSE', color = 'blue' )
loss_plt.axvline( x = int( optimal_epoch_num ), color = 'red', alpha = 0.2,
                  label = 'Model Epoch used (epoch ' + str( optimal_epoch_num ) + ')' )
loss_plt.set_title( 'Training and Validation RMSE' )
loss_plt.set_xlabel( 'Epochs' )
loss_plt.set_ylabel( 'RMSE' )
loss_plt.legend( )

plt.figtext( 0.5, 0.01, f'Stock history (format:{ date_format }) is from { start_date } to { end_date }. Prediction is 20 trading days after end date.', ha = 'center', fontsize = 10 )
plt.show( )