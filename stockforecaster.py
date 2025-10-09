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

import glob 
import os
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
import yfinance as yf

# Constants
DAYS_AHEAD = 20  # Predicting 20 days ahead
WINDOW_SIZE = 20  # Size of window (20 days)
DIRECTORY = '/tmp/saved_models/' 
PATH = DIRECTORY + 'model_epoch_*.keras'
CONSTANT = 12  # Arbitrary constant seed such that Tensorflow and numPy can reproduce results
STD_DROPOUT_LAYER = 0.20  # Dropped out 20% of neurons to prevent overfitting
ALPHA = 0.25  # Scoring/Accuracy formula: ALPHA * training RMSE + (1 - ALPHA) * validation RMSE


def slidingWindow(input, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates windows used to partition and prepare data for LSTM predictive 
    training.  The model needs to learn to predict a value based on a sequence
    of previous values.

    This function uses a times series data array and window size as parameters 
    for partitioning time series data sets, and constructs an array of sliding 
    windows on the training data with corresponding target values.  Each target
    immediately follows its corresponding window.

    Parameters:
    - ndarray
        A 2D NumPy array of shape (n, 1) such that each row represents a
        sequential and chronological value from the time series stock data.
    window_size : int
        Length of the window and the number of data points to include in each
        window.  Therefore there will be n - window_size windows generated.

    Returns
    -------
    tuple of ndarray
        Tuple containing two elements, the first being a 3D NumPy array of
        shape (n_windows, 1, window_size) where each element is a window
        of 'window_size' consecutive samples from the input array.  
        The second element is a 1D array of length `n_windows` where each 
        element is the target value for the corresponding window.
    """
    WINDOW, TARGET = [], [] 

    for i in range((len(input) - 1) - window_size):  # Number of windows to be generated
        window = input[i:(i + window_size), 0]  # Creates windows of size "WINDOW_SIZE"
        WINDOW.append(window)  # Appends windows into WINDOW array
        TARGET.append(input[i + window_size, 0])  # Appends a "target" data point which will be the 21st point (relatively) of each window

    return np.array(WINDOW), np.array(TARGET)  # Creating numpy arrays WITHOUT INDICES which contain necessary data points (thus, returns tuples)


def dataLeakOccured(merge_shape: tuple, shape_1: tuple, shape_2: tuple) -> bool:
    """
    Checks if any entry was repeated or missing as a result of the splitting (or, alternatively, an accidental merge): data leak
    
    Parameters:
    - merge_shape (tuple): 
    - shape_1 (tuple): 
    - shape_2 (tuple): 
    """
    return not(merge_shape[0] == (shape_1[0] + shape_2[0]))


def retrieve_stock_data(symbol: str, start_date: str, end_date: str) -> Tuple[np.ndarray, MinMaxScaler, pd.DataFrame]:
    """
    Retrieves and cleans stock data for model training.
    
    Parameters:
    - symbol (str): Stock ticker symbol
    - start_date (str): Start date in format YYYY-MM-DD
    - end_date (str): End date in format YYYY-MM-DD
    
    Returns:
    - Tuple containing:
        - prices_df (np.ndarray): Scaled price data
        - scaler (MinMaxScaler): Fitted scaler object
        - raw_df (pd.DataFrame): Original dataframe with date and close columns
    """
    # Get stock history using yfinance
    stock = yf.Ticker(symbol)
    df = stock.history(start=start_date, end=end_date)
    df = df.reset_index()
    
    # Lowercasing all column labels in the DataFrame
    df.columns = df.columns.str.lower()
    
    # Convert all date entries to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Such that the newest stock data point is last
    df = df.sort_values(by='date', ascending=True)
    
    # Store raw dataframe before processing
    raw_df = df[['date', 'close']].copy()
    
    # Isolated date and close columns, but drops the date column
    df = (df[['date', 'close']]).drop('date', axis=1)
    
    # Removes original indices so that only the raw indices for the prices remain (just prices)
    df = df.reset_index(drop=True)
    
    # Converts all price data to 32 digit float values
    prices_df = df.astype('float32')
    
    # Reshapes data into necessary rows and one column
    prices_df = np.reshape(prices_df, (-1, 1))
    
    # Reshapes data into variable "scaler" so that it shrinks all values of data into range between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Calculates the minimum and maximum values in data, then scales it into a range between 0 and 1
    prices_df = scaler.fit_transform(prices_df)
    
    return prices_df, scaler, raw_df


def perform_prediction(prices_df: np.ndarray, scaler: MinMaxScaler) -> dict:
    """
    Trains LSTM model and performs stock price prediction.
    
    Parameters:
    - prices_df (np.ndarray): Scaled stock price data
    - scaler (MinMaxScaler): Fitted scaler object for inverse transformation
    
    Returns:
    - dict containing:
        - 'forecast': 20-day price forecast
        - 'training_prediction': Training phase predictions
        - 'val_prediction': Validation phase predictions
        - 'training_target': Actual training prices
        - 'val_target': Actual validation prices
        - 'scores_df': DataFrame with epoch scores
        - 'optimal_epoch_num': Best performing epoch number
        - 'training_loss_list': Training loss history
        - 'val_loss_list': Validation loss history
    """
    os.makedirs(DIRECTORY, exist_ok = True)
    
    # Wipes out saved models folder before proceeding
    for epoch in glob.glob(PATH): 
        os.remove(epoch)  # Removes all old model epochs, if they exist
    
    DATA_LENGTH = len(prices_df)
    
    # Separating the training data of the program into the first 80% of the price movement
    training_phase = int(DATA_LENGTH * 0.80)
    
    # Separating the validation/testing data of the program into the second 20% of the price movement
    val_phase = int(DATA_LENGTH - training_phase)
    
    # First 80% of stock data
    training_df = prices_df[0:training_phase]
    
    # Last 20% of stock data
    val_df = prices_df[training_phase:DATA_LENGTH]
    
    # Training array of 20-day windows with corresponding prediction target 
    training_window, training_target = slidingWindow(training_df, WINDOW_SIZE)
    
    # Validation array of 20-day windows with corresponding prediction target
    val_window, val_target = slidingWindow(val_df, WINDOW_SIZE)
    
    # Reformatting train_window into time-series data: Each 20-day sliding window training sample is being classified into a single time step
    training_window = np.reshape(training_window, (training_window.shape[0], 1, training_window.shape[1]))
    
    # Reformatting val_window into time-series data: Each 20-day sliding window training sample is being classified into a single time step
    val_window = np.reshape(val_window, (val_window.shape[0], 1, val_window.shape[1]))
    
    # Variable to check if data leak occurred (i.e. dimension concatenation error)
    prices_dimensions = prices_df.shape
    
    # First 80% of the data
    training_df_dimensions = training_df.shape
    
    # Remaining 20% of the data
    val_df_dimensions = val_df.shape
    
    if dataLeakOccured(prices_dimensions, training_df_dimensions, val_df_dimensions):
        print("Data leak HAS occurred.\nPlease check data as final result will likely be affected.\n")
    else:
        print("Data leak has NOT occurred.\nTraining and validation phases were partitioned correctly.\n")
    
    # Set random seeds for reproducibility
    tf.random.set_seed(CONSTANT)
    np.random.seed(CONSTANT)
    
    # Important for time series data, which is sequential
    model = Sequential()
    
    # Model is a Long Short-Term Memory model using the ReLU Activation model to introduce non-linearity concept to training
    model.add(LSTM(units=50, activation='relu', input_shape=(training_window.shape[1], WINDOW_SIZE)))
    
    # Dropping out 20% of neurons (random selection)
    model.add(Dropout(STD_DROPOUT_LAYER))
    model.add(Dense(1)) 
    
    # Calculating mean squared error of training and validation
    model.compile(loss='mean_squared_error')
    
    filepath = f'{DIRECTORY}model_epoch_{{epoch:02d}}.keras'
    
    # Model only remembers best training points and forgets lossy training data
    checkpoint = ModelCheckpoint(filepath=filepath, 
                                 verbose=1,
                                 mode='min')
    
    # History of all model epochs (15)
    history = model.fit(training_window, training_target,
                        epochs=15,
                        batch_size=64, 
                        validation_data=(val_window, val_target),
                        callbacks=[checkpoint],
                        verbose=1,
                        shuffle=False)
    
    # Training MSE of model epochs
    training_loss_list = history.history['loss']
    
    # Validation MSE of model epochs
    val_loss_list = history.history['val_loss']
    
    # Returns list of all model epochs generated
    epochs = glob.glob(PATH)
    
    # Removes scaler for plotting purposes
    training_target = scaler.inverse_transform([training_target])
    
    # Removes scaler for plotting purposes
    val_target = scaler.inverse_transform([val_target])
    
    # Scores list meant to store statistics of each model epoch
    scores = []
    
    for i in range(len(epochs)):
        # Calculating RMSE of training phase
        training_RMSE = np.sqrt(training_loss_list[i])
        
        # Calculating RMSE of validation phase
        val_RMSE = np.sqrt(val_loss_list[i])
        
        # Scoring formula to evaluate model epoch
        score = 1 / (ALPHA * (training_RMSE) + (1 - ALPHA) * (val_RMSE))
        
        # Appending statistics to score list
        scores.append((str(i + 1), training_RMSE, val_RMSE, score))
    
    # Converting to dataframe for further ease of analysis
    scores_df = pd.DataFrame(scores)
    
    # Changing column names
    scores_df.columns = ['epoch', 'training RMSE', 'validation RMSE', 'score']
    
    # Highest score means least loss
    optimal_index = scores_df['score'].idxmax()
    
    # For string use, since epochs are 1-indexed
    optimal_epoch_num = scores_df.iloc[optimal_index, 0]
    
    # Preventing index errors
    if int(optimal_epoch_num) < 10:
        optimal_epoch_num = '0' + optimal_epoch_num
    
    # Epoch that had the highest score
    optimal_model = load_model(f'{DIRECTORY}model_epoch_' + str(optimal_epoch_num) + '.keras')
    
    # Training RMSE of highest score epoch
    optimal_training_RMSE = scores_df.iloc[optimal_index, 1]
    
    # Validation RMSE of highest score epoch
    optimal_val_RMSE = scores_df.iloc[optimal_index, 2]
    
    # training phase, forecasting 20-day window predictions
    training_prediction = scaler.inverse_transform(model.predict(training_window))
    
    # Reshaping for concatenation with validation predictions, visualization
    training_prediction = np.reshape(training_prediction, training_prediction.shape[0])
    
    # validation phase, forecasting 20-day window predictions
    val_prediction = scaler.inverse_transform(model.predict(val_window))
    
    # Reshaping for concatenation with training predictions, visualization
    val_prediction = np.reshape(val_prediction, val_prediction.shape[0])
    
    # Reshaping for concatenation with validation target data
    training_target = np.reshape(training_target, training_prediction.shape[0])
    
    # Reshaping for concatenation with training validation target data
    val_target = np.reshape(val_target, val_prediction.shape[0])
    
    # Future predictions begin with last 20-day window of stock history data
    last_window = prices_df[-WINDOW_SIZE:].reshape((1, 1, WINDOW_SIZE))
    
    # Model epoch future forecast list of 20-day movement of price
    forecast = []
    
    # Using final 20-day window from validation phase as a starting point for prediction
    current_window = last_window
    
    # Actual model prediction: Looking 20 days into the future and continunally predicting using a sliding window
    # There is no training/validation phase here: This is all epoch prediction
    for _ in range(DAYS_AHEAD):
        # Predicting next price using the current 20-day window
        next_price = model.predict(current_window)
        
        # Prediction price is appended to the future list for storage
        forecast.append(next_price[0, 0])
        
        # Current window is shifted one day in the future
        # i.e. if start index was k and end index was n, then new window is [k+1, n+1]
        current_window = np.roll(current_window, -1, axis=2)
        
        # Current window now contains model predicted price for "21st" day
        current_window[0, 0, -1] = next_price
    
    forecast = np.array(forecast).reshape(-1, 1)
    
    # Removes scaler of [0, 1] to original interval
    forecast = scaler.inverse_transform(forecast)
    
    return {
        'forecast': forecast,
        'training_prediction': training_prediction,
        'val_prediction': val_prediction,
        'training_target': training_target,
        'val_target': val_target,
        'scores_df': scores_df,
        'optimal_epoch_num': optimal_epoch_num,
        'training_loss_list': training_loss_list,
        'val_loss_list': val_loss_list
    }


def get_forecast_data(prediction_results: dict):
    """
    Extracts and calculates forecast metrics from prediction results.

    Parameters:
    - prediction_results (dict): Results dictionary from perform_prediction()

    Returns:
    - dict containing:
        - history (np.ndarray): Actual historical prices of time range
        - predicted_history (np.ndarray): Model predicted historical prices of time range
        - model_forecast (np.ndarray): Total model forecast including both history and 20-day prediction of movements
        - mape (float): Mean absolute percentage error calculation of model historical accuracy 
    """
    training_target = prediction_results['training_target']
    val_target = prediction_results['val_target']
    training_prediction = prediction_results['training_prediction']
    val_prediction = prediction_results['val_prediction']
    forecast = prediction_results['forecast']

    # Historical prices of range concatenated into one list
    history = np.concatenate((training_target, val_target), axis=0)

    # Model predicted historical prices of range concatenated into one list
    predicted_history = np.concatenate((training_prediction, val_prediction), axis=0)

    # Model predicted history with 20-day forecast of stock movement
    model_forecast = np.concatenate((predicted_history, forecast.squeeze()), axis=0)

    # Mean Absolute Percentage Error calculation between historical vs. predicted history
    mape = np.mean(np.abs((history - predicted_history) / history)) * 100

    return {
        'history': history, 
        'predicted_history': predicted_history, 
        'model_forecast': model_forecast, 
        'mape': mape
    }
    
'''
def plot_prediction(symbol: str, start_date: str, end_date: str, prediction_results: dict) -> None:
    """
    Plots the stock price prediction results and model performance metrics.
    
    Parameters:
    - symbol (str): Stock ticker symbol
    - start_date (str): Start date in format YYYY-MM-DD
    - end_date (str): End date in format YYYY-MM-DD
    - prediction_results (dict): Results dictionary from perform_prediction()
    """
    # Get forecast data using helper function
    forecast_data = get_forecast_data(prediction_results)
    
    prices_actual = forecast_data['history']
    forecasted_prices = forecast_data['model_forecast']
    
    scores_df = prediction_results['scores_df']
    optimal_epoch_num = prediction_results['optimal_epoch_num']
    training_loss_list = prediction_results['training_loss_list']
    val_loss_list = prediction_results['val_loss_list']
    
    STOCK_DAYS = range(len(prices_actual))
    FORECAST_DAYS = range(len(prices_actual) + DAYS_AHEAD)
    EPOCHS_RANGE = range(1, len(training_loss_list) + 1)
    FINAL_PRICE = str(round(forecasted_prices[-1], 2))
    
    # Preparing data visualization for performance metrics and model prediction
    fig, (stock_plt, loss_plt) = plt.subplots(1, 2, figsize=(18, 6))
    
    stock_plt.plot(STOCK_DAYS, prices_actual, label='Stock Prices in USD', color='green')
    stock_plt.plot(FORECAST_DAYS, forecasted_prices, label='Forecasted Prices in USD', color='red')
    stock_plt.plot(FORECAST_DAYS[-1], forecasted_prices[-1], marker='o', markersize='5', markerfacecolor='blue',
                   alpha=0.2, label=f'20-day price: ${FINAL_PRICE}')
    stock_plt.set_title(f'{str(DAYS_AHEAD)}-Day Stock Price Prediction for {symbol.upper()}')
    stock_plt.set_xlabel('Days')
    stock_plt.set_ylabel('Stock Price in USD')
    stock_plt.legend()
    
    loss_plt.plot(EPOCHS_RANGE, scores_df['training RMSE'], label='Training RMSE', color='orange')
    loss_plt.plot(EPOCHS_RANGE, scores_df['validation RMSE'], label='Validation RMSE', color='blue')
    loss_plt.axvline(x=int(optimal_epoch_num), color='red', alpha=0.2,
                     label='Model Epoch used (epoch ' + str(optimal_epoch_num) + ')')
    loss_plt.set_title('Training and Validation RMSE')
    loss_plt.set_xlabel('Epochs')
    loss_plt.set_ylabel('RMSE')
    loss_plt.legend()
    
    plt.figtext(0.5, 0.01, f'Stock history (format: YYYY-MM-DD) is from {start_date} to {end_date}. Prediction is 20 trading days after end date.', ha='center', fontsize=10)
    plt.show()

# Example usage
if __name__ == "__main__":
    symbol = 'SLNH'
    start_date = '2024-01-01'
    end_date = '2025-10-02'
    
    # Step 1: Retrieve and clean stock data
    prices_df, scaler, raw_df = retrieve_stock_data(symbol, start_date, end_date)
    
    # Step 2: Perform prediction
    prediction_results = perform_prediction(prices_df, scaler)
    
    # Step 3: Plot results
    plot_prediction(symbol, start_date, end_date, prediction_results)
'''