# Major Update as of October 7, 2025
Stockforecaster is currently being upgraded with a workable API through AWS.<br>
Since security is exposed during development, the first push will be prod-ready and will display the executable AWS link for the function.  Expected update: October 10, 2025

<h3>Description:</h3>
This machine learning Python project leverages the neural network capabilities of the TensorFlow and Keras packages.  It formats historic stock prices into time series data to train a long short-term memory (LSTM) model.  The model predicts stock price movements x amount of days into the future; it generally works best with predictions spanning 20 days and stock history spanning just over 1 year.  The model allows 15 epochs- with 64 batches each- of the entire data set, which, according to scientific literature found in https://www.geeksforgeeks.org, is enough to train the model and to prevent overfitting.

<h1>Reproducible results: Confirmation samples with accuracy score</h1>
Confirmation results following update of model epoch scoring formula (75% weighting on validation data)

<h3>Disclaimer of Results</h3>
Data has been heavily cherrypicked.  Forecasting <i>seems</i> to work best on stable stocks with large amounts of historical data.  All claims of MAPE and accuracy are <b>backtesting accuracy results</b> and not indicators of future performance.  

<h3>Apple (AAPL) Prediction (0.41% MAPE)</h3>
<ul>
  <li>Actual April 24, 2024 close: $169.02 USD (downtrend)</li>
  <li>Predicted April 24, 2024 close: $169.71 USD (predicted downtrend)</li>
  <li>Training and validation data spanning January 1, 2023 to March 27, 2023 (20 trading days before April 24)</li>
  <li>Model forecasted 20-day price accuracy: 99.59% (pre Q1-earnings)</li>
</ul>

![Image](stockdata/sample_outputs/confirmation_outputs/AAPL_Sample_Output.png)

<h3>Disney (DIS) Prediction (1.75% MAPE)</h3>
<ul>
  <li>Actual April 24, 2024 close: $113.92 USD (downtrend)</li>
  <li>Predicted April 24, 2024 close: $111.93 USD (predicted downtrend)</li>
  <li>Training and validation data spanning January 1, 2023 to March 27, 2023 (20 trading days before April 24)</li>
  <li>Model forecasted 20-day price accuracy: 98.25% (post Q1-earnings)</li>
</ul>

![Image](stockdata/sample_outputs/confirmation_outputs/DIS_Sample_Output.png)

<h3>Oracle (ORCL) Prediction (6.98% MAPE)</h3>
<ul>
  <li>Actual April 24, 2024 close: $115.34 USD (downtrend)</li>
  <li>Predicted April 24, 2024 close: $123.39 USD (predicted downtrend)</li>
  <li>Training and validation data spanning January 1, 2023 to March 27, 2023 (20 trading days before April 24)</li>
  <li>Model forecasted 20-day price accuracy: 93.02% (post Q1-earnings)</li>
</ul>

![Image](stockdata/sample_outputs/confirmation_outputs/ORCL_Sample_Output.png)

<h3>Alphabet/Google (GOOG) Prediction (4.46% MAPE)</h3>
<ul>
  <li>Actual April 24, 2024 close: $161.10 USD (uptrend)</li>
  <li>Predicted April 24, 2024 close: $153.92 USD (predicted uptrend)</li>
  <li>Training and validation data spanning January 1, 2023 to March 27, 2023 (20 trading days before April 24)</li>
  <li>Model forecasted 20-day price accuracy: 95.54% (pre Q1-earnings)</li>
</ul>

![Image](stockdata/sample_outputs/confirmation_outputs/GOOG_Sample_Output.png)
