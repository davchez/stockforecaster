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

Picture below shows Apple stock prediction with the following statistics
- Actual April 24, 2024 close: $169.02 USD (downtrend)
- Predicted April 24, 2024 close: $169.71 USD (predicted downtrend)
- Training and validation data spanning January 1, 2023 to March 27, 2023 (20 trading days before April 24)
- Model forecasted 20-day price accuracy: 99.59% (pre Q1-earnings)
![Image](stockdata/sample_outputs/confirmation_outputs/AAPL_Sample_Output.png)

<h3>Disney (DIS) Prediction (98.25% Accuracy)</h3>
> - **Actual April 24, 2024 close: $113.92 USD** (downtrend)
> - **Predicted April 24, 2024 close: $111.93 USD** (predicted downtrend)
> - Training and validation data spanning January 1, 2023 to March 27, 2023 (20 trading days before April 24)
> - Model forecasted 20-day price accuracy: 98.25% (post Q1-earnings)
![Image](stockdata/sample_outputs/confirmation_outputs/DIS_Sample_Output.png)

<h3>Oracle (ORCL) Prediction (93.02% Accuracy)</h3>
> - **Actual April 24, 2024 close: $115.34 USD** (downtrend)
> - **Predicted April 24, 2024 close: $123.39 USD** (predicted downtrend)
> - Training and validation data spanning January 1, 2023 to March 27, 2023 (20 trading days before April 24)
> - Model forecasted 20-day price accuracy: 93.02% (post Q1-earnings)
![Image](stockdata/sample_outputs/confirmation_outputs/ORCL_Sample_Output.png)