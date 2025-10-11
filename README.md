<div align = "center">
  
# StockForecaster
##### Author: David Sanchez &nbsp; • &nbsp; Development: Aug 2023 — Present &nbsp; • &nbsp; [Live Demo - Vercel Deployment](https://stockforecaster-tau.vercel.app/)
  
</div>

&nbsp;

## Description

StockForecaster is a full-stack stock price prediction platform that uses LSTM neural networks to forecast prices 20 days into the future. The project consists of a Python/TensorFlow backend deployed as a serverless AWS Lambda function, accessible via a public REST API with authentication and rate limiting, and a Next.js/TypeScript frontend deployed to Vercel.

The machine learning model leverages TensorFlow and Keras to format historic stock prices into time series data for training a Long Short-Term Memory (LSTM) network. The model works best with predictions spanning 20 days and stock history of at least 1 year. Training uses 15 epochs with 64 batches each, following best practices from scientific literature to prevent overfitting.

&nbsp;

## Technical Stack

#### Backend 
> Python 3.11, TensorFlow 2.15, AWS Lambda (Docker containerized), API Gateway, DynamoDB  

#### Frontend 
> Next.js 14, TypeScript, React, Tailwind CSS (deployed to Vercel)

#### ML Model
> LSTM with 50 units, 20% dropout, custom weighted RMSE epoch selection (75% validation, 25% training)

#### Features 
> Async request/response pattern, VADER sentiment analysis, cold start retry logic

&nbsp;

## API Endpoints

#### POST /predict
> Submit prediction request (returns request_id immediately)

#### GET /status/{request_id}
> Poll for completion and retrieve results

_Note_: Model training takes 1-2 minutes but uses async architecture to avoid API Gateway timeout limits.

&nbsp;

## Reproducible results: Confirmation samples with accuracy score

Confirmation results following update of model epoch scoring formula (75% weighting on validation data)

#### Disclaimer of Results
> Data has been heavily cherrypicked. Forecasting works best on stable stocks with greater than or equal to 6 months of historical data. All claims of MAPE and accuracy are backtesting accuracy results and are not indicators of future performance or future model accuracy.

<div align = "center">

&nbsp;
  
#### Apple (AAPL) Prediction (0.41% MAPE)
##### April 24, 2024 Model Prediction: $169.02 &nbsp; — &nbsp; April 24, 2024 Actual Price: $169.71

</div>

- Training and validation data spanning January 1, 2023 to March 27, 2024 (20 trading days before April 24)
- Model forecasted 20-day price accuracy: 99.59% (pre Q1-earnings)

![Image](stockdata/sample_outputs/confirmation_outputs/AAPL_Sample_Output.png)

&nbsp;

<div align = "center">

#### Disney (DIS) Prediction (1.75% MAPE)
##### April 24, 2024 Model Prediction: $113.92 &nbsp; — &nbsp; April 24, 2024 Actual Price: $111.93

</div>

- Training and validation data spanning January 1, 2023 to March 27, 2024 (20 trading days before April 24)
- Model forecasted 20-day price accuracy: 98.25% (post Q1-earnings)

![Image](stockdata/sample_outputs/confirmation_outputs/DIS_Sample_Output.png)

&nbsp;

<div align = "center">

#### Oracle (ORCL) Prediction (6.98% MAPE)
##### April 24, 2024 Model Prediction: $115.34 &nbsp; — &nbsp; April 24, 2024 Actual Price: $123.39

</div>

- Training and validation data spanning January 1, 2023 to March 27, 2024 (20 trading days before April 24)
- Model forecasted 20-day price accuracy: 93.02% (post Q1-earnings)

![Image](stockdata/sample_outputs/confirmation_outputs/ORCL_Sample_Output.png)

&nbsp;

<div align = "center">

#### Alphabet/Google (GOOG) Prediction (4.46% MAPE)
##### April 24, 2024 Model Prediction: $161.10 &nbsp; — &nbsp; April 24, 2024 Actual Price: $153.92

</div>

- Training and validation data spanning January 1, 2023 to March 27, 2024 (20 trading days before April 24)
- Model forecasted 20-day price accuracy: 95.54% (pre Q1-earnings)

![Image](stockdata/sample_outputs/confirmation_outputs/GOOG_Sample_Output.png)
