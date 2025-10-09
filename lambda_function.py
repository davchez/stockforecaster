import stockforecaster
import webscraper
import json

import boto3 # AWS SDK: Talk to DynamoDB
import uuid # Generates UUIDs
from datetime import datetime, timezone

from decimal import Decimal

"""
Initializing DynamoDB and table
"""
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('stock-forecaster-table')

def lambda_handler(event, context):

    print(f"DEBUG: Full event received: {json.dumps(event, default=str)}")
    print(f"DEBUG: httpMethod = {event.get('httpMethod')}")
    print(f"DEBUG: path = {event.get('path')}")
    
    if event.get('httpMethod') == 'POST' and '/predict' in event.get('path', ''):
        return post_prediction(event)
    
    elif event.get('source') == 'async-invoke':
        return process_prediction(event)
    
    elif event.get('httpMethod') == 'GET' and '/status/' in event.get('path', ''):
        return get_status(event)
    
def post_prediction(event):
    """
    Handles POST /predict - submits new prediction request
    Returns request_id immediately, processes in background
    """

    try:
        # Step 1: Parse incoming request from user
        body = json.loads(event.get('body', '{}'))
        ticker = body.get('ticker', '').upper()
        start_date = body.get('start_date')
        end_date = body.get('end_date')
        include_sentiment = body.get('include_sentiment', True)

        # Step 2: Validate required fields from user request
        if not ticker or not start_date or not end_date:
            return post_status_response(
                status_code = 400, # (Malformed user request, 'BAD REQUEST 400')
                data = {'error': 'Missing required fields: ticker, start_date, end_date'}
            )
        
        # Step 3: Generate unique request ID
        request_id = str(uuid.uuid4())

        # Step 4: Store in DynamoDB with 'PENDING' status
        table.put_item(
            Item = {
                'request-id': request_id,
                'status': 'PENDING',
                'ticker': ticker,
                'start_date': start_date,
                'end_date': end_date,
                'include_sentiment': include_sentiment,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'updated_at': datetime.now(timezone.utc).isoformat()
            }
        )

        # Step 5: Invoke Lambda Function async to process in background to avoid timeout
        lambda_client = boto3.client('lambda')
        lambda_client.invoke(
            FunctionName = 'stock-forecaster',
            InvocationType = 'Event', # (async)
            Payload = json.dumps({
                'source': 'async-invoke',
                'request_id': request_id,
                'ticker': ticker,
                'start_date': start_date,
                'end_date': end_date,
                'include_sentiment': include_sentiment
            })
        )

        # Step 6: Return immediately with request_id
        return post_status_response(
            status_code = 202, # (accepted)
            data = {
                'request_id': request_id,
                'status': 'PENDING',
                'message': 'Prediction request successfully submitted. Check status with GET /status/{request_id}'
            }
        )
    
    except Exception as e:
        return post_status_response(
            status_code = 500, # (Internal Server Error)
            data = {
                'error': 'Failed to submit prediction',
                'details': str(e)
            }
        )
    
def process_prediction(event):
    """
    Processes the 20-day stock forecast using Tensorflow LSTM in background (async invocation)
    This is where the long-waited heavy LSTM training happens
    """

    try:
        # Step 1: Extract data from async invocation payload
        request_id = event.get('request_id')
        ticker = event.get('ticker').upper()
        start_date = event.get('start_date')
        end_date = event.get('end_date')
        include_sentiment = event.get('include_sentiment', True)

        print(f'[{request_id}] Starting prediction for {ticker}')

        # Step 2: Update DynamoDB status to PROCESSING
        table.update_item(
            Key = {'request-id': request_id},
            UpdateExpression = 'SET #status = :status, updated_at = :updated_at', # 'status' is a reserved word (research)
            ExpressionAttributeNames = {'#status': 'status'},
            ExpressionAttributeValues = {
                ':status': 'PROCESSING',
                ':updated_at': datetime.now(timezone.utc).isoformat()
            }
        )

        print(f'[{request_id}] Retrieving stock data for {ticker}')

        # Step 3: Using stockforecaster native program to retrieve and prepare stock data
        prices_df, scaler, _ = stockforecaster.retrieve_stock_data(
            ticker, start_date, end_date
        )

        if len(prices_df) < 125:
            raise ValueError(
                f'Insufficient history: need at least 125 days, got {len(prices_df)}'
            )

        print(f'[{request_id}] Running LSTM training and prediction')

        # Step 4: Stockforecaster native program performs LSTM training and prediction
        prediction_results = stockforecaster.perform_prediction(prices_df, scaler)

        # Step 5: Extract forecasted data from LSTM prediction
        forecast_data = stockforecaster.get_forecast_data(prediction_results)

        # Step 6: Perform sentiment analysis if requested by user (default: will perform)
        sentiment_data = None
        if include_sentiment:
            try:
                print(f'[{request_id}] Analyzing news sentiment for {ticker} from Finviz articles')
                news_rows = webscraper.scrape_finviz_news(ticker)
                news_df = webscraper.clean_data(news_rows, ticker)

                if not news_df.empty:
                    # calculate_sentiment automatically removes all VADER non-calculable headlines where sentiment = 0.00
                    avg_sentiment = webscraper.calculate_sentiment(news_df, adjust = True)
                    sentiment_data = {
                        'average_sentiment': str(float(avg_sentiment)) if avg_sentiment is not None else None,
                        'news_headlines': [
                            {
                                'datetime': row['datetime'],
                                'headline': row['headline'],
                                'sentiment': str(row['sentiment']),
                            }
                            for row in news_df.head().to_dict('records')
                        ]
                    }
            except Exception as e:
                print(f'[{request_id}] Sentiment analysis failed: {str(e)}')
                sentiment_data = None
        
        # Step 7: Build the response data
        response_data = {
            "ticker": ticker,
            "date_range": {
                "start": start_date,
                "end": end_date
            },
            "prediction": {
                "predicted_price_20d": f"${float(forecast_data['model_forecast'][-1]):.2f}",
                "current_price": f"${float(forecast_data['history'][-1]):.2f}",
                "predicted_change_20d": f"{float((forecast_data['model_forecast'][-1] - forecast_data['history'][-1]) / forecast_data['history'][-1]) * 100:.2f}%"
            },
            "model_performance": {
                "mape": f"{float(forecast_data['mape']):.2f}%"
            },
            "forecast_timeline": {
                "historical_days": len(forecast_data["history"]),
                "forecast_days": 20,
                "total_days": len(forecast_data["model_forecast"]),
            }
        }
        
        if sentiment_data:
            response_data['sentiment'] = sentiment_data
        
        # Step 8: Store completed result in DynamoDB table
        table.update_item(
            Key = {'request-id': request_id},
            UpdateExpression = 'SET #status = :status, #result = :result, updated_at = :updated_at',
            ExpressionAttributeNames = {'#status': 'status', '#result': 'result'},
            ExpressionAttributeValues = {
                ':status': 'COMPLETED',
                ':result': response_data,
                ':updated_at': datetime.now(timezone.utc).isoformat()
            }
        )

        print(f'[{request_id}] Prediction completed successfully')
        return post_status_response(
            status_code = 200,
            data = {
                'message': 'Stockforecaster program successfully completed prediction',
                'request_id': request_id
            }
        )
    
    except Exception as e:
        print(f'[{request_id}] Error during processing: {str(e)}')
        import traceback 
        traceback.print_exc()

        table.update_item(
            Key = {'request-id': request_id},
            UpdateExpression = 'SET #status = :status, error = :error, updated_at = :updated_at',
            ExpressionAttributeNames = {'#status': 'status', '#error': 'error'},
            ExpressionAttributeValues = {
                ':status': 'FAILED',
                ':error': str(e),
                ':updated_at': datetime.now(timezone.utc).isoformat()
            }
        )

        return post_status_response(
            status_code = 500,
            data = {
                'message' : f'Stockforecaster program failed: {str(e)}',
                'request_id': request_id,
            }
        )

def get_status(event):
    """
    Handles GET/status/{request_id} to check prediction status
    Returns current status and result if completed
    """

    try:
        # Step 1: Extract request_id from URL path
        # Format: /status/f47ac10b-58cc-4372...
        path = event.get('path', '')
        request_id = path.split('/status/')[-1]

        if not request_id:
            return post_status_response(
                status_code = 400,
                data = {
                    'error': 'Missing request_id in path'
                }
            )
        
        print(f'[{request_id}] Checking status')

        # Step 2: Look up request in DynamoDB table
        response = table.get_item(
            Key = {'request-id': request_id}
        )

        # Step 3: Check if request exists
        if 'Item' not in response:
            return post_status_response(
                status_code = 404, # (Not Found Error)
                data = {
                    'error': 'Request not found',
                    'request_id': request_id
                }
            )
        
        # Step 4: Extract the item from DynamoDB table
        item = response['Item']
        status = item.get('status')

        # Step 5: Build response based on status
        response_data = {
            'request_id': request_id,
            'status': status,
            'ticker': item.get('ticker', '').upper(),
            'created_at': item.get('created_at'),
            'updated_at': item.get('updated_at')
        }

        # Step 6;1: If status is COMPLETED, add the result to the response
        if status == 'COMPLETED':
            response_data['result'] = convert_decimals(item.get('result'))
        
        # Step 6.2: If status is FAILED, add the error to the response
        elif status == 'FAILED':
            response_data['error'] = item.get('error')
        
        # Step 7: Return the response to the user
        return post_status_response(
            status_code = 200, # (Success),
            data = response_data
        )
    
    except Exception as e:
        print(f'Error checking status: {str(e)}')
        import traceback
        traceback.print_exc()
        
        return post_status_response(
            status_code = 500, # (Internal Server Error),
            data = {
                'error': 'Failed to check status',
                'details': str(e)
            }
        )

def post_status_response(status_code: int, data: dict):
    """
    Modularized handling of posting status responses.

    Parameters:
    - status_code (int): HTTP status code (202, 400, 404, 500, etc.)
    - data (dict): Dictionary to be JSON-serialized in body

    Returns:
    - dict: Formatted Lambda response with headers and JSON body
    """
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
        },
        'body': json.dumps(data)
    }

def convert_decimals(obj):
    """
    Convert DynamoDB decimals to Python types for JSON serialization.

    Created with the help of Claude Sonnet 4.5 to navigate around API gateway issues.
    """
    if isinstance(obj, list):
        return [convert_decimals(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: convert_decimals(v) for k, v in obj.items()}
    elif isinstance(obj, Decimal):
        return int(obj) if obj % 1 == 0 else float(obj)
    else:
        return obj