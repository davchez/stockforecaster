// Importing env variables (API URL and Key from .env.local)
const API_URL = process.env.NEXT_PUBLIC_API_URL;
const API_KEY = process.env.NEXT_PUBLIC_API_KEY;

// TypeScript interface - Defining shape of data SENDING to the API ensuring structure with POST
export interface PredictionRequest {
    ticker: string;                 // Stock symbol ("ticker")
    start_date: string;             // Format: "YYYY-MM-DD"
    end_date: string;               // Format: "YYYY-MM-DD"
    include_sentiment: boolean;     // Boolean check on whether or not to include news sentiment
}

// TypeScript interface - What we GET BACK when sumitting a prediction after a POST request
export interface PredictionResponse {
    request_id: string;                                          // UUID to track prediction
    status: 'PENDING' | 'PROCESSING' | 'COMPLETED' | 'FAILED';   
    message?: string;                                            // Optional message
}

// TypeScript interface: What we GET BACK when checking status using GET
export interface StatusResponse {
    request_id: string;
    status: 'PENDING' | 'PROCESSING' | 'COMPLETED' | 'FAILED';
    ticker: string;
    created_at: string;                     // ISO timestamp
    updated_at: string;                     // ISO timestamp

    // Optional result field which only exists when status is 'COMPLETED'
    result?: {
        ticker: string;
        date_range: {
            start: string;
            end: string;
        };
        prediction: {
            predicted_price_20d: string;    // 20 day prediction (i.e. "$202.33")
            current_price: string;          // Price of final day in time range
            predicted_change_20d: string;   // Percentage change (i.e. "-22.38%") from final day
        };
        model_performance: {
            mape: string;                   // Backtest mean average percent error (e.g. "3.55%")
        };
        sentiment?: {                       // Optional; only if include_sentiment is true
            average_sentiment: string;
            news_headlines: Array<{         // Array declares all objects must have these 3 properties
                datetime: string;           // Format: {month}-{day}
                headline: string;           // News headline
                sentiment: string;          // Calculated sentiment from VADER
            }>;
        };
        forecast_timeline: {
            historical_days: number;        // Number of trading days LSTM used to train
            forecast_days: number;          // Number of days out LSTM is forecasting
            total_days: number;             // Total number of days (history + forecast)
        };
    };

    // 'error' field for if status is 'FAILED'
    error?: string;                  
}

// Function: Submits a new predction request
// Returns: returns a "promise" that resolves to PredictionResponse
// Note: Similar to Future in Dart
export async function submitPrediction(request: PredictionRequest): Promise<PredictionResponse> {
    // Make HTTP POST request to /predict endpoint of API
    const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'x-api-key': API_KEY!,
        },
        body: JSON.stringify(request)               // Similar to json.dump in Python, I believe
    });

    // If request failed for internal or malformed request
    if (!response.ok) {
        throw new Error(`Failed to submit prediction: ${response.statusText}`)
    }

    // Parse JSON response and return it
    return response.json();
}

// Function: Checks status of existing prediction
// Returns: Promise that resolves to StatusResponse
export async function checkStatus(requestId: string): Promise<StatusResponse> {
    // Make HTTP GET request to /status/{requestID} endpoint of API
    const response = await fetch(`${API_URL}/status/${requestId}`, {
        method: 'GET',
        headers: {
            'x-api-key': API_KEY!,
        },
    });

    // If request failed for internal or malformed request
    if (!response.ok) {
        throw new Error(`Failed to check status: ${response.statusText}`);
    }

    // Parse JSON response and return it
    return response.json();
}