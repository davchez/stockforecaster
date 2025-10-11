// Telling next.js this is a client component and not server rendered
// Needed since we're using React hooks and browser APIs
'use client';

/*
Note: TypeScript language (superset of JavaScript), React Framework (UI Library)
      - Includes some level of JavaScript as well though ("functional programming" looks semi-OOP)

Note: Server-Side Rendering (SSR): AWS Server (Next.js) -> Generates HTML -> Sends to browser
      - Code runs on Node.js server
      - No access to browser APIs (no window, document, localStorage)
      - Happens ONCE when page loads
      - Good for SEO, fast initial load

Note: Client-Side Rendering (CSR): Browser receives minimal HTML -> JavaScript runs in Browser -> builds UI
      - Code runs in user's browser
      - Full access to browser APIs
      - Can be interactive (responds to user actions)
      - NECESSARY for dynamic apps

Note: React Hooks
      - useState: Memory (Component memory which is NOT persistent; survives re-renders, NOT page refreshes)
      - useEffect: Side effects (Like an observer that changes once a dependency is changed)

Note: Browser API and why servers can't use them
      - Browser APIs can only work in browsers because the screen, window, user-specific storage, and interactions don't exist on the server
      - Server only has node.js so nothing much happens

Note: @/ starts from the src/ directory
*/

import { useState, useEffect } from 'react';                // React hooks for state management
import { submitPrediction, checkStatus } from '@/lib/api';   // Our API functions
import type { PredictionResponse, StatusResponse } from '@/lib/api';        // TypeScript type for the response

export default function Home() {

  // ================================ FORM INPUT STATE ===============================

  const [ticker, setTicker] = useState('AAPL');                   // Default AAPL ticker

  const [startDate, setStartDate] = useState('2024-01-01');       // Default January 1, 2024 start

  const [endDate, setEndDate] = useState('2025-01-01');           // Default January 1, 2025 end

  const [includeSentiment, setIncludeSentiment] = useState(true); // Default include sentiment == true

  // ================================= REQUEST STATE =================================

  const [requestId, setRequestId] = useState<string | null>(null);    // These lines actually instantiate the function like setRequestId to which variable requestId is updated
  // This can be a string OR null, nothing else.  Also, random, but this triggers a re-render since its a React setter

  const [status, setStatus] = useState<string | null>(null);
  // Current status: 'PENDING' | 'PROCESSING' | 'COMPLETED' | 'FAILED' | null

  const [isLoading, setIsLoading] = useState(false);

  // ================================= RESULTS STATE =================================

  const [results, setResults] = useState<StatusResponse | null>(null);

  const [error, setError] = useState<string | null>(null);

  // ============================ FORM SUBMISSION HANDLER ============================

  const handleSubmit = async (e: React.FormEvent) => {
    // Prevent default form behavior (page refresh)
    e.preventDefault();

    // Reset previous results and errors
    setError(null);       // Clearing any old error messages
    setResults(null);     // Clearing any old prediciton results
    setStatus(null);      // Clear old status
    setIsLoading(true);   // Shows loading spinner

    const MAX_ATTEMPTS = 2;
    let attempt = 0;

    while (attempt < MAX_ATTEMPTS) {
      try {
        attempt++;
        // Cold start possibly encountered
        if (attempt > 1) {
          console.log(`Retrying attempt ${attempt}/${MAX_ATTEMPTS}...`);
        }
        // Calls our API function from api.ts
        // Does: POST /predict with our form data
        const response = await submitPrediction({
          ticker,
          start_date: startDate,
          end_date: endDate,
          include_sentiment: includeSentiment
        });

        // If we're here, it was a success: we got request_id back!
        setRequestId(response.request_id);
        setStatus('PENDING');
        return;

      } catch (err) {
        // Checking if it's a timeout or cold start error
        const errorMessage = err instanceof Error ? err.message : String(err);
        const isTimeoutError = errorMessage.includes('timeout') ||
                               errorMessage.includes('504') ||
                               errorMessage.includes('Gateway');
        
        // If it's the last attempt OR not a timeout, give up
        if (attempt >= MAX_ATTEMPTS || !isTimeoutError) {
          setIsLoading(false);
          setError(err instanceof Error ? err.message : `Failed to submit request after ${attempt} attempts.`);
          return;
        }

        // Otherwise, wait a little bit before retrying for cold start issue
        console.log(`Attempt ${attempt} failed: likely cold start, retrying in 5 seconds...`);
        await new Promise(resolve => setTimeout(resolve, 5000));
      }
    }
  };

  // ================================= POLLING LOGIC =================================

  useEffect(() => {
    // Code runs when dependencies change (Observer!)
    // If no requestId, don't poll
    if (!requestId) return;

    // If already completed or failed, don't poll
    if (status === 'COMPLETED' || status === 'FAILED') return;

    console.log('Starting to poll for request: ', requestId);

    let pollCount = 0;        // Poll count to ensure that if a request fails or is hung up, to stop
    const MAX_POLLS = 20;     // 20 * 3 seconds = 1 minute

    const pollStatus = async () => {
      pollCount++;

      if (pollCount > MAX_POLLS) {
        clearInterval(pollInterval);
        setIsLoading(false);
        setError('Prediction timed out after 1 minute. Please try again with at least 6 months of training data.')
        return;
      }

      try {
        // Call our API function from api.ts, does GET /status/{requestId}
        const response = await checkStatus(requestId);

        console.log('Poll response: ', response.status);

        setStatus(response.status);

        if (response.status === 'COMPLETED') {
          setResults(response);
          setIsLoading(false);
          clearInterval(pollInterval);  // Stop polling
          console.log('Prediction successfully completed!');

        } else if (response.status === 'FAILED') {
          setError('Prediction failed! Check your ticker and/or please try again with at least 6 months of training data.');
          setIsLoading(false);
          clearInterval(pollInterval);  // Stop polling
          console.log('Prediction failed');
        }
      } catch (err) {
        console.error('Polling error: ', err);
        setError(err instanceof Error ? err.message : 'Failed to check status');
        setIsLoading(false);
        clearInterval(pollInterval);
      }
    };

    // Start polling immediately 
    pollStatus();

    // Poll every 3 seconds
    const pollInterval = setInterval(pollStatus, 3000);

    // Cleanup: When component unmounts (navigates away, closes tab, refreshes page) or requestId changes
    return () => {
      console.log('Stopping polling');
      clearInterval(pollInterval);
    };
    // Removed status from dependencies so website doesn't reload with every new status update (caused UX errors)
  }, [requestId]);

  return (
    /*
    - min-h-screen: Full viewport, height minimum (minimum: viewport height, but grow taller if content needs; i.e. content fills screen not stretches past)
    - bg-gradient-to-br: Background gradient form top-left to bottom-right
    - from-gray-900 to-gray-900: Dark gray gradient colors
    - p-8: Padding of 32 px (2rem) all round (Tailwind spacing scale...  annoying)
    - Note: "rem" is relative to the ROOT (html) font size (i.e. user sets font size to 20px from 40?  REM scales with it)
    */
    <main className = "min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 text-white p-8">
      {/*Container*/}
      
      <div className = "max-w-4xl mx-auto">

        { /* Header */ }
        <div className = "text-center mb-12">
          <h1 className = "text-4xl font-bold mb-4">
            StockForecaster
          </h1>
          <p className = "text-gray-400 text-lg">
            LSTM-based 20-day stock price forecasting with sentiment analysis
          </p>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className = "bg-gray-800 rounded-lg p-8 shadow-2xl mb-8">

          {/* Step 1: Stock Ticker Input*/ }
          <div className="mb-6">
            <label htmlFor = "ticker" className = "block text-sm font-medium mb-2">
              Stock Ticker Symbol
            </label>
            <input 
              type = "text"
              id = "ticker"
              value = {ticker}
              onChange = {(e) => setTicker(e.target.value.toUpperCase())}
              placeholder = "AAPL"
              className = "w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
              required
            />
          </div>

          {/* Step 2: Start Date Input */}
          <div className = "mb-6">
            <label htmlFor = "startDate" className = "block text-sm font-medium mb-2">
              Start Date
            </label>
            <input 
              type = "date"
              id = "startDate"
              value = {startDate}
              onChange = {(e) => setStartDate(e.target.value)}
              className = "w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              required
            />
          </div>

          {/* Step 3: End Date Input */}
          <div className = "mb-6">
            <label htmlFor = "startDate" className = "block text-sm font-medium mb-2">
              End Date
            </label>
            <input 
              type = "date"
              id = "endDate"
              value = {endDate}
              onChange = {(e) => setEndDate(e.target.value)}
              className = "w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              required
            />
          </div>

          {/* Step 4: Sentiment Check */}
          <div className = "mb-8 flex items-center">
            <input
              type = "checkbox"
              id = "includeSentiment"
              checked = {includeSentiment}
              onChange = {(e) => setIncludeSentiment(e.target.checked)}
              className = "w-5 h-5 bg-gray-700 border-gray-600 rounded focus:ring-2 focus:ring-blue-500"
            />
            <label htmlFor = "includeSentiment" className = "ml-3 text-sm">
              Include sentiment analysis (scrapes recent news)
            </label>
          </div>

          {/* Step 5: Submit Button, submit request form to API */}
          <button 
            type = "submit"
            disabled = {isLoading}
            className = "w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-semibold py-4 px-6 rounded-lg transition-colors duration-200"
          >
            {isLoading ? 'Training Model...' : 'Get Prediction'}
          </button>
        </form>

        {/* Loading State */}
        {isLoading && (
          <div className = "text-center p-8">
            <div className = "inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-white mb-4"></div>
            <p className = "text-lg">{status === 'PENDING' ? 'Submitting request...' : 'Training model...'}</p>
          </div>
        )}

        {/* Error State */} 
        {error && (
          <div className = "bg-red-900 border border-red-700 text-white px-6 py-4 rounded-lg">
            <p className = "font-semibold">Error:</p>
            <p>{error}</p>
          </div>
        )}

        {/* Results Display */}
        {results && results.result && results.request_id === requestId && (
          <div className="bg-gray-800 rounded-lg p-8 shadow-2xl">
            <h2 className="text-3xl font-bold mb-6">{results.result.ticker} 4-Week Price Forecast</h2>
            
            <div className="grid grid-cols-2 gap-6 mb-6">
              <div>
                <p className="text-gray-400 text-sm">Current Price</p>
                <p className="text-2xl font-bold">{results.result.prediction.current_price}</p>
              </div>
              <div>
                <p className="text-gray-400 text-sm">20-Day Prediction</p>
                <p className="text-2xl font-bold">{results.result.prediction.predicted_price_20d}</p>
              </div>
              <div>
                <p className="text-gray-400 text-sm">Expected Change</p>
                <p className={`text-2xl font-bold ${parseFloat(results.result.prediction.predicted_change_20d) > 0 
                  ? 'text-green-400' : 'text-red-400'}`}>
                    {results.result.prediction.predicted_change_20d}
                </p>
              </div>
              <div>
                <p className="text-gray-400 text-sm">Model Accuracy (100% - MAPE)</p>
                <p className="text-2xl font-bold">{100.00 - parseFloat(results.result.model_performance.mape)}%</p>
              </div>
              {results.result.sentiment && (
                <div>
                  <p className="text-gray-400 text-sm">Average Sentiment</p>
                  <p className={`text-2xl font-bold ${parseFloat(results.result.sentiment.average_sentiment) > 0 
                    ? 'text-green-400' : 'text-red-400'}`}>
                      {(parseFloat(results.result.sentiment.average_sentiment) * 100).toFixed(2)}%
                    </p>
                </div>
              )}
            </div>
          </div>
        )}
        </div>
    </main>
  );
};