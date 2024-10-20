# Stock Price Prediction using LSTM and ARIMA

This Jupyter Notebook implements time series forecasting for stock prices using two approaches: **Long Short-Term Memory (LSTM) networks** and **ARIMA models**. The project aims to predict Apple stock prices, comparing the performance of both models using key metrics like **RMSE** (Root Mean Squared Error) and **MAE** (Mean Absolute Error).

## Key Features

1. **Data Preparation**:
   - Uses raw Apple stock closing prices.
   - Splits data into training and testing sets.

2. **LSTM Model**:
   - Implements an LSTM network for stock price prediction.
   - Includes data scaling, model training, and inverse scaling for results interpretation.
   - Forecasts future stock prices and evaluates model performance on the test data.

3. **ARIMA Model**:
   - Performs a grid search to find the optimal ARIMA parameters (`p`, `d`, `q`).
   - Trains the ARIMA model on the same dataset and makes predictions for both the test period and a future time frame.
   - Selects the best ARIMA model based on the **AIC** (Akaike Information Criterion).
   - Calculates RMSE and MAE for both the training and testing periods.

4. **Comparison and Visualization**:
   - Plots actual stock prices alongside LSTM and ARIMA predictions.
   - Visualizes a comparison of LSTM and ARIMA models for the test set and future forecasts.

## Dependencies

- `pandas`, `numpy`, `matplotlib` for data handling and plotting.
- `statsmodels` for ARIMA modeling.
- `scikit-learn` for error metric calculations.
- LSTM implementation using `Keras` (or similar deep learning framework).

## How to Use

1. Ensure all required dependencies are installed.
2. Load the Apple stock price dataset.
3. Run the cells to train the LSTM and ARIMA models, generate predictions, and visualize the results.

## Results

The notebook provides **RMSE** and **MAE** scores for both training and testing sets, helping you assess the performance of the models for stock price prediction.
