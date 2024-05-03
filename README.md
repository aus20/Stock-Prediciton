# Stock Price Prediction with LSTM

This project uses Long Short-Term Memory (LSTM) neural networks to predict stock prices based on historical data. The model is trained on past stock prices and can forecast future prices.

## Overview

The goal of this project is to demonstrate how LSTM networks can be used for time series forecasting, particularly in the context of stock price prediction. The model is trained on historical stock price data and then tested on unseen data to evaluate its predictive performance.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
git clone https://github.com/aus20/Stock-Prediction.git

markdown
Copy code
2. Install the required dependencies:
pip install -r requirements.txt

css
Copy code
3. Run the main Python script:
python stock_prediction.py

markdown
Copy code

## Usage

1. Modify the `ticker`, `start_date`, and `end_date` variables in `stock_prediction.py` to specify the stock symbol and the date range for the data.
2. Run the `stock_prediction.py` script to train the LSTM model and make predictions.
3. View the results, including plots of actual vs. predicted stock prices and the calculated RMSE, in the console output and generated plots.

## Dependencies

- numpy
- matplotlib
- scikit-learn
- keras
- yfinance
