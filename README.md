# yannixmaxfinalproject
Leveraging deep learning techniques, this model predicts the stock prices of Apple, Amazon, Google, and Microsoft with high accuracy, enabling informed investment decisions.
#  Overview
This repository provides source code for a stock price prediction application that makes use of Long Short-Term Memory (LSTM) networks. The app concentrates on four technology stocks: AAPL (Apple), GOOG (Google), MSFT (Microsoft), and AMZN (Amazon). The prediction model is constructed using historical stock price data from Yahoo Finance.
# Functionalities
1. Data Collection: Retrieves historical stock price data from Yahoo Finance for the selected firms.
2. Data Preprocessing: This step prepares the data for training the LSTM model, including scaling and sequence creation.
3. Model Training: Creates and trains LSTM models using stock data from each firm.

4. GridSearchCV is used to discover ideal hyperparameters for the LSTM model during hyperparameter tuning.

5. Stock Price Prediction: Estimates future stock prices and assesses model performance.
 6. Plots real stock prices, forecasted stock prices, and model assessment metrics.
# Features
LSTM neural network for time series prediction.
Hyperparameter tuning using GridSearchCV.
Data visualization using Matplotlib and Seaborn
# Technologies Used
Python
Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, yfinance, Keras (with TensorFlow backend), google.colab
Jupyter Notebook (Google Colab environment)
# How to Use
Open the Jupyter Notebook in a Google Colab environment.
Run each code cell sequentially to fetch data, preprocess it, train the LSTM model, and make predictions.
Visualize the results and evaluation metrics for each stock.
# Note
nsure that the necessary libraries are installed using !pip install scikeras at the beginning of the notebook.
Individual models for each stock are saved using pickle. You can load them later for further analysis.
Feel free to explore and modify the code to suit your needs. Happy predicting!
