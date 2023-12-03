import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import yfinance as yf
from datetime import datetime, timedelta


# Function to load the pre-trained LSTM model
def load_lstm_model(model_path):
    return load_model(model_path)


# Function to preprocess data and make predictions
def predict_stock_price(model, scaler, stock_data, interval):
    # Scale the data
    scaled_data = scaler.transform(stock_data)

    # Create sequences with the specified interval
    x_test, y_test = create_sequences(scaled_data, interval)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Get the models predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    return predictions


# Function to create sequences with a given interval
def create_sequences(data, sequence_length):
    x, y = [], []
    for i in range(len(data) - sequence_length):
        x.append(data[i : (i + sequence_length), 0])
        y.append(data[i + sequence_length, 0])
    return np.array(x), np.array(y)


# Streamlit app
def main():
    st.title("Stock Price Prediction App")

    st.markdown(
        "This app predicts the closing stock price of a selected company using LSTM."
    )

    # Select a company
    selected_company = st.selectbox(
        "Select a company", ["AAPL", "GOOG", "MSFT", "AMZN"]
    )

    # Date range for prediction
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    # Download stock data
    stock_data = yf.download(selected_company, start_date, end_date)

    # Create a new dataframe with only the 'Close' column
    stock_data = stock_data.filter(["Close"])

    # Convert the dataframe to a numpy array
    dataset = stock_data.values

    # Load the pre-trained model and scaler
    model_path = f"{selected_company.lower()}_best_grid_model.pkl"  # Update with your model path
    model = load_lstm_model(model_path)

    # Create the scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Prediction interval (should match the interval used during training)
    prediction_interval = 30

    # Make predictions
    predictions = predict_stock_price(model, scaler, scaled_data, prediction_interval)

    # Display historical and predicted prices
    st.subheader(f"{selected_company} Stock Price Prediction")
    st.line_chart(pd.DataFrame({"Close": dataset.flatten()}, index=stock_data.index))
    st.line_chart(
        pd.DataFrame({"Predictions": predictions.flatten()}, index=stock_data.index)
    )


if __name__ == "__main__":
    main()
