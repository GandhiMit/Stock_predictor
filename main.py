import streamlit as st
from keras.models import Sequential, Model
from keras.layers import (
    LSTM, Dense, Dropout, AdditiveAttention, Permute, Reshape,
    Multiply, BatchNormalization, Input, Concatenate
)
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
from datetime import date, timedelta

st.title("Stock and Cryptocurrency Price Prediction")

# Default values
DEFAULT_COMPANY = "TCS.NS"
DEFAULT_START_DATE = date(2020, 1, 1)
DEFAULT_END_DATE = date(2024, 6, 10)
DEFAULT_START_DATE_PREDICTION = date(2024, 3, 7)
DEFAULT_END_DATE_PREDICTION = date(2024, 6, 11)
DEFAULT_FACTOR = 28

# User inputs
asset_type = st.selectbox("Asset Type", ["Stock", "Cryptocurrency"])
company = st.text_input("Company/Crypto Symbol", value=DEFAULT_COMPANY)
start_date = st.date_input("Start Date for Training Data", value=DEFAULT_START_DATE)
end_date = st.date_input("End Date for Training Data", value=DEFAULT_END_DATE)
start_date_prediction = st.date_input("Start Date for Prediction Data", value=DEFAULT_START_DATE_PREDICTION)
end_date_prediction = st.date_input("End Date for Prediction Data", value=DEFAULT_END_DATE_PREDICTION)
factor = st.number_input("Average-Factor", value=DEFAULT_FACTOR)

# Function to create a date range excluding weekends
def generate_prediction_dates(start_date, num_days):
    dates = []
    current_date = start_date
    while len(dates) < num_days:
        if asset_type == "Cryptocurrency" or current_date.weekday() < 5:
            dates.append(current_date)
        current_date += timedelta(days=1)
    return dates

# Function to load and preprocess data, train model, and make predictions
def run_model():
    # Load data
    @st.cache_data
    def load_data(company, start, end):
        return yf.download(company, start=start, end=end)

    data = load_data(company, start_date, end_date)

    # Display data
    st.subheader("Stock/Crypto Data")
    st.write(data.head())

    # Check for missing values
    if data.isnull().sum().any():
        data.fillna(method="ffill", inplace=True)

    # Preprocess data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1, 1))

    X, y = [], []
    for i in range(factor, len(scaled_data)):
        X.append(scaled_data[i - factor : i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Define the model
    input_layer = Input(shape=(X_train.shape[1], 1))
    lstm_out = LSTM(50, return_sequences=True)(input_layer)
    lstm_out = LSTM(50, return_sequences=True)(lstm_out)

    # Attention mechanism
    query = Dense(50)(lstm_out)
    value = Dense(50)(lstm_out)
    attention_out = AdditiveAttention()([query, value])

    # Combine LSTM output with attention output
    multiply_layer = Multiply()([lstm_out, attention_out])

    # Flatten and output
    flatten_layer = tf.keras.layers.Flatten()(multiply_layer)
    output_layer = Dense(1)(flatten_layer)

    # Compile model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.summary()

    # Train the model
    early_stopping = EarlyStopping(monitor="val_loss", patience=10)
    history = model.fit(X_train, y_train, epochs=100, batch_size=25, validation_split=0.2, callbacks=[early_stopping])

    # Evaluate the model
    test_loss = model.evaluate(X_test, y_test)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    st.subheader("Model Evaluation")
    st.write(f"Test Loss: {test_loss}")
    st.write(f"Mean Absolute Error: {mae}")
    st.write(f"Root Mean Square Error: {rmse}")

    # Fetching the latest factor days of stock data for prediction
    data = load_data(company, start_date_prediction, end_date_prediction)
    st.write("Latest Stock/Crypto Data for Prediction")
    st.write(data.tail())

    closing_prices = data["Close"].values

    # Scaling the data
    scaled_data = scaler.fit_transform(closing_prices.reshape(-1, 1))

    # Reshaping the data for the model
    X_latest = np.array([scaled_data[-factor:].reshape(factor)])
    X_latest = np.reshape(X_latest, (X_latest.shape[0], X_latest.shape[1], 1))

    # Predicting the next 4 days
    predicted_stock_price = model.predict(X_latest)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

    st.write("Predicted Prices for the next day: ", predicted_stock_price)

    # Predicting the next 4 days iteratively
    predicted_prices = []
    current_batch = scaled_data[-factor:].reshape(1, factor, 1)

    for i in range(4):
        next_prediction = model.predict(current_batch)
        next_prediction_reshaped = next_prediction.reshape(1, 1, 1)
        current_batch = np.append(current_batch[:, 1:, :], next_prediction_reshaped, axis=1)
        predicted_prices.append(scaler.inverse_transform(next_prediction)[0, 0])

    st.write("Predicted Prices for the next 4 days: ", predicted_prices)

    # Creating a DataFrame for the predictions
    last_date = data.index[-1]
    next_day = last_date + timedelta(days=1)
    prediction_dates = generate_prediction_dates(next_day, 4)
    predictions_df = pd.DataFrame(index=prediction_dates, data=predicted_prices, columns=["Close"])

    # Displaying the predicted prices with dates
    st.subheader("Predicted Prices with Dates")
    st.write(predictions_df)

    # Plotting the actual data with predictions
    st.subheader("Price Prediction")

    # Combining both actual and predicted data
    combined_data = pd.concat([data["Close"], predictions_df["Close"]])
    combined_data = combined_data[-(factor + 4):]

    # Plotting the combined actual and predicted data
    plt.figure(figsize=(10, 6))
    plt.plot(data.index[-factor:], data["Close"][-factor:], linestyle="-", marker="o", color="blue", label="Actual Data")
    plt.plot(prediction_dates, predicted_prices, linestyle="-", marker="o", color="red", label="Predicted Data")
    plt.title(f"{company} Price: Last {factor} Days and Next 4 Days Predicted")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(plt)

# Add an OK button to start the process
if st.button("OK"):
    run_model()
