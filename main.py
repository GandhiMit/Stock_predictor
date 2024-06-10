from keras.models import Sequential
from keras.layers import (
    LSTM,
    Dense,
    Dropout,
    AdditiveAttention,
    Permute,
    Reshape,
    Multiply,
)
from sklearn.preprocessing import MinMaxScaler
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np

# Check TensorFlow version
print("TensorFlow Version: ", tf.__version__)
company = "TSLA"
# Fetch AAPL data
company_data = yf.download(company, start="2020-01-01", end="2024-06-01")

# Display the first few rows of the dataframe
company_data.head()

# Checking for missing values
company_data.isnull().sum()

# Filling missing values, if any
company_data.fillna(method="ffill", inplace=True)


scaler = MinMaxScaler(feature_range=(0, 1))
aapl_data_scaled = scaler.fit_transform(company_data["Close"].values.reshape(-1, 1))

X = []
y = []

for i in range(60, len(aapl_data_scaled)):
    X.append(aapl_data_scaled[i - 60 : i, 0])
    y.append(aapl_data_scaled[i, 0])

train_size = int(len(X) * 0.8)
test_size = len(X) - train_size

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


model = Sequential()

# Adding LSTM layers with return_sequences=True
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=True))

# Adding self-attention mechanism
# The attention mechanism
attention = AdditiveAttention(name="attention_weight")
# Permute and reshape for compatibility
model.add(Permute((2, 1)))
model.add(Reshape((-1, X_train.shape[1])))
attention_result = attention([model.output, model.output])
multiply_layer = Multiply()([model.output, attention_result])
# Return to original shape
model.add(Permute((2, 1)))
model.add(Reshape((-1, 50)))

# Adding a Flatten layer before the final Dense layer
model.add(tf.keras.layers.Flatten())

# Final Dense layer
model.add(Dense(1))

# Compile the model
# model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
# history = model.fit(X_train, y_train, epochs=100, batch_size=25, validation_split=0.2)


# Adding Dropout and Batch Normalization
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.compile(optimizer="adam", loss="mean_squared_error")

model.summary()

# Assuming X_train and y_train are already defined and preprocessed
# history = model.fit(X_train, y_train, epochs=10, batch_size=25, validation_split=0.2)


early_stopping = EarlyStopping(monitor="val_loss", patience=10)
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=25,
    validation_split=0.2,
    callbacks=[early_stopping],
)


# Convert X_test and y_test to Numpy arrays if they are not already
X_test = np.array(X_test)
y_test = np.array(y_test)

# Ensure X_test is reshaped similarly to how X_train was reshaped
# This depends on how you preprocessed the training data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Now evaluate the model on the test data
test_loss = model.evaluate(X_test, y_test)
print("Test Loss: ", test_loss)


# Making predictions
y_pred = model.predict(X_test)

# Calculating MAE and RMSE
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print("Mean Absolute Error: ", mae)
print("Root Mean Square Error: ", rmse)


# Fetching the latest 60 days of AAPL stock data
data = yf.download(company, start="2024-03-07", end="2024-06-9")
print(data)

# Selecting the 'Close' price and converting to numpy array
closing_prices = data["Close"].values

# Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(closing_prices.reshape(-1, 1))

# Since we need the last 60 days to predict the next day, we reshape the data accordingly
X_latest = np.array([scaled_data[-60:].reshape(60)])

# Reshaping the data for the model (adding batch dimension)
X_latest = np.reshape(X_latest, (X_latest.shape[0], X_latest.shape[1], 1))

# Making predictions for the next 4 candles
predicted_stock_price = model.predict(X_latest)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

print("Predicted Stock Prices for the next 4 days: ", predicted_stock_price)


# -----------------------------------------------------------------------------


# Fetch the latest 60 days of AAPL stock data
data = yf.download(company, start="2024-03-07", end="2024-06-9")

# Select 'Close' price and scale it
closing_prices = data["Close"].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(closing_prices)

# Predict the next 4 days iteratively
predicted_prices = []
current_batch = scaled_data[-60:].reshape(1, 60, 1)  # Most recent 60 days

for i in range(4):  # Predicting 4 days
    # Get the prediction (next day)
    next_prediction = model.predict(current_batch)

    # Reshape the prediction to fit the batch dimension
    next_prediction_reshaped = next_prediction.reshape(1, 1, 1)

    # Append the prediction to the batch used for predicting
    current_batch = np.append(current_batch[:, 1:, :], next_prediction_reshaped, axis=1)

    # Inverse transform the prediction to the original price scale
    predicted_prices.append(scaler.inverse_transform(next_prediction)[0, 0])

print("Predicted Stock Prices for the next 4 days: ", predicted_prices)

# -------------------------------------------------------------------------------


# Assuming 'data' is your DataFrame with the fetched AAPL stock data
# Make sure it contains Open, High, Low, Close, and Volume columns

# Creating a list of dates for the predictions
last_date = data.index[-1]
next_day = last_date + pd.Timedelta(days=1)
prediction_dates = pd.date_range(start=next_day, periods=4)

# Assuming 'predicted_prices' is your list of predicted prices for the next 4 days
predictions_df = pd.DataFrame(
    index=prediction_dates, data=predicted_prices, columns=["Close"]
)

# Plotting the actual data with mplfinance
mpf.plot(data, type="candle", style="charles", volume=True)

# Overlaying the predicted data
plt.figure(figsize=(10, 6))
plt.plot(
    predictions_df.index,
    predictions_df["Close"],
    linestyle="dashed",
    marker="o",
    color="red",
)

plt.title(f"{company} Stock Price with Predicted Next 4 Days")
plt.show()


# Fetch the latest 60 days of AAPL stock data
data = yf.download(
    company, start="2024-03-07", end="2024-06-9"
)  # Fetch 64 days to display last 60 days in the chart

# Select 'Close' price and scale it
closing_prices = data["Close"].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(closing_prices)

# Predict the next 4 days iteratively
predicted_prices = []
current_batch = scaled_data[-60:].reshape(1, 60, 1)  # Most recent 60 days

for i in range(4):  # Predicting 4 days
    next_prediction = model.predict(current_batch)
    next_prediction_reshaped = next_prediction.reshape(1, 1, 1)
    current_batch = np.append(current_batch[:, 1:, :], next_prediction_reshaped, axis=1)
    predicted_prices.append(scaler.inverse_transform(next_prediction)[0, 0])

# Creating a list of dates for the predictions
last_date = data.index[-1]
next_day = last_date + pd.Timedelta(days=1)
prediction_dates = pd.date_range(start=next_day, periods=4)

# Adding predictions to the DataFrame
predicted_data = pd.DataFrame(
    index=prediction_dates, data=predicted_prices, columns=["Close"]
)

# Combining both actual and predicted data
combined_data = pd.concat([data["Close"], predicted_data["Close"]])
combined_data = combined_data[
    -64:
]  # Last 60 days of actual data + 4 days of predictions

# Plotting the actual data
plt.figure(figsize=(10, 6))
plt.plot(
    data.index[-60:],
    data["Close"][-60:],
    linestyle="-",
    marker="o",
    color="blue",
    label="Actual Data",
)

# Plotting the predicted data
plt.plot(
    prediction_dates,
    predicted_prices,
    linestyle="-",
    marker="o",
    color="red",
    label="Predicted Data",
)

plt.title(f"{company} Stock Price: Last 60 Days and Next 4 Days Predicted")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()
