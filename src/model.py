import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Multiply, Input, AdditiveAttention
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

def build_model(X_train):
    # Define and compile the model
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
    return model

def train_model(model, X_train, y_train):
    # Train the model
    early_stopping = EarlyStopping(monitor="val_loss", patience=10)
    history = model.fit(X_train, y_train, epochs=100, batch_size=25, validation_split=0.2, callbacks=[early_stopping])
    return model, history

def evaluate_model(model, X_test, y_test):
    # Evaluate the model
    test_loss = model.evaluate(X_test, y_test)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return test_loss, mae, rmse

def predict(model, X_latest, scaler):
    # Make predictions
    predicted_stock_price = model.predict(X_latest)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

    predicted_prices = []
    current_batch = X_latest

    for i in range(4):
        next_prediction = model.predict(current_batch)
        next_prediction_reshaped = next_prediction.reshape(1, 1, 1)
        current_batch = np.append(current_batch[:, 1:, :], next_prediction_reshaped, axis=1)
        predicted_prices.append(scaler.inverse_transform(next_prediction)[0, 0])

    return predicted_stock_price, predicted_prices
