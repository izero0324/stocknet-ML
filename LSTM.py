import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

from preprocess.price_plot import LSTM_plot

# Suppose your DataFrame (filtered_data) has columns: Date, Open, High, Low, Close, Volume



def create_dataset(dataset, time_step=60):
    X, Y = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)


def LSTM_fit(filtered_data):
    filtered_data.set_index("date", inplace=True)
    print(filtered_data.head())
    # Scale the Open prices
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(filtered_data['Open'].values.reshape(-1,1))

    # Create the dataset suitable for time series forecasting
    time_step = 60
    X, y = create_dataset(scaled_data, time_step)

    # Reshaping the input to be [samples, time steps, features] which is required for LSTM
    X = X.reshape(X.shape[0],X.shape[1], 1)


    ### Step 4: Split the Data

    training_size = int(len(X) * 0.67)
    test_size = len(X) - training_size
    X_train, X_test = X[:training_size], X[training_size:]
    y_train, y_test = y[:training_size], y[training_size:]


    ### Step 5: Build the LSTM Model
    feature_count = X_train.shape[2]

    # Build the LSTM model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(60,feature_count)),
        LSTM(64, return_sequences=False),
        Dense(32),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, batch_size=1, epochs=1)


    ### Step 6: Make Predictions and Evaluate the Model


    # Making predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)  # Undo scaling

    # Let's compare with the actual values. For a comprehensive evaluation, use metrics like RMSE.

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    print(f'RMSE: {rmse}')
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate the starting index of your y_test dataset within the original dataframe
    test_start_index = len(filtered_data) - len(y_test)

    # Get the dates corresponding to your y_test dataset
    test_dates = filtered_data.index[test_start_index:]

    # Ensure we align with the length of predictions and y_test; take the date range accordingly
    prediction_dates = test_dates[:len(predictions)]
    LSTM_plot(prediction_dates,y_test_rescaled,predictions)

    return 0

