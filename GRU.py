import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU

from preprocess.price_plot import LSTM_plot  # Consider renaming this to a more generic name, like "price_plot"

def create_dataset(dataset, time_step=60):
    X, Y = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

def GRU_fit(filtered_data):
    filtered_data.set_index("date", inplace=True)
    print(filtered_data.head())
    # Scale the Open prices
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(filtered_data['Open'].values.reshape(-1,1))

    # Create the dataset suitable for time series forecasting
    time_step = 60
    X, y = create_dataset(scaled_data, time_step)

    # Reshaping the input to be [samples, time steps, features] which is required for GRU
    X = X.reshape(X.shape[0],X.shape[1], 1)

    ### Split the Data
    training_size = int(len(X) * 0.67)
    test_size = len(X) - training_size
    X_train, X_test = X[:training_size], X[training_size:]
    y_train, y_test = y[:training_size], y[training_size:]

    ### Build the GRU Model
    feature_count = X_train.shape[2]

    # Build the GRU model
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=(60,feature_count)),
        GRU(64, return_sequences=False),
        Dense(32),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, batch_size=1, epochs=1)

    ### Make Predictions and Evaluate the Model
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)  # Undo scaling

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
    LSTM_plot(prediction_dates,y_test_rescaled,predictions)  # Consider renaming this function to a more generic name as mentioned above

    return 0