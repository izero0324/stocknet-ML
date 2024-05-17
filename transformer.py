import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, MultiHeadAttention, Dropout, LayerNormalization, Input, GlobalAveragePooling1D
from tensorflow.keras import Model

from preprocess.price_plot import LSTM_plot  # Assuming you have this from your previous setup

def create_transformer_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks):
    inputs = Input(shape=input_shape)
    x = inputs
    
    for _ in range(num_transformer_blocks):
        # Multi Head Self Attention
        x1 = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=0.1)(x, x)
        x1 = Dropout(0.1)(x1)
        x = LayerNormalization(epsilon=1e-6)(x + x1)
        
        # Feed Forward Network (FFN)
        x2 = Dense(ff_dim, activation="relu")(x)
        x2 = Dropout(0.1)(x2)
        x2 = Dense(input_shape[-1])(x2)
        x = LayerNormalization(epsilon=1e-6)(x + x2)
    
    x = GlobalAveragePooling1D(data_format="channels_last")(x)
    x = Dropout(0.1)(x)
    output = Dense(1, activation="linear")(x)
    
    return Model(inputs, output)

def create_dataset(dataset, time_step=60):
    X, Y = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)


def transformer_fit(filtered_data):
    filtered_data.set_index("date", inplace=True)
    print(filtered_data.head())
    # Scale the Open prices
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(filtered_data['Open'].values.reshape(-1,1))

    # Create the dataset suitable for time series forecasting
    time_step = 60
    X, y = create_dataset(scaled_data, time_step)

    # Reshaping the input to be [samples, time steps, features] 
    X = X.reshape(X.shape[0], X.shape[1], 1)

    training_size = int(len(X) * 0.67)
    test_size = len(X) - training_size
    X_train, X_test = X[:training_size], X[training_size:]
    y_train, y_test = y[:training_size], y[training_size:]

    input_shape = X_train.shape[1:]
    model = create_transformer_model(input_shape, head_size=64, num_heads=4, ff_dim=64, num_transformer_blocks=2)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()

    # Train the model
    model.fit(X_train, y_train, batch_size=16, epochs=10)

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions) 

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    print(f'RMSE: {rmse}')
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    test_start_index = len(filtered_data) - len(y_test)
    test_dates = filtered_data.index[test_start_index:]

    prediction_dates = test_dates[:len(predictions)]
    LSTM_plot(prediction_dates, y_test_rescaled, predictions)  # You might need to adjust this for transformer

    debug = (y_test_rescaled,predictions)

    return model, debug