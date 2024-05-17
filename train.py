import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from preprocess.price_plot import plot_predicted_close


#feature_cols = ["Open", "High", "Low", "Close", "Adj Close", 
#                "Volume", "change", "candle_stick", "tweets_count"]
feature_cols = [ "Open", "High", "Low", "Close", "candle_stick"]

def create_features(df, window_size=5):  # Using 5 days ~ 1 week
    """
    Generate rolling window features and targets (next day's 'Open').
    """
    
    df_features = df[feature_cols]
    
    # Rolling window features
    rolling_features = df_features.rolling(window=window_size)
    
    # Target: Next day's 'Open' price
    df['target'] = df['Open'].shift(-1)
    
    # Combining features and removing NaN rows created by rolling and shifting
    df_combined = pd.concat([rolling_features, df['target']], axis=1).dropna()
    
    return df_combined

def linear_fit(df_filtered, diagram=False):

    df_features = create_features(df_filtered)

    X = df_features.drop('target', axis=1)
    y = df_features['target']
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ### Step 2: Creating and Training the Model

    #Now, train a Linear Regression model on your training data:
    

    

    model = LinearRegression()
    model.fit(X_train, y_train)


    ### Step 3: Making Predictions and Evaluating the Model 

    #Make predictions on the testing data and evaluate the model's performance:

    y_pred = model.predict(X_test)
    if diagram is True:
        plot_predicted_close(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")


def create_classification_features(df, window_size=90):  # Using 90 days ~ 3 months
    """
    Generate rolling window features and classify if next day's 'Open' is higher (1) or lower (0).
    """
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df_features = df[feature_cols]
    
    # Rolling window features
    rolling_features = df_features.rolling(window=window_size).mean().shift(1)
    
    # Classification target: If next day's 'Open' is higher than today's
    df['target'] = (df['Open'].shift(-1) > df['Open']).astype(int)
    
    # Combining features and removing NaN rows created by rolling and shifting
    df_combined = pd.concat([rolling_features, df['target']], axis=1).dropna()
    
    return df_combined

def model_train(df_filtered):
    # Preparing the dataset
    df_features = create_classification_features(df_filtered)

    X = df_features.drop('target', axis=1)
    y = df_features['target']

    # Splitting the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    return classification_report(y_test, y_pred)