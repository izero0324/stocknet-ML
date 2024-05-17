import pandas as pd
from preprocess.csv_reader import read
from preprocess.tweet import count_tweets_for_date

def clean_stock_data(stock_name):
    # Read the CSV file is named after the stock and located in 'price/raw/' folder
    
    file_name = stock_name+'.csv'
    df = read(file_name)
    # Perform cleaning operations
    missing_data(df,stock_name)
    df_dropna = df.dropna()
    
    # Correcting a specific data type
    df_dropna['date'] = pd.to_datetime(df_dropna['Date'])

    # Filter 2014-2015
    start_date = '2014-01-01'
    end_date = '2015-12-31'
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    df_tweet = df_dropna.loc[mask].copy() 
    df_tweet['tweets_count'] = df_tweet['Date'].apply(lambda x: count_tweets_for_date(stock_name, x))

    # Adding 'change' column
    df_tweet['change'] = 100*(df_tweet['Open'] - df_tweet['Close'])/df_tweet['Open']

    # Adding 'candle_stick' column
    df_tweet['candle_stick'] = df_tweet['High'] - df_tweet['Low']

    # ReOrdering the columns
    df_tweet = adjust_column_order(df_tweet)

    return df_tweet

def missing_data(df,stock_name):
    # Check for missing data in each column
    missing_data = df.isnull().sum()
    missing_data_summary = missing_data[missing_data > 0].sort_values(ascending=False)
    if not missing_data_summary.empty:
        print(f"Warning: Missing data detected for {stock_name}")
        print(missing_data_summary)
    else:
        print(f"No missing data detected for {stock_name}.")
    return 0

def adjust_column_order(df_filtered):
    # Specifying the desired column order
    ordered_columns = ["date", "Open", "High", "Low", "Close", "Adj Close", 
                       "Volume", "tweets_count", "change", "candle_stick"]
    
    # Reordering the DataFrame columns
    df_filtered = df_filtered[ordered_columns]
    
    return df_filtered

