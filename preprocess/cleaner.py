import pandas as pd
from preprocess.csv_reader import read

def clean_stock_data(stock_name):
    # Read the CSV file is named after the stock and located in 'price/raw/' folder
    
    file_name = stock_name+'.csv'
    df = read(file_name)
    # Perform cleaning operations
    df_cleaned = df.dropna()
    
    # Correcting a specific data type
    df_cleaned['date'] = pd.to_datetime(df_cleaned['Date'])
    
    return df_cleaned
