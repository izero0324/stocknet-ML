import pandas as pd
from preprocess.csv_reader import read

def clean_stock_data(stock_name):
    # Read the CSV file is named after the stock and located in 'price/raw/' folder
    
    file_name = stock_name+'.csv'
    df = read(file_name)
    # Perform cleaning operations
    missing_data(df,stock_name)
    df_cleaned = df.dropna()
    
    # Correcting a specific data type
    df_cleaned['date'] = pd.to_datetime(df_cleaned['Date'])
    
    return df_cleaned

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
