import pandas as pd
import sys

def clean_stock_data(stock_name):
    # Read the CSV file is named after the stock and located in 'price/raw/' folder
    file_path = f'./price/raw/{stock_name}.csv'
    
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Perform cleaning operations
    df_cleaned = df.dropna()
    
    # Correcting a specific data type
    df_cleaned['date'] = pd.to_datetime(df_cleaned['date'])
    
    return df_cleaned

if __name__ == "__main__":
    # Argument passed via command line
    stock_name = sys.argv[1]
    cleaned_data = clean_stock_data(stock_name)
    print(cleaned_data.head())