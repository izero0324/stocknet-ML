import json
import pandas as pd
from preprocess.cleaner import clean_stock_data
from train import linear_fit

with open('preprocess/interested_stocks.json', 'r') as file:
    stock_data = json.load(file)

# looking at the "Technology" sector in the json file
technology_stocks = stock_data["Technology"]
stock_indi, total_stock_num = 1, len(technology_stocks)
# Loop through each stock in the sector
for stock_symbol in technology_stocks:
    print(f"==================={stock_symbol} Stock {stock_indi}/{total_stock_num} ===================")
    cleaned_data = clean_stock_data(stock_symbol)
    
    # print the describe of the cleaned data
    print(f"Cleaned data preview for {stock_symbol}:")
    print(cleaned_data.describe())
    
    #linear_fit(cleaned_data)

    stock_indi += 1
    
