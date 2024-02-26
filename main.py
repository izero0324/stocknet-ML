from preprocess.cleaner import clean_stock_data

# Example stock name 'AAPL'
stock_name = 'AAPL'
cleaned_data = clean_stock_data(stock_name)

# Take a look at the cleaned_data
print(cleaned_data.head())