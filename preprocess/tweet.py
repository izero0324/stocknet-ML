import pandas as pd
import json
import os

def count_tweets_for_date(stock_name, date):
    file_path = f'./tweet/preprocessed/{stock_name}/{date}'

    # Check if the file exists
    if not os.path.exists(file_path):
        return 0  # Return 0 count if file doesn't exist
    
    tweet_count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            tweet_count += 1
    
    return tweet_count
