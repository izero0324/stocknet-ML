import pandas as pd

def read(file_name):

    file_path = f'./price/raw/{file_name}'
    
    # Load the dataset
    df = pd.read_csv(file_path)

    return df