import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'df_filtered' is your cleaned DataFrame containing stock prices
def plot_stock_prices(df_filtered):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_filtered, x='date', y='Close')
    plt.title('Stock Close Prices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

