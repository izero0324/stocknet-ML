import matplotlib.pyplot as plt
import seaborn as sns

# 'df_filtered' is the cleaned DataFrame containing stock prices
def plot_stock_prices(df_filtered):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_filtered, x='date', y='Close')
    plt.title('Stock Close Prices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_predicted_close(y_test, y_pred):
    y_test_sorted = y_test.sort_index()
    index_for_plotting = y_test_sorted.index
    print(y_test_sorted.index)

    # Plot actual prices
    plt.figure(figsize=(14, 7))
    plt.plot(index_for_plotting, y_test_sorted, color='blue', label='Actual Prices')

    # Since y_pred is a numpy array without a date index, align it with y_test_sorted's index for plotting
    plt.plot(index_for_plotting, y_pred, color='red', linestyle='--', label='Predicted Prices')

    plt.title('Comparison of Actual and Predicted Stock Price Change %')
    plt.xlabel('Date')
    plt.ylabel('Price change %')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def LSTM_plot(prediction_dates,y_test_rescaled,predictions):
    plt.figure(figsize=(16,8))
    plt.title('Model Predictions vs Actual Prices')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Open Price USD ($)', fontsize=18)

    # Actual prices line
    plt.plot(prediction_dates, y_test_rescaled, label='Actual Price')

    # Predicted prices line
    plt.plot(prediction_dates, predictions, color='red', label='Predicted Price')

    plt.legend(loc='upper left')
    plt.xticks(rotation=45)  # Rotate dates for better readability

    plt.show()