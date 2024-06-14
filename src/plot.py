import matplotlib.pyplot as plt


def plot_data(data, predictions_df, factor):
    # Plot data
    plt.figure(figsize=(10, 6))

    # Plot actual data
    plt.plot(data.index[-factor:], data["Close"].values[-factor:], linestyle="-", marker="o", color="blue",
             label="Actual Data")

    # Plot predicted data
    prediction_dates = predictions_df.index
    predicted_prices = predictions_df["Close"].values
    plt.plot(prediction_dates, predicted_prices, linestyle="-", marker="o", color="red", label="Predicted Data")

    plt.title("Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    plt.show()
