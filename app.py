import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import yfinance as yf

# Load the saved model
model = load_model("Latest_stock_price_model.keras")

# Define scaler (match the one used in training)
scaler = MinMaxScaler(feature_range=(0, 1))

# Function to generate future predictions
def predict_future(data, steps, model, scaler):
    data_scaled = scaler.fit_transform(data)
    future_predictions = []
    input_data = data_scaled[-100:]  # Last 100 steps as input

    for _ in range(steps):
        input_data = input_data.reshape(1, 100, 1)
        pred = model.predict(input_data)[0]
        future_predictions.append(pred)
        input_data = np.append(input_data[:, 1:, :], [[pred]], axis=1)

    future_predictions = scaler.inverse_transform(future_predictions)
    return future_predictions.flatten()

# Add background image using custom CSS (image URL)
background_image_url = "https://img.freepik.com/premium-vector/binary-code-background-with-software-programming_185386-684.jpg?w=740"  # Replace with your image URL
st.markdown(
    f"""
    <style>
        .stApp {{
            background-image: url("{background_image_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-color: white;
        }}
         .stApp::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.4);  /* Semi-transparent dark overlay */
            z-index: 0;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app
st.title("Stock Price Prediction App")
st.write("Predict future stock prices using an LSTM model.")

# Input section for stock ticker and number of days to predict (1-3 days range)
ticker = st.text_input("Enter Stock Ticker (e.g., 'AAPL' for Apple)", value="AAPL")
prediction_steps = st.slider("Select number of future days to predict", min_value=1, max_value=3, step=1)

if ticker:
    # Fetch historical stock data from Yahoo Finance
    historical_data = yf.download(ticker, period="5y", progress=False)

    if historical_data.empty:
        st.error("No data found for this ticker. Please check the ticker symbol and try again.")
    else:
        st.write(f"Historical Data for {ticker}", historical_data)

        # Choose 'Adj Close' or 'Close' based on availability
        if "Adj Close" in historical_data.columns:
            column = "Adj Close"
        else:
            column = "Close"

        data_to_predict = historical_data[[column]].values
        future_predictions = predict_future(data_to_predict, prediction_steps, model, scaler)

        # Plot results
        future_dates = pd.date_range(start=historical_data.index[-1], periods=prediction_steps + 1, freq="D")[1:]
        predicted_df = pd.DataFrame({column: future_predictions}, index=future_dates)

        st.write("Predicted Prices for Future Dates", predicted_df)

        # Plot historical and predicted data
        plt.figure(figsize=(12, 6))
        plt.plot(historical_data.index, historical_data[column], label="Historical Data")
        # plt.plot(predicted_df.index, predicted_df[column], label="Predicted Data", linestyle="--")
        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.legend()
        st.pyplot(plt)
