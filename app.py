import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

# Function to create sequences
def create_sequences(data, n_steps):
    X = []
    y = []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    return np.array(X), np.array(y)

st.title("📈 Stock Price Prediction using LSTM")
symbol = st.text_input("Enter Stock Ticker (e.g., GOOG, AAPL, MSFT):", value="GOOG")

if symbol:
    with st.spinner("Downloading and processing data..."):
        end = datetime.today()
        start = datetime(end.year - 3, end.month, end.day)
        data = yf.download(symbol, start=start, end=end)
        
        if data.empty:
            st.error("No data found for the given ticker!")
        else:
            prices = data['Close'].values.reshape(-1, 1)

            # Scale prices
            scaler = MinMaxScaler()
            scaled_prices = scaler.fit_transform(prices)

            # Create sequences
            n_steps = 50
            X, y = create_sequences(scaled_prices, n_steps)
            X = X.reshape((X.shape[0], X.shape[1], 1))

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # LSTM Model
            model = Sequential([
                LSTM(60, activation='tanh', input_shape=(n_steps, 1)),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

            # Evaluation
            loss = model.evaluate(X_test, y_test, verbose=0)
            st.success(f"Model Test MSE: {loss:.6f}")

            # Predictions
            y_pred = model.predict(X_test)
            y_pred_original = scaler.inverse_transform(y_pred)
            y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

            # Generate test set date range
            test_dates = data.index[-len(y_test_original):]

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(test_dates, y_test_original, label='Actual Price', color='blue')
            ax.plot(test_dates, y_pred_original, label='Predicted Price', color='orange', linestyle='dashed')
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for readability
            st.pyplot(fig)


            # Predict next 10 days
            st.subheader("🔮 Forecast for Next 10 Days")
            last_seq = scaled_prices[-n_steps:]
            future_preds = []
            input_seq = last_seq.copy()

            for _ in range(10):
                pred = model.predict(input_seq.reshape(1, n_steps, 1), verbose=0)
                future_preds.append(pred[0, 0])
                input_seq = np.append(input_seq[1:], pred, axis=0)

            future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

            next_dates = [data.index[-1] + timedelta(days=i+1) for i in range(10)]
            for date, pred in zip(next_dates, future_preds):
                st.write(f"{date.date()}: ${pred[0]:.2f}")
