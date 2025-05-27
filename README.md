# 📈 Stock Price Prediction using LSTM

This project demonstrates how to use **Long Short-Term Memory (LSTM)** neural networks to forecast stock prices. Built with **Streamlit**, it allows users to enter any stock ticker (e.g., GOOG, AAPL, MSFT) and view predictions, error evaluation, and forecasts for the next 10 days.

---

## 📌 Features

- Download historical stock data using `yfinance`
- Scale data using `MinMaxScaler`
- Generate time-series sequences
- Train LSTM model to learn price patterns
- Visualize predicted vs. actual prices
- Forecast future 10 days of closing prices
- Built using an interactive Streamlit interface

---

## 📂 Project Structure
├── app.py # Streamlit application
├── requirements.txt # Python dependencies
└── README.md # Project documentation

---

## 📈 Example

Enter `GOOG` in the input box:

- Model Test MSE: `0.00042`
- Prediction vs Actual chart
- Forecast for the next 10 days with actual dates and price predictions

---

## 🛠️ How it Works

1. **Data Loading**  
   Downloads 3 years of historical stock data using `yfinance`.

2. **Data Preprocessing**  
   - Selects the 'Close' price
   - Applies MinMax scaling (range 0–1)
   - Creates sequences of length `n_steps=50`

3. **Model Training**  
   - Uses an LSTM layer with 60 units
   - Trains with `epochs=100` and `batch_size=32`
   - Loss function: Mean Squared Error (MSE)

4. **Prediction**  
   - Evaluates the model on the test set
   - Inverse transforms predictions for interpretability
   - Predicts the next 10 days based on last 50 values

5. **Visualization**  
   - Plots predicted vs. actual prices using `matplotlib`
   - Displays the 10-day future forecast with dates

---

## 💡 Tech Stack

- Python
- Streamlit
- YFinance
- NumPy & Pandas
- Matplotlib
- scikit-learn
- TensorFlow / Keras

---

## 📥 Installation

```bash
# 1. Clone the repository
git clone https://github.com/KushAmbre/stock-lstm-app.git
cd stock-lstm-app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py

