import pandas as pd
import numpy as np
import pickle

# Load the raw data
with open("sp500_data.pkl", "rb") as f:
    sp500_data = pickle.load(f)

# Function to flatten MultiIndex DataFrame
def flatten_multiindex(df):
    # Rename MultiIndex columns to single level
    df.columns = [col[0] for col in df.columns]  # Use only the first level (Price)
    return df

# Feature engineering function
def engineer_features(df):
    df = df.dropna()

    # Feature: Daily Returns
    df['Daily_Return'] = df['Close'].pct_change()

    # Feature: Moving Averages
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()

    # Feature: Bollinger Bands
    df['BB_Upper'] = df['MA_20'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_Lower'] = df['MA_20'] - 2 * df['Close'].rolling(window=20).std()

    # Feature: Relative Strength Index (RSI)
    delta = df['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Feature: Average True Range (ATR)
    df['TR'] = np.maximum.reduce([
        df['High'] - df['Low'],
        abs(df['High'] - df['Close'].shift(1)),
        abs(df['Low'] - df['Close'].shift(1)),
    ])
    df['ATR'] = df['TR'].rolling(window=14).mean()

    # Feature: Lagged Prices
    df['Lag_1_Close'] = df['Close'].shift(1)
    df['Lag_2_Close'] = df['Close'].shift(2)

    # Feature: Volatility (Rolling Std of Returns)
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std()

    # Drop NaN rows created by rolling calculations
    df = df.dropna()

    return df

# Apply feature engineering to all stocks
engineered_data = {}
for symbol, data in sp500_data.items():
    try:
        print(f"Processing {symbol}...")
        # Flatten the MultiIndex structure
        flat_data = flatten_multiindex(data)
        # Perform feature engineering
        engineered_data[symbol] = engineer_features(flat_data)
    except Exception as e:
        print(f"Error processing {symbol}: {e}")

# Save the engineered features
with open("sp500_engineered_features.pkl", "wb") as f:
    pickle.dump(engineered_data, f)

print("Feature engineering completed and saved to sp500_engineered_features.pkl")
