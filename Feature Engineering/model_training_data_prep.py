import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

with open("Data/sp500_engineered_features.pkl", "rb") as f:
    data = pickle.load(f)

train_test_data = {}

# Loop through each stock in the dataset
for stock, df in data.items():
    try:
        df['Target'] = df['Close'].shift(-1)  # Predict next day's closing price
        df = df.dropna()
        
        X = df[['MA_10', 'MA_20', 'RSI', 'ATR', 'Volatility', 'Daily_Return', 'Lag_1_Close', 'Lag_2_Close']]
        y = df['Target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)
        
        # Normalize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        train_test_data[stock] = {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test
        }
        
        print(f"Processed {stock} successfully.")
    
    except Exception as e:
        print(f"Error processing {stock}: {e}")

with open("Data/sp500_train_test_data.pkl", "wb") as f:
    pickle.dump(train_test_data, f)

print("Feature processing for all stocks completed and saved.")
