import pandas as pd
import numpy as np
import pickle
# Load processed data
with open("Data/sp500_train_test_data.pkl", "rb") as f:
    train_test_data = pickle.load(f)

# Example: Access data for AAPL
aapl_data = train_test_data['AAPL']
X_train, X_test, y_train, y_test = (
    aapl_data['X_train'],
    aapl_data['X_test'],
    aapl_data['y_train'],
    aapl_data['y_test'],
)

print("AAPL training data shape:", X_train.shape)
