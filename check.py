import pandas as pd
import numpy as np
import pickle

# Load the engineered data
with open("sp500_engineered_features.pkl", "rb") as f:
    engineered_data = pickle.load(f)

# Example: Access features for a specific stock
sample_stock = "AAPL"  # Replace with any symbol to check
if sample_stock in engineered_data:
    print(f"Engineered data for {sample_stock}:")
    print(engineered_data[sample_stock].head())
else:
    print(f"No engineered data for {sample_stock}")
