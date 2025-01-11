import pandas as pd
import numpy as np
import pickle

with open("sp500_engineered_features.pkl", "rb") as f:
    engineered_data = pickle.load(f)

sample_stock = "AAPL"
if sample_stock in engineered_data:
    print(f"Engineered data for {sample_stock}:")
    print(engineered_data[sample_stock].head())
else:
    print(f"No engineered data for {sample_stock}")
