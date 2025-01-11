import os
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Create directories if they don't exist
os.makedirs("Models", exist_ok=True)
os.makedirs("Results", exist_ok=True)  # Directory for saving results

# Load processed train-test data
with open("Data/sp500_train_test_data.pkl", "rb") as f:
    train_test_data = pickle.load(f)

# Dictionary to store trained models and performance
model_results = {}
mse_results = []

# Train a Linear Regression model for each stock
for stock, data in train_test_data.items():
    try:
        # Load train-test data
        X_train, X_test = data['X_train'], data['X_test']
        y_train, y_test = data['y_train'], data['y_test']
        
        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate model performance
        mse = mean_squared_error(y_test, y_pred)
        
        # Append MSE results for CSV
        mse_results.append({'Stock': stock, 'MSE': mse})
        
        # Save the model and performance
        model_results[stock] = {
            'model': model,
            'mse': mse,
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        # Save the trained model to a file
        model_filename = f"Models/{stock}_linear_regression.pkl"
        joblib.dump(model, model_filename)
        
        print(f"{stock}: Model trained successfully with MSE = {mse:.4f}")
    
    except Exception as e:
        print(f"Error training model for {stock}: {e}")

# Save the model results to a file
with open("Results/sp500_LP_model_results.pkl", "wb") as f:
    pickle.dump(model_results, f)

# Save the MSE results to a CSV file
mse_df = pd.DataFrame(mse_results)
mse_df.to_csv("Results/mse_results.csv", index=False)

print("Model training completed for all stocks. MSE results saved to 'Results/mse_results.csv'.")
