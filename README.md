# S&P500 Stock Predictive Modeling

## Overview
- Obtaining historical stock data from the S&P500 index. Using Yahoo Finance, the project extracts, processes, and generates engineered features to aid in building predictive models for stock prices movements.

### Dataset Structure
- Each stock includes:
    - Open
    - High
    - Low
    - Close
    - Volume
 
### Features Engineered
1. Moving Averages
   - MA_10: 10-day moving average of closing price
   - MA_20: 20-day moving average of closing price
   - MA_50: 50-day moving average of closing price
   - MA_200: 200-day moving average of closing price
  
       - Use Cases:
           - Identify Short-term and Long-term trends

2.  Bollinger Bands
   - BB_Upper: Upper Bollinger Band = MA_20 + (2 * 20-day standard deviation)
   - BB_Lower: Lower Bollinger Band = MA_20 - (2 * 20-day standard deviation)

      - Use Cases:
          - Measure Market Volatility
       
3. Relative Strenght Index (RSI)
   - Measures momentum by comparing average gains and losses over a 14-day window
   -   RSI Ranges from 0 to 100:
       - RSI > 70: Overbought condition
       - RSI < 30: Oversold condition
        
5. Average True Range
   - Measures market volatility using high, low, and close prices over a 14-day window

      - Use Cases:
          - Set stop-loss levels based on volatility
          - Adjusts position sizes dynamically
      
7. Daily Returns
   - Percentage change in closing price between consecutive days
  
   - Formula: Close t - Close t-1 / Close t-1
    
9. Lagged Prices
    - Lag_1_Close: Closing price of the previous day
    - Lag_2_Close: Clsoing price two days ago
      
11. Volatility
    - Rolling 20-day standard deviation of daily returns
   
