import pandas as pd
import requests
from datetime import datetime, timedelta
import yfinance as yf


# Fetch the HTML content
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
response = requests.get(url)
response.raise_for_status()

sp500_tables = pd.read_html(response.text)
sp500 = sp500_tables[0]
sp500_symbols = sp500['Symbol'].tolist()

end_date = datetime.today()
start_date = end_date - timedelta(days=5 * 365)

# Fetching Historical Data
all_data = {}
for symbol in sp500_symbols:
    try:
        print(f"Fetching data for {symbol}...")
        stock_data = yf.download(symbol, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
        all_data[symbol] = stock_data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")

# Save the data
with open("sp500_data.pkl", "wb") as f:
    pd.to_pickle(all_data, f)

print("Data saved successfully!")
