import requests
from datetime import datetime, timedelta
from elasticsearch import Elasticsearch
import time

# Function to get historical BTC price and volume in USD from CoinGecko using specific date
def fetch_historical_data(days_ago):
    target_date = datetime.now() - timedelta(days=days_ago)
    formatted_date = target_date.strftime('%d-%m-%Y')  # Format for CoinGecko

    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/history"
    params = {'date': formatted_date, 'localization': 'false'}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        market_data = data.get('market_data', {})
        
        btc_price = market_data.get('current_price', {}).get('usd')
        btc_volume = market_data.get('total_volume', {}).get('usd')
        
        if btc_price is not None and btc_volume is not None:
            print(f"Fetched historical data - Date: {target_date.isoformat()}, Price: {btc_price}, Total Volume: {btc_volume}")
            return btc_price, btc_volume, target_date.isoformat()
        else:
            print(f"No volume or price data available for {target_date.isoformat()}")
    else:
        print(f"Failed to fetch data for {target_date.isoformat()}: Status code {response.status_code}")

    return None, None, target_date.isoformat()

# Function to fetch an extended range of historical BTC data
def fetch_extended_historical_data(start_date, end_date):
    days_diff = (end_date - start_date).days
    for days_ago in range(1, days_diff + 1):
        btc_price, btc_volume, target_date = fetch_historical_data(days_ago)
        
        if btc_price is not None and btc_volume is not None:
            document = {
                'timestamp': target_date,
                'symbol': 'BTC',
                'price_usd': btc_price,
                'total_volume_usd': btc_volume
            }
            es.index(index="coingecko_btc_data_historical", body=document)
            print(f"Historical data indexed successfully for {target_date}")
        
        # Shortened sleep for testing, adjust as needed based on API rate limits
        time.sleep(15)

# Function to get real-time BTC price and volume in USD from CoinGecko
def fetch_real_time_data():
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": "bitcoin",
        "vs_currencies": "usd",
        "include_24hr_vol": "true"
    }
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        if 'bitcoin' in data:
            btc_price = data['bitcoin']['usd']
            btc_volume = data['bitcoin']['usd_24h_vol']
            timestamp = datetime.utcnow().isoformat()
            print(f"Fetched real-time data - Time: {timestamp}, Price: {btc_price}, Total Volume: {btc_volume}")
            return btc_price, btc_volume, timestamp
    else:
        print(f"Failed to fetch real-time data: Status code {response.status_code}")
    return None, None, None

# Connect to Elasticsearch
es = Elasticsearch([{'host': '127.0.0.1', 'port': 9200, 'scheme': 'http'}])

# Step 1: Fetch extended historical data for a specified range
start_date = datetime.now() - timedelta(days=365)  # Fetch data starting from one year ago
end_date = datetime.now()
fetch_extended_historical_data(start_date, end_date)

# Step 2: Fetch recent historical data for the past 30 days (optional)
days_to_fetch = 30
for day in range(1, days_to_fetch + 1):
    btc_price, btc_volume, target_date = fetch_historical_data(day)

    if btc_price is not None and btc_volume is not None:
        document = {
            'timestamp': target_date,
            'symbol': 'BTC',
            'price_usd': btc_price,
            'total_volume_usd': btc_volume
        }
        res = es.index(index="coingecko_btc_data_historical", body=document)
        print(f"Historical data indexed successfully for {target_date}: {res['result']}")

    time.sleep(15)  # Reduced for testing; adjust if rate limits are hit

# Step 3: Fetch real-time data every 100 seconds
while True:
    btc_price, btc_volume, timestamp = fetch_real_time_data()

    if btc_price is not None and btc_volume is not None:
        document = {
            'timestamp': timestamp,
            'symbol': 'BTC',
            'price_usd': btc_price,
            'total_volume_usd': btc_volume
        }
        res = es.index(index="coingecko_btc_data", body=document)
        print(f"Real-time data indexed successfully for {timestamp}: {res['result']}")

    time.sleep(100)  # Adjust to 100 seconds
