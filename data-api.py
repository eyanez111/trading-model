import requests
from datetime import datetime
from elasticsearch import Elasticsearch
import time

# Function to fetch real-time BTC price and volume in USD from CoinGecko
def fetch_real_time_data():
    print("Fetching live data...")
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
            print(f"Fetched live data - Time: {timestamp}, Price: {btc_price}, Total Volume: {btc_volume}")
            return btc_price, btc_volume, timestamp
    else:
        print(f"Failed to fetch live data: Status code {response.status_code}")
    return None, None, None

# Connect to Elasticsearch
es = Elasticsearch([{'host': '127.0.0.1', 'port': 9200, 'scheme': 'http'}])

# Verify Elasticsearch connection
try:
    if not es.ping():
        print("Elasticsearch connection failed. Exiting...")
        exit(1)
    else:
        print("Connected to Elasticsearch successfully.")
except Exception as e:
    print(f"Error connecting to Elasticsearch: {e}")
    exit(1)

# Start fetching and indexing real-time data
print("Starting live data fetch for coingecko_btc_data index...")
while True:
    try:
        start_time = time.time()

        # Fetch live data
        btc_price, btc_volume, timestamp = fetch_real_time_data()

        # Index the data if fetched successfully
        if btc_price is not None and btc_volume is not None:
            document = {
                'timestamp': timestamp,
                'symbol': 'BTC',
                'price_usd': btc_price,
                'total_volume_usd': btc_volume
            }
            try:
                res = es.index(index="coingecko_btc_data", body=document)
                print(f"Live data indexed successfully for coingecko_btc_data - Time: {timestamp}")
            except Exception as e:
                print(f"Failed to index live data: {e}")
        else:
            print("No valid data fetched. Skipping this iteration.")

        # Ensure consistent interval
        loop_duration = time.time() - start_time
        sleep_time = max(50 - loop_duration, 0)  # Fetch every 50 seconds
        print(f"Loop completed. Sleeping for {sleep_time:.2f} seconds...")
        time.sleep(sleep_time)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        time.sleep(10)
