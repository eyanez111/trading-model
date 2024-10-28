import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch

# Connect to Elasticsearch
es = Elasticsearch([{'host': '88.99.147.210', 'port': 9200, 'scheme': 'http'}])

# Parameters
index_name = "coingecko_btc_data_historical"
output_csv = "/home/francisco/trading-model/cleaned_data.csv"

# Step 1: Fetch Data from Elasticsearch
def fetch_data(index_name):
    query = {
        "size": 10000,
        "_source": ["timestamp", "price_usd", "total_volume_usd"],
        "query": {
            "range": {
                "timestamp": {
                    "gte": "now-30d/d",   # Fetch data from the last 30 days
                    "lt": "now/d"
                }
            }
        },
        "sort": [{"timestamp": {"order": "asc"}}]
    }
    
    response = es.search(index=index_name, body=query)
    data = [
        (hit["_source"]["timestamp"], hit["_source"].get("price_usd"), hit["_source"].get("total_volume_usd"))
        for hit in response["hits"]["hits"]
    ]
    
    return pd.DataFrame(data, columns=["timestamp", "price_usd", "total_volume_usd"])

# Step 2: Clean the Data
def clean_data(df):
    # Handle missing values for price and volume
    df["price_usd"] = df["price_usd"].fillna(df["price_usd"].rolling(5, min_periods=1).median())
    df["total_volume_usd"] = df["total_volume_usd"].fillna(df["total_volume_usd"].rolling(5, min_periods=1).median())
    
    # Remove outliers using z-score and replace them with rolling median values
    z_scores_volume = (df["total_volume_usd"] - df["total_volume_usd"].rolling(5, min_periods=1).mean()) / df["total_volume_usd"].rolling(5, min_periods=1).std()
    df["total_volume_usd"] = np.where(np.abs(z_scores_volume) < 3, df["total_volume_usd"], df["total_volume_usd"].rolling(5, min_periods=1).median())
    
    z_scores_price = (df["price_usd"] - df["price_usd"].rolling(5, min_periods=1).mean()) / df["price_usd"].rolling(5, min_periods=1).std()
    df["price_usd"] = np.where(np.abs(z_scores_price) < 3, df["price_usd"], df["price_usd"].rolling(5, min_periods=1).median())
    
    # Smooth volume data by applying exponential moving average (EMA)
    df["total_volume_usd"] = df["total_volume_usd"].ewm(span=10, adjust=False).mean()
    
    return df

# Step 3: Fetch, Clean, and Export Data
data_df = fetch_data(index_name)

if not data_df.empty:
    cleaned_df = clean_data(data_df)
    cleaned_df.to_csv(output_csv, index=False)
    print(f"Cleaned data saved to {output_csv}")
else:
    print("No data found for the specified range.")
