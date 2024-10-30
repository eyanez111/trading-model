from elasticsearch import Elasticsearch

# Connect to Elasticsearch
es = Elasticsearch([{'host': '88.99.147.210', 'port': 9200, 'scheme': 'http'}])

# Function to retrieve and display recent data from a specified index
def check_volume_data(index_name, num_records=5):
    query = {
        "size": num_records,
        "sort": [
            {"timestamp": {"order": "desc"}}
        ]
    }
    response = es.search(index=index_name, body=query)
    
    print(f"Recent entries from index '{index_name}':")
    for hit in response['hits']['hits']:
        source = hit['_source']
        print(f"Timestamp: {source['timestamp']}, Price: {source['price_usd']}, Volume: {source.get('volume_usd', 'Not found')}")

# Check recent data in both historical and real-time indices
check_volume_data("coingecko_btc_data_historical")
check_volume_data("coingecko_btc_data")

