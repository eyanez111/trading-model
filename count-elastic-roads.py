# Sample code to check count of documents and consistency in timestamps
from elasticsearch import Elasticsearch

es = Elasticsearch([{'host': '127.0.0.1', 'port': 9200, 'scheme': 'http'}])

# Count documents in each index
for index_name in ["coingecko_btc_data", "coingecko_btc_data_historical"]:
    doc_count = es.count(index=index_name)['count']
    print(f"Total records in {index_name}: {doc_count}")

# Check for gaps in timestamps by fetching data in ascending order
def fetch_data(index_name, size=1000):
    query = {
        "size": size,
        "sort": [{"timestamp": "asc"}]
    }
    response = es.search(index=index_name, body=query)
    timestamps = [hit['_source']['timestamp'] for hit in response['hits']['hits']]
    return timestamps

timestamps = fetch_data("coingecko_btc_data")
print("Sample timestamps:", timestamps[:5])  # Display first few timestamps
