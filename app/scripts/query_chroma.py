import chromadb

client = chromadb.HttpClient(host="localhost", port=9001)

collection = client.get_collection("novels_collection")

results = collection.query(
    query_texts=["Elizabeth was the real cause of all the mischief?"],
    n_results=3
)

print(results)