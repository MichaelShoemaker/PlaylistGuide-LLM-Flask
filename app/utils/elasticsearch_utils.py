import os
from elasticsearch import Elasticsearch

elasticsearch_host = os.getenv("ELASTICSEARCH_HOST", "elasticsearch")
elasticsearch_port = os.getenv("ELASTICSEARCH_PORT", "9200")
elasticsearch_url = f"http://{elasticsearch_host}:{elasticsearch_port}"

es = Elasticsearch(elasticsearch_url)

def knn_query(question, model):
    return  {
        "field": "text_vector",
        "query_vector": model.encode(question),
        "k": 5,
        "num_candidates": 10000,
        "boost": 0.5,
    }

def keyword_query(question):
    return {
        "bool": {
            "must": {
                "multi_match": {
                    "query": f"{question}",
                    "fields": ["description^3", "text", "title"],
                    "type": "best_fields",
                    "boost": 0.5,
                }
            },
        }
    }

def multi_search(key_word, model):
    response = es.search(
        index='video-content',
        query=keyword_query(key_word),
        knn=knn_query(key_word, model),
        size=10
    )
    return [
        {
            'title': record['_source']['title'],
            'timecode_text': record['_source']['timecode_text'],
            'link': record['_source']['link'],
            'text': record['_source']['text'] 
        }
        for record in response["hits"]["hits"]
    ]



