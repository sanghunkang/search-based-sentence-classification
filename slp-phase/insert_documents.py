import json

from datetime import datetime
from elasticsearch import Elasticsearch

es = Elasticsearch()


with open("sample.json", "r", encoding="utf8") as f: 
    intents = json.loads(f.read())["intents"]

    index = 1
    for intent in intents:
        for example in intent["examples"]:
            print(example)

            doc = {
                "text": example["text"],
                "timestamp": datetime.now(),
            }

            res = es.index(index="test-index", doc_type='tweet', id=index, body=doc)
            print(index, res['result'])

            res = es.get(index="test-index", doc_type='tweet', id=1)
            print(res['_source'])

            index += 1






