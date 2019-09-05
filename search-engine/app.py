from datetime import datetime
from elasticsearch import Elasticsearch

es = Elasticsearch()


# res = es.index(index="test-index", doc_type='tweet', id=2, body=doc)
# print(res['result'])

# res = es.get(index="test-index", doc_type='tweet', id=1)
# print(res['_source'])

es.indices.refresh(index="test-index")
query = {
    "match": {"text": "가 뭐"}
}


res = es.search(index="test-index", body={"query": query})
print("Got %d Hits:" % res['hits']['total']['value'])
for hit in res['hits']['hits']:
    print(hit["_source"])
    print(hit)
    # print("%(timestamp)s %(author)s: %(text)s" % hit["_source"])


# Search
# Find neighbourhood
    # input: expectable patterns which annotators expect
    # collect N~100 for each class

# Label
# For each class
    # Select subsets
    # 10
    #   4 of high ranks 
    #   3 of middle ranks
    #   3 of low ranks

    # Prompt IN or OUT (or NOT SURE)

# Propagate
# For each class
    # if either of ratio > 0.6(arbitrarily chosen value)  
    #   -> propagte label for the entire neighbourhood
    # else
    #   -> only use the existing labels



# Automated labelling
    # for each candidate
    # label 1, l2, l3 ... ln = Lambda:example -> [1, 0, 1, -1 .. 0]