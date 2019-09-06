from datetime import datetime
from elasticsearch import Elasticsearch


import random

SIZE_OF_NEIGHBORHOOD = 100

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
neighborhood = es.search(index="test-index", body={"query": query})["hits"]["hits"][:SIZE_OF_NEIGHBORHOOD]
for hit in neighborhood:
    # print(hit["_source"])
    print(hit)
    # input: expectable patterns which annotators expect
    # collect N~100 for each class

# Label
actual_n = len(neighborhood)
# For each class
subset = []
# Select subsets summing up to 10
#   4 of high ranks 
neighbourhood_of_high_rank = neighborhood[:int(actual_n/3*1)]
subset.append(**random.choices(neighbourhood_of_high_rank, k=4))

#   3 of middle ranks
neighborhood_middle_ranks = neighborhood[int(actual_n/3*1):int(actual_n/3*2)]
subset.append(**random.choices(neighbourhood_of_high_rank, k=3))

#   3 of low ranks
neighborhood_low_ranks = neighborhood[int(actual_n/3*2):]
subset.append(**random.choices(neighbourhood_of_high_rank, k=3))
    
# Prompt IN or OUT (or NOT SURE)
print("____")
print(actual_n)
for elem in subset:
    print(elem)

# Propagate
# For each class
    # if either of ratio > 0.6(arbitrarily chosen value)  
    #   -> propagte label for the entire neighbourhood
    # else
    #   -> only use the existing labels



# Automated labelling
    # for each candidate
    # label 1, l2, l3 ... ln = Lambda:example -> [1, 0, 1, -1 .. 0]