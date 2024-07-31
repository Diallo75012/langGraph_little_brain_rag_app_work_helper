"""
# query_matching.py

# Here we handle the user query going to the cache first before calling any `embedding_and_retrieval.py` library functions.
# It will check exact match, then semantic mathc, then if not foudn call the `embedding_and_retrieval.py` in order to do a vector database search

"""
#### STORE DOUBLE ENTRY INREDIS TO BE ABLE TO PERFORM EXACT SEARCH AND SEMANTIC SEARCH ON CACHE
"""
Here we having the cache with TTL so that it will expire but it will permit us to do exact search of user query to pull the answer corresponding if exists in cache, otherwise, we will do a semantic search using vector of query, so we have two different keys but the values of those cached queres are the same. This is just to permit us to do those two kind of search and if they fail, then. to do the expensive vector datavase search.
""" 
import redis
import hashlib
import json
from datetime import timedelta
from dotenv import load_dotenv
import numpy as np
from langchain_community.embeddings import OllamaEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from embedding_and_retrieval import *


load_dotenv()

# Connect to Redis
redis_client = redis.StrictRedis(host=os.getenv("REDIS_HOST"), port=os.getenv("REDIS_PORT"), db=0)

# Initialize embeddings
embeddings = OllamaEmbeddings(temperature=0)

def get_query_hash(query: str) -> str:
    """Generate a hash for the query."""
    return hashlib.sha256(query.encode()).hexdigest()

# Record in Redis Both Kind of Keys (hashed for exact match and vector for semantic match)
def cache_response(query: str, response: dict, ttl: int = 3600):
    """Cache the response in Redis with both hash and vector representation."""
    query_hash = get_query_hash(query)
    query_vector = embeddings.embed_text(query)

    # Store with query hash as key
    redis_client.setex(query_hash, timedelta(seconds=ttl), json.dumps(response))

    # Store with query vector as key (serialize vector)
    vector_key = f"vector:{json.dumps(query_vector)}"
    redis_client.setex(vector_key, timedelta(seconds=ttl), json.dumps(response))

# exact match search fucntions
def fetch_cached_response_by_hash(query: str) -> dict:
    """Fetch the cached response from Redis using query hash if it exists."""
    query_hash = get_query_hash(query)
    data = redis_client.get(query_hash)
    return json.loads(data) if data else None

def fetch_all_cached_embeddings() -> dict:
    """Fetch all cached embeddings from Redis."""
    keys = redis_client.keys('vector:*')
    all_embeddings = {}
    for key in keys:
        embedding_str = key.decode().split("vector:")[1]
        embedding = json.loads(embedding_str)
        data = json.loads(redis_client.get(key))
        all_embeddings[embedding_str] = {
            "embedding": embedding,
            "response": data
        }
    return all_embeddings


# Semantic search function
def perform_semantic_search_in_redis(query_embedding: list, threshold: float = 0.8) -> str:
    """Perform semantic search on the cached query embeddings in Redis."""
    all_embeddings = fetch_all_cached_embeddings()
    if not all_embeddings:
        return None
    
    query_embedding_np = np.array(query_embedding).reshape(1, -1)
    max_similarity = 0
    best_match_response = None

    for data in all_embeddings.values():
        embedding_np = np.array(data["embedding"]).reshape(1, -1)
        similarity = cosine_similarity(query_embedding_np, embedding_np)[0][0]
        if similarity > max_similarity:
            max_similarity = similarity
            best_match_response = data["response"]
    
    if max_similarity >= threshold:
        return best_match_response
    return None

#### This the meat function which does the exact match, then, semantic match, then the expensive vector database retireval and cache it if it finds anything for the response
def handle_query(query: str) -> dict:
    """Handle the user query by deciding between cache, semantic search, and vector search."""
    # Embed the query
    query_embedding = embeddings.embed_text(query)

    # Try to fetch from cache (Exact Query Match)
    cached_response = fetch_cached_response_by_hash(query)
    if cached_response:
        return cached_response

    # Perform semantic search if exact query is not found
    semantic_response = perform_semantic_search_in_redis(query_embedding)
    if semantic_response:
        return semantic_response

    # Perform vector search if semantic search is not relevant
    """
      Here we can maybe call our library to perform a vector database search (AKA retrieval)
    """
    vector_response = perform_vector_search(query)
    if vector_response:
        # Cache the new response with TTL
        cache_response(query, vector_response, ttl=3600)
        return vector_response

    # If no relevant result found, return a default response
    return {"message": "No relevant information found."}

#### ALL THE FOLLOWING ARE EXAMPLES BUT LIBRARIES `query_matching` and `embedding_and_retrieval` will need to be imported to the main `app.py` and not here
# Example usage how to query the Redis cache
"""
query_redis_cache_then_vecotrdb_if_no_cache(query: str) -> any: #check types returned or format all returns to be dict
  response = handle_query(query)
  print(json.dumps(response, indent=4))
  return response
"""

## example usage to query using the imported library `embedding_and_retrieval`
# example of how to use it and query for retrieval vector in collection
"""
def perform_vector_search(query: str) -> list:

  response = answer_retriever(query, 0.7)
  print(json.dumps(response, indent=4))
  content_list = []
  score_list = [0]
  for elem in response:
    if elem["score"] > score_list[0]:
      score_list.append(elem["score"])
      doc_id = elem["row_data"]["id"]
      doc_title = elem["row_data"]["title"]
      doc_name = elem["row_data"]["doc_name"]
      content = elem["content"]
      content_list.append(content)
      # ...etc...
      # as it as been retrieved we use the function `update_retrieved_status` from the library `embedding_and_retrieval` to update the db column `retirved` to True
      # this could be done somewhere else as well inside 'answer_retriever' with all retireved data being labelled as `retireved` in the db and cached
      update_retrieved_status(doc_id)
  # then cache the query with the result content obtained in the hash a=way and the vectorized way
  content_dict = {"responses": []}
  for elem in content_list:
    content_dict["response"].append(elem)
  cache_response(query, content_dict, ttl=3600)
  return content_list
"""

# Example how to perform embeding of docs
USAGE:
# Assuming we have the document ID and content
document_id = uuid.uuid4()
document_content = "Example content of the document chunk."
cache_document(document_id, document_content)
update_retrieved_status(document_id)

# clear chache periodically
import schedule
import time

# Schedule the cache clearing function
# maybe here use subprocess to run this kind of app cron job, so TTL is for 1h cached queries and cache is completely cleared once a day with this job here
schedule.every().day.at("00:00").do(clear_cache)
schedule.run_pending()

### Store document quality chunks in the postgresql DB
"""
Here we need to use the `pdf_parser` library or the `webpage_parser` library. those are going to to load the doc/page and use pandas to get best content store in the db after data cleaning and embedd those documents under the same collection name
from pdf_parser import *
from webpage_parser import *
# workflow:
 - document is loaded
 - docuemtn is saved to db after being cleaned (just quality data from it)
 - now embedding needs to be done on the database data using the `embedding_and_retireval` library functions
 - after that we are ready to use here the  library `query_matching` to get user/llm query and perform cahce search first and then vector search if not found in cache
 - need to implement a tool internet search in the agent logic to perform internet search only if this fails
 - better use lots of mini tools than one huge and let the llm check. We will just prompting the agent to tell it what is the workflow. So revise functions here to be decomposed in mini tools
 - then build all those mini tools

"""

