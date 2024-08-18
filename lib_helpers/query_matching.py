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
from lib_helpers.embedding_and_retrieval import update_retrieved_status, answer_retriever, redis_client, embeddings
from typing import Dict, List, Any


load_dotenv()

# Connect to Redis
REDIS_CLIENT = redis_client

# Initialize embeddings
EMBEDDINGS = embeddings

def get_query_hash(query: str) -> str:
    """Generate a hash for the query."""
    return hashlib.sha256(query.encode()).hexdigest()

# Record in Redis Both Kind of Keys (hashed for exact match and vector for semantic match)
def cache_response(query: str, response: List[Dict[str, Any]], ttl: int = 3600) -> dict:
    """Cache the response in Redis with both hash and vector representation."""
    query_hash = get_query_hash(query)
    query_vector = EMBEDDINGS.embed_query(query)

    # Store with query hash as key
    REDIS_CLIENT.setex(query_hash, timedelta(seconds=ttl), json.dumps(response))

    # Store with query vector as key but change it to tuple and hash it as vectors are list therefore mutable which makes it not hashable as list can change
    query_vector_tuple = tuple(query_vector)
    vector_key = f"vector:{json.dumps(query_vector_tuple)}"
    REDIS_CLIENT.setex(vector_key, timedelta(seconds=ttl), json.dumps(response))
    return {"query_hash": query_hash, "query_vector": query_vector}

# exact match search fucntions
def fetch_cached_response_by_hash(query: str) -> dict|List[Dict[str,Any]]|None:
    """Fetch the cached response from Redis using query hash if it exists."""
    response = []
    
    # search using hash of query
    query_hash = get_query_hash(query)
    data_from_query_hash = REDIS_CLIENT.get(query_hash)
    if data_from_query_hash:
      response.append(json.loads(data_from_query_hash) if data_from_query_hash else None)
    # search using hash of vector of query
    query_vector = EMBEDDINGS.embed_query(query)
    hash_query_vector_tuple = hash(tuple(query_vector))
    data_from_vector_hash = REDIS_CLIENT.get(hash_query_vector_tuple)
    if data_from_vector_hash:
      response.append(json.loads(data_from_vector_hash) if data_from_vector_hash else None)
    return response

def fetch_all_cached_embeddings() -> dict:
    """Fetch all cached embeddings from Redis."""
    keys = REDIS_CLIENT.keys('vector:*')
    all_embeddings = {}
    for key in keys:
        embedding_str = key.decode().split("vector:")[1]
        embedding = json.loads(embedding_str)
        data = json.loads(REDIS_CLIENT.get(key))
        all_embeddings[embedding_str] = {
            "embedding": embedding,
            "response": data
        }
    return all_embeddings


# Semantic search function
def perform_semantic_search_in_redis(query_embedding: list, threshold: float = 0.7) -> List[Dict[str, Any]]|None:
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

# Vector search function with postgresql table update and caching of new retrieved content
def perform_vector_search(query: str, score: float, top_n: int) -> List[Dict[str, Any]]:

  doc_ids = []
   
  response = answer_retriever(query, score, top_n)
  print("JSON RESPONSE: ", json.dumps(response, indent=4))
  for elem in response:
    doc_ids.append(elem["UUID"])

  # as it as been retrieved we use the function `update_retrieved_status` from the library `embedding_and_retrieval` to update the db column `retrieved` to True
  # this could be done somewhere else as well inside 'answer_retriever' with all retireved data being labelled as `retireved` in the db and cached
  for doc_id in doc_ids:
    update_retrieved_status(doc_id)

  # then cache the query with the result content obtained in the hash way and the vectorized way
  cache_response(query, response, ttl=86400) # 86400=1d, 3600=1h
  return response

#### This the meat function which does the exact match, then, semantic match, then the expensive vector database retireval and cache it if it finds anything for the response
def handle_query_by_calling_cache_then_vectordb_if_fail(query: str, score: float, top_n: int) -> Dict[str, Any]:
    """Handle the user query by deciding between cache, semantic search, and vector search."""
    
    # vectorize the query for semantic search
    query_vectorized = EMBEDDINGS.embed_query(query)
    #print("Initial Query: ", query, "Query Vectorized: ", query_vectorized)
    
    # Try to fetch from cache (Exact Query Match)
    try:
      # returns List[Dict]
      cached_response = fetch_cached_response_by_hash(query)
    except Exception as e:
      print({"An error occured while trying to fetch cached response by hash": e})
    if cached_response:
        return {"exact_match_search_response_from_cache": cached_response}

    # Perform semantic search if exact query is not found
    try:
      semantic_response = perform_semantic_search_in_redis(query_vectorized, score)
    except Exception as e:
      print({"An error occured while trying to perform semantic search in redis": e})
    if semantic_response:
        return {"semantic_search_response_from_cache": semantic_response}

    # Perform vector search with score if semantic search is not relevant
    try:
      vector_response = perform_vector_search(query, score, top_n)
    except Exception as e:
      print({"An error occured while trying to perform vectordb search query": e})
    if vector_response:
        # Cache the new response with TTL
        cache_response(query, vector_response, ttl=3600)
        return {"vector_search_response_after_cache_failed_to_find", vector_response}

    # If no relevant result found, return a default response
    return {"message": "No relevant information found."}



