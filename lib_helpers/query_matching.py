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
from typing import Dict, List, Any


load_dotenv()

# Connect to Redis
redis_client = redis.StrictRedis(host=os.getenv("REDIS_HOST"), port=os.getenv("REDIS_PORT"), db=0)

# Initialize embeddings
embeddings = OllamaEmbeddings(temperature=0)

def get_query_hash(query: str) -> str:
    """Generate a hash for the query."""
    return hashlib.sha256(query.encode()).hexdigest()

# Record in Redis Both Kind of Keys (hashed for exact match and vector for semantic match)
def cache_response(query: str, response: List[Dict[str, Any]], ttl: int = 3600) -> dict:
    """Cache the response in Redis with both hash and vector representation."""
    query_hash = get_query_hash(query)
    query_vector = embeddings.embed_text(query)

    # Store with query hash as key
    redis_client.setex(query_hash, timedelta(seconds=ttl), json.dumps(response))

    # Store with query vector as key (serialize vector)
    vector_key = f"vector:{json.dumps(query_vector)}"
    redis_client.setex(vector_key, timedelta(seconds=ttl), json.dumps(response))
    return {"query_hash": query_hash, "query_vector": query_vector}

# exact match search fucntions
def fetch_cached_response_by_hash(query: str) -> dict|List[Dict[str,Any]]|None:
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
def perform_vector_search(query: str, score: float) -> List[Dict[str, Any]]:

  doc_ids = []
   
  response = answer_retriever(query, 0.7, 4)
  print(json.dumps(response, indent=4))
  for elem in response:
    doc_ids.append(elem["UUID"])

  # as it as been retrieved we use the function `update_retrieved_status` from the library `embedding_and_retrieval` to update the db column `retrieved` to True
  # this could be done somewhere else as well inside 'answer_retriever' with all retireved data being labelled as `retireved` in the db and cached
  for doc_id in doc_ids:
    update_retrieved_status(doc_id)

  # then cache the query with the result content obtained in the hash way and the vectorized way
  cache_response(query, response, ttl=3600) # 86400=1d, 3600=1h
  return response

#### This the meat function which does the exact match, then, semantic match, then the expensive vector database retireval and cache it if it finds anything for the response
def handle_query_by_calling_cache_then_vectordb_if_fail(query: str, score: float) -> dict:
    """Handle the user query by deciding between cache, semantic search, and vector search."""
    # vectorize the query for semantic search
    query_vectorized = embeddings.embed_text(query)

    # Try to fetch from cache (Exact Query Match)
    cached_response = fetch_cached_response_by_hash(query)
    if cached_response:
        return {"exact_match_search_response_from_cache": cached_response}

    # Perform semantic search if exact query is not found
    semantic_response = perform_semantic_search_in_redis(query_vectorized)
    if semantic_response:
        return {"semantic_search_response_from_cache": semantic_response}

    # Perform vector search with score if semantic search is not relevant
    vector_response = perform_vector_search(query, score)
    if vector_response:
        # Cache the new response with TTL
        cache_response(query, vector_response, ttl=3600)
        return {"vector_search_response_after_cache_failed_to_find", vector_response}

    # If no relevant result found, return a default response
    return {"message": "No relevant information found."}



