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

# create hash of query or embedding list, using hashlib and not python hash() for consistency and cryptographic hashing sha256
def get_query_hash(query: str|list) -> str:
    """Generate a hash for the query."""
    if isinstance(query, list):
      query = str(tuple(query))
      print("\n\nQuery is a list therefore an embedding, see str(tuple(version)) before hash: ", query, "Type: ", type(query))
    print("\n\nQuery is a str before hash version: ", query, "Type: ", type(query))
    return hashlib.sha256(query.encode()).hexdigest()

# Record in Redis Both Kind of Keys (hashed for exact match and vector for semantic match)
def cache_response(query: str, response: List[Dict[str, Any]], ttl: int = 3600) -> dict:
    """Cache the response in Redis with both hash and vector representation."""
    query_hash = get_query_hash(query)
    print("\n\nQuery hash: ", query_hash)
    query_vector = EMBEDDINGS.embed_query(query)

    # Store Query as Hash Key with Response as Value
    REDIS_CLIENT.setex(query_hash, timedelta(seconds=ttl), json.dumps(response))

    # Store with query vector as key but use the function to change the list embedding to a tuple and then str and then hash
    query_vector_tuple_hash = get_query_hash(query_vector)
    print("\n\nQuery vector str tupled hash: ", query_vector_tuple_hash)
    vector_key = f"vector:{query_vector_tuple_hash}"

    # Store Query Embeddings as Hash Key and as Value the Vector of the Query for Semantic Search and the Response to Render Corresponding Answer
    REDIS_CLIENT.setex(vector_key, timedelta(seconds=ttl), json.dumps({
        "embedding": query_vector,  # Store original embedding for easy fetching when performing similarity search
        "response": response
    }))
    
    return {"query_hash": query_hash, "query_vector": query_vector, "query_vector_tuple_hash": query_vector_tuple_hash}


# exact match search fucntions
def fetch_cached_response_by_hash(query: str) -> dict|List[Dict[str,Any]]|None:
    """Fetch the cached response from Redis using query hash if it exists."""
    response = []
    
    # search using hash of query
    query_hash = get_query_hash(query)
    data_from_query_hash = REDIS_CLIENT.get(query_hash)
    print("Get from Redis using query ONLY hash: ", query_hash, " ; Response from Redis: ", data_from_query_hash)
    if data_from_query_hash:
      response.append(json.loads(data_from_query_hash) if data_from_query_hash else None)
    # search using hash of vector of query
    query_vector = EMBEDDINGS.embed_query(query)
    hash_query_vector_tuple = get_query_hash(query_vector)
    data_from_vector_hash = REDIS_CLIENT.get(hash_query_vector_tuple)
    print("Get from Redis using query VECTOR(tuppled) hash: ", hash_query_vector_tuple, " ; Response from Redis: ", data_from_vector_hash)
    if data_from_vector_hash:
      response.append(json.loads(data_from_vector_hash) if data_from_vector_hash else None)
    return response

# fetch all for semantic search only on embeddings stored in Redis
def fetch_all_cached_embeddings() -> dict:
    """Fetch all cached embeddings from Redis."""
    # Fetch all keys in Redis that are prefixed with 'vector:'
    keys = REDIS_CLIENT.keys('vector:*')
    all_embeddings = {}
    
    # Iterate over each key and retrieve the associated embedding and response
    for key in keys:
        # Extract the embedding hash from the Redis key
        embedding_hash = key.decode().split("vector:")[1]
        
        # Retrieve the data from Redis
        data = json.loads(REDIS_CLIENT.get(key))
        
        # Store the embedding and its associated response in the dictionary
        all_embeddings[embedding_hash] = {
            "embedding": data["embedding"],  # The original embedding
            "response": data["response"]     # The response associated with this embedding
        }
    
    return all_embeddings


# Semantic search function
def perform_semantic_search_in_redis(query_embedding: list, threshold: float = 0.7) -> List[Dict[str, Any]]|None:
    """Perform semantic search on the cached query embeddings in Redis."""

    # Fetch all cached embeddings from Redis.
    all_embeddings = fetch_all_cached_embeddings()
    print("All embeddings for seamantic search in Redis: ", all_embeddings)

    # If there are no embeddings in Redis, return None.
    if not all_embeddings:
        return None
    
    # Convert the query embedding list into a NumPy array and reshape it for similarity calculation.
    query_embedding_np = np.array(query_embedding).reshape(1, -1)

    # Initialize variables to track the highest similarity and the best matching response.
    max_similarity = 0
    best_match_response = None

    # Iterate through all cached embeddings.
    for data in all_embeddings.values():
        # Convert the stored embedding list into a NumPy array and reshape it.
        embedding_np = np.array(data["embedding"]).reshape(1, -1)

        # Calculate the cosine similarity between the query embedding and the current embedding.
        similarity = cosine_similarity(query_embedding_np, embedding_np)[0][0]
        print("Similarity 'cosine' from perform_semantic_search on Redis: ", similarity)

        # If the calculated similarity is higher than the current maximum, update the maximum and store the corresponding response.
        if similarity > max_similarity:
            max_similarity = similarity
            best_match_response = data["response"]
    
    # If the highest similarity found is above the threshold, return the corresponding response.
    if max_similarity >= threshold:
        return best_match_response
    
    # If no suitable match is found, return None.
    return None


# Vector search function with postgresql table update and caching of new retrieved content
def perform_vector_search(table_name: str, query: str, score: float, top_n: int) -> List[Dict[str, Any]]:

  doc_ids = []
   
  response = answer_retriever(table_name, query, score, top_n)
  print("JSON RESPONSE: ", json.dumps(response, indent=4))
  for elem in response:
    doc_ids.append(elem["UUID"])

  # as it as been retrieved we use the function `update_retrieved_status` from the library `embedding_and_retrieval` to update the db column `retrieved` to True
  # this could be done somewhere else as well inside 'answer_retriever' with all retireved data being labelled as `retireved` in the db and cached
  print("DOC IDS LIST", doc_ids)
  for doc_id in doc_ids:
    update_retrieved_status(table_name, doc_id)

  # then cache the query with the result content obtained in the hash way and the vectorized way
  cache_response(query, response, ttl=86400) # 86400=1d, 3600=1h
  return response

#### This the meat function which does the exact match, then, semantic match, then the expensive vector database retireval and cache it if it finds anything for the response
def handle_query_by_calling_cache_then_vectordb_if_fail(table_name: str, query: str, score: float, top_n: int) -> Any:
    """Handle the user query by deciding between cache, semantic search, and vector search."""
    
    try:
      # vectorize the query for semantic search
      query_vectorized = EMBEDDINGS.embed_query(query)
    
      #print("Initial Query: ", query, "Query Vectorized: ", query_vectorized)

      # Try to fetch from cache (Exact Query Match)
      try:
        # returns List[Dict]
        cached_response = fetch_cached_response_by_hash(query)
        if cached_response:
          return {"exact_match_search_response_from_cache": cached_response}
      except Exception as e:
        print(f"An error occured while trying to fetch cached response by hash: {e}")
        return {"error": f"An error occured while trying to fetch cached response by hash: {e}"}

      # Perform semantic search if exact query is not found
      try:
        semantic_response = perform_semantic_search_in_redis(query_vectorized, score)
        print("Semantic search in Redis response (as hash query not found response): ", semantic_response)
        if semantic_response:
          return {"semantic_search_response_from_cache": semantic_response}
      except Exception as e:
        print(f"An error occured while trying to perform semantic search in redis: {e}")
        return {"error": f"An error occured while trying to perform semantic search in redis: {e}"}


      # Perform vector search with score if semantic search is not relevant
      try:
        vector_response = perform_vector_search(table_name, query, score, top_n)
        print("Cache had nothing, therefore, performing vectordb search, response: ", vector_response)
        if vector_response:
            return {"vector_search_response_after_cache_failed_to_find": vector_response}
            # Cache the new response with TTL
            try:
              cache_response(query, vector_response, ttl=3600)
            except Exception as e:
              return {"error": f"An error occured while trying to cache the vector search response: {e}"}
      except Exception as e:
        print(f"An error occured while trying to perform vectordb search query {e}")
        return {"error": f"An error occured while trying to perform vectordb search query: {e}"}

    except Exception as e:
      return {"error": f"An error occured while trying to 'handle_query_by_calling_cache_then_vectordb_if_fail': {e}"}
    
    # If no relevant result found, return a default response
    return {"message": f"No relevant information found in Cache or VectorDB, Please make an Internet Search to answer the to this query -> {query}."}



