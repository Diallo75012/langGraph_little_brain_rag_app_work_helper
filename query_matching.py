# EXACT MATCHING
import redis
import hashlib
import json

# Connect to Redis
redis_client = redis.StrictRedis(host=os.getenv("REDIS_HOST"), port=os.getenv("REDIS_PORT"), db=0)

def get_query_hash(query: str) -> str:
    """Generate a hash for the query."""
    return hashlib.sha256(query.encode()).hexdigest()

def cache_response(query: str, response: str, ttl: int = 3600):
    """Cache the response in Redis with a TTL."""
    query_hash = get_query_hash(query)
    redis_client.setex(query_hash, ttl, response)

def fetch_cached_response(query: str) -> str:
    """Fetch the cached response from Redis if it exists."""
    query_hash = get_query_hash(query)
    response = redis_client.get(query_hash)
    return response.decode() if response else None


######################################################################
#SEMANTIC MATHCING
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.embeddings import OllamaEmbeddings

# Use Ollama to create embeddings
embeddings = OllamaEmbeddings(temperature=0)

# Define connection to PGVector database
CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver=os.getenv("DRIVER"),
    host=os.getenv("HOST"),
    port=int(os.getenv("PORT")),
    database=os.getenv("DATABASE"),
    user=os.getenv("USER"),
    password=os.getenv("PASSWORD"),
)

COLLECTION_NAME = os.getenv("COLLECTION_NAME")

def vector_db_retrieve(collection: str, connection: str, embedding: OllamaEmbeddings) -> PGVector:
    """Retrieve the vector database instance."""
    return PGVector(
        collection_name=collection,
        connection_string=connection,
        embedding_function=embedding,
    )

def perform_vector_search(query: str) -> str:
    """Perform the vector search and return the response."""
    db = vector_db_retrieve(COLLECTION_NAME, CONNECTION_STRING, embeddings)
    docs_and_similarity_score = db.similarity_search_with_score(query)
    if docs_and_similarity_score:
        top_result = docs_and_similarity_score[0]
        if top_result[1] > 0.8:  # Threshold for relevance
            return top_result[0].page_content
    return None
#########################################################################

def handle_query(query: str) -> str:
    """Handle the user query by deciding between cache and vector search."""
    # Try to fetch from cache
    cached_response = fetch_cached_response(query)
    if cached_response:
        return cached_response
    
    # Perform vector search if not found in cache
    vector_response = perform_vector_search(query)
    if vector_response:
        # Cache the new response
        cache_response(query, vector_response)
        return vector_response
    
    # If no relevant result found, return a default response
    return "No relevant information found."

# Example usage
query = "Example user query"
response = handle_query(query)
print(response)
