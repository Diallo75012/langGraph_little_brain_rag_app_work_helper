"""
# embedding_and_retrieval.py

# complete workflow to embed documents into a PostgreSQL database using PGVector, perform similarity searches, and integrate the results with additional context from the original PostgreSQL table.
# what is particular with this iteration is that we store the content of the row with the unique id as json object and use it to fetch the UUID from the retireved data to get the original row from the database in the SQL side (vs the PGvector side) and then combine both to have a full answer for the llm or the user
"""
import os
import json
import psycopg2
import uuid
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_community.vectorstores.pgvector import PGVector
from langchain.docstore.document import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama
import redis
from datetime import timedelta
from typing import Dict, List




#### UTILITY FUNCTIONS & CONF. VARS
load_dotenv()

# Use Ollama to create embeddings
embeddings = OllamaEmbeddings(temperature=0)

# DB CREATION
def connect_db() -> psycopg2.extensions.connection:
    """Connect to the PostgreSQL database."""
    return psycopg2.connect(database=os.getenv("DATABASE"), user=os.getenv("USER"), password=os.getenv("PASSWORD"),
                            host=os.getenv("HOST"), port=os.getenv("PORT"))

# Create the table if it doesn't exist
def create_table_if_not_exists():
    """Create the documents table if it doesn't exist."""
    conn = connect_db()
    cursor = conn.cursor()
    """
    # here doc_name is the document where the data have been extracted from, title is the section/chapter.paragraph title, content is the chunck content, retrieved is a boolean to check if document has been retrieved in the past or not to create later on a cache of that data so that it will improve retrieval if we get same kind of query (using special agent node to check that, and cache Redis with TTL and reset of that column so we need a function that does that, one that created the cache based on that column True and put in in Redis with TTL and another which clears the cache everyday.week.month.. and reset that column to False)
    """
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id UUID PRIMARY KEY,
            doc_name TEXT,
            title TEXT,
            content TEXT,
            retrieved BOOLEAN
        );
    """)
    conn.commit()
    cursor.close()
    conn.close()

create_table_if_not_exists()

# REDIS CACHE FUNC

# Connect to Redis
redis_client = redis.StrictRedis(host=os.getenv("REDIS_HOST"), port=os.getenv("REDIS_PORT"), db=0)

# cache doc in Redis function
def cache_document(doc_id: uuid.UUID, content: str, ttl: int = 3600):
    """Cache the document content in Redis with a TTL (time-to-live)."""
    redis_client.setex(str(doc_id), timedelta(seconds=ttl), content)

# clear Redis cache function which sets the `retrieved` column to FALSE
def clear_cache():
    """Clear the Redis cache and reset the retrieved column in PostgreSQL."""
    conn = connect_db()
    cursor = conn.cursor()
    # Flush all keys in Redis
    redis_client.flushdb()
    # Reset the retrieved column to False in PostgreSQL
    cursor.execute("UPDATE documents SET retrieved = FALSE WHERE retrieved = TRUE")
    conn.commit()
    cursor.close()
    conn.close()

# sets the `retireved` column to TRUE. Can be used when retrieval is performed and maybe then also cache that using the function in `query_matching` library
def update_retrieved_status(doc_id: uuid.UUID):
    """Update the retrieved status of the document in PostgreSQL."""
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("UPDATE documents SET retrieved = TRUE WHERE id = %s", (doc_id,))
    conn.commit()
    cursor.close()
    conn.close()


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

# gets all documents from DB
def fetch_documents() -> List[Dict[str, Any]]:
    """Fetch documents from the PostgreSQL database."""
    conn = connect_db()
    cursor = conn.cursor()
    # here can customize using the extra columns that we have in the db and this will help create the object to embed
    cursor.execute("SELECT id, doc_name, title, content FROM documents ORDER BY id")
    rows = cursor.fetchall()
    conn.close()
    return [{'id': row[0], 'content': row[1]} for row in rows]

#### EMBED DOC IN PGVECTOR
# Embeds the documents
def vector_db_create(doc: List[Document], collection: str, connection: str) -> PGVector|dict:
  try:
    """Create and store embeddings in PGVector."""
    db_create = PGVector.from_documents(
        embedding=embeddings,
        documents=doc,
        collection_name=collection,
        connection_string=connection,
        distance_strategy=DistanceStrategy.COSINE,
    )
    return db_create
  except Exception as e:
    return {"error": f"An error occured while trying to embed in vector db -> {e}"}

# function that loops through docs to embed those one by one using the embeding function 'vector_db_create'
def create_embedding_collection(all_docs: List[Document], collection_name: str, connection_string: str) -> None|dict:
  """Create an embedding collection in PGVector."""
  try:
    vector_db_create([all_docs], collection_name, connection_string)
  except Exception as e:
    return {"error": f"An error occured while trying to create embeddings -> {e}"}

# function that created custom document embedding object. Can be used to embed the full database or part of it after web/pdf parsing 
def embed_all_db_documents(all_docs: List[Dict[str,Any]], collection_name: str, connection_string: str) -> None|dict:
  # Convert documents `all_docs` to langchain Document format (List)
  # `all_docs` here is a parameter that is representing our chunk that we want to embed it has the inofrmationof several rows
  docs = [Document(page_content=json.dumps(all_docs)),]

  # embed all those documents in the vectore db
  try:
    create_embedding_collection(docs, collection_name, connection_string)
  except Exception as e:
    return {"error": f"An error occured while trying embed documents -> {e}"}

#### RETRIEVE WITH RELEVANCY SCORE AND FETCH CORRESPONDING POSTGRESQL ROWS

# embedding retrieval func
def vector_db_retrieve(collection: str, connection: str, embedding: OllamaEmbeddings) -> PGVector:
    """Retrieve the vector database instance."""
    return PGVector(
        collection_name=collection,
        connection_string=connection,
        embedding_function=embedding,
    )
# retrieve `n` releavnt embedding to user query with score
def retrieve_relevant_vectors(query: str, top_n: int = 3) -> List[Dict[str, Any]]:
    """Retrieve the most relevant vectors from PGVector."""
    db = vector_db_retrieve(COLLECTION_NAME, CONNECTION_STRING, embeddings)
    docs_and_similarity_score = db.similarity_search_with_score(query)
    results = []
    # `docs` here is List[Document]
    for docs, score in docs_and_similarity_score[:top_n]:
        # one doc store can ahve several chunk object in it which are dict with keys UUID and content
        # here `elem` is a List[Dict[str,Any]]
        for elem in docs:
          data = json.loads(elem.page_content)
          # here `info` is Dict[str,Any]
          for info in data:
            # we form an object dict that we want to collect from vector db with the relevant keys
            results.append({'UUID': info['UUID'], 'content': info['content'], 'score': score})
    return results

# use the UUID of the retirved doc to fetch the postgresql database row and get extra infos from it later on, it is also to verify the qulity of embedding and catch errors if the content is different for example (at the beginning of dev work then when we are sure of the code we get rid of the checks, but we will definetly check thatthe content is the same from embedding to database row content.).
def fetch_document_by_uuid(doc_id: uuid.UUID) -> Dict[str, Any]:
    """Fetch the document content by UUID from the PostgreSQL database."""
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM documents WHERE id = %s", (doc_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {'id': row[0], 'doc_name': row[1], 'title': row[2], 'content': row[3]}
    else:
        return {}

#### BUSINESS LOGIC OF RETIREVAL: goes to db qith query and get relevance score and then goes to the postgresql db to getthe rest of the row content to form a nice context for quality answer.
def answer_retriever(query: str, relevance_score: float, top_n: int) -> List[Dict[str, Any]]:
    """Retrieve answers using PGVector and ChatOllama."""
    relevant_vectors = retrieve_relevant_vectors(query, top_n)
    results = []
    for vector in relevant_vectors:
      if vector["score"] > relevance_score:
        print("Vector: ", vector, ", Vectore score: ", vector["score"], f" vs relevant score: {relevant_score}")
        for doc in vector:
          print("Doc: ", doc)
          doc_id = uuid.UUID(doc['UUID'])
          # here document will be a dict with keys: id, doc_name, title, content coming from the postgresql db, saved in the dict with key raw_data
          document = fetch_document_by_uuid(doc_id)
          results.append({
            'UUID': doc['UUID'],
            'score': doc['score'],
            'content': doc['content'],
            'row_data': document # document is a Dict with id, doc_name, title, content as key (full db row)
          })
    return results











