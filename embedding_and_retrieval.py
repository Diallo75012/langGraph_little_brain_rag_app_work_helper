# complete workflow to embed documents into a PostgreSQL database using PGVector, perform similarity searches, and integrate the results with additional context from the original PostgreSQL table.
# what is particular with this iteration is that we store the content of the row with the unique id as json object and use it to fetch the UUID from the retireved data to get the original row fromt he database in the SQL side (vs the PGvector side) and then combine both to have a full answer for the llm or the user
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
    # here doc_name is the document where the data have been extracted from, title is the section/chapter.paragraph title, content is the chunck content, retrieved is a boolean to check if document has been retrieved in the past or not to create later on a cache of that data so that it will improve retrieval if we get same kind of query (using special agent node to check that, and cache Redis with TTL and reset of that column so we need a function that does that, one that created the cache based on that column True and put in in Redis with TTL and another which clears the cache everyday.week.month.. and reset that column to False)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id UUID PRIMARY KEY,
            doc_name TEXT,
            title TEXT,
            content TEXT,
            retrieved BOOLEAN,
        );
    """)
    conn.commit()
    cursor.close()
    conn.close()

create_table_if_not_exists()

# REDIS CACHE FUNC
"""
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
schedule.every().day.at("00:00").do(clear_cache)

while True:
    schedule.run_pending()
    time.sleep(1)

"""
# Connect to Redis
redis_client = redis.StrictRedis(host=os.getenv("REDIS_HOST"), port=os.getenv("REDIS_PORT"), db=0)

def cache_document(doc_id: uuid.UUID, content: str, ttl: int = 3600):
    """Cache the document content in Redis with a TTL (time-to-live)."""
    redis_client.setex(str(doc_id), timedelta(seconds=ttl), content)

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


def fetch_documents() -> List[Dict[str, Any]]:
    """Fetch documents from the PostgreSQL database."""
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, content FROM documents")
    rows = cursor.fetchall()
    conn.close()
    return [{'id': row[0], 'content': row[1]} for row in rows]

#### EMBED DOC IN PGVECTOR
def vector_db_create(doc: List[Document], collection: str, connection: str) -> PGVector:
    """Create and store embeddings in PGVector."""
    db_create = PGVector.from_documents(
        embedding=embeddings,
        documents=doc,
        collection_name=collection,
        connection_string=connection,
        distance_strategy=DistanceStrategy.COSINE,
    )
    return db_create

def create_embedding_collection(all_docs: List[Document], collection_name: str, connection_string: str) -> None:
    """Create an embedding collection in PGVector."""
    for doc in all_docs:
        vector_db_create([doc], collection_name, connection_string)

documents = fetch_documents()

# Convert documents to langchain Document format
docs = [Document(page_content=json.dumps({'UUID': str(doc['id']), 'content': doc['content']})) for doc in documents]

create_embedding_collection(docs, COLLECTION_NAME, CONNECTION_STRING)

#### RETRIEVE WITH RELEVANCY SCORE AND FETCH CORRESPONDING POSTGRESQL ROWS
def vector_db_retrieve(collection: str, connection: str, embedding: OllamaEmbeddings) -> PGVector:
    """Retrieve the vector database instance."""
    return PGVector(
        collection_name=collection,
        connection_string=connection,
        embedding_function=embedding,
    )

def retrieve_relevant_vectors(query: str, top_n: int = 2) -> List[Dict[str, Any]]:
    """Retrieve the most relevant vectors from PGVector."""
    db = vector_db_retrieve(COLLECTION_NAME, CONNECTION_STRING, embeddings)
    docs_and_similarity_score = db.similarity_search_with_score(query)
    results = []
    for doc, score in docs_and_similarity_score[:top_n]:
        data = json.loads(doc.page_content)
        results.append({'UUID': data['UUID'], 'content': data['content'], 'score': score})
    return results

def fetch_document_by_uuid(doc_id: uuid.UUID) -> Dict[str, Any]:
    """Fetch the document content by UUID from the PostgreSQL database."""
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM documents WHERE id = %s", (doc_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {'id': row[0], 'title': row[1], 'content': row[2]}
    else:
        return {}

def answer_retriever(query: str) -> Dict[str, Any]:
    """Retrieve answers using PGVector and ChatOllama."""
    relevant_vectors = retrieve_relevant_vectors(query)
    results = []
    for vector in relevant_vectors:
        doc_id = uuid.UUID(vector['UUID'])
        document = fetch_document_by_uuid(doc_id)
        results.append({
            'UUID': vector['UUID'],
            'score': vector['score'],
            'content': vector['content'],
            'row_data': document
        })
    return results

query = "Your query here"
response = answer_retriever(query)
print(json.dumps(response, indent=4))

