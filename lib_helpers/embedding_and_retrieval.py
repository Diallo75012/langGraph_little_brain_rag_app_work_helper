"""
# embedding_and_retrieval.py

# complete workflow to embed documents into a PostgreSQL database using PGVector, perform similarity searches, and integrate the results with additional context from the original PostgreSQL table.
# what is particular with this iteration is that we store the content of the row with the unique id as json object and use it to fetch the UUID from the retireved data to get the original row from the database in the SQL side (vs the PGvector side) and then combine both to have a full answer for the llm or the user
"""
import os
import json
import psycopg2
from psycopg2 import sql
import uuid
from typing import List, Dict, Any
from langchain_community.vectorstores.pgvector import PGVector, DistanceStrategy
from langchain.docstore.document import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
import redis
from datetime import timedelta
from dotenv import load_dotenv


#### UTILITY FUNCTIONS & CONF. VARS
# load env vars
load_dotenv(dotenv_path='.env', override=False)
load_dotenv(dotenv_path=".vars", override=True)

# Use Ollama to create embeddings
embeddings = OllamaEmbeddings(model="mistral:7b", temperature=int(os.getenv("EMBEDDINGS_TEMPERATURE")))

# DB CREATION
def connect_db() -> psycopg2.extensions.connection:
    """Connect to the PostgreSQL database."""
    return psycopg2.connect(database=os.getenv("DATABASE"), user=os.getenv("USER"), password=os.getenv("PASSWORD"),
                            host=os.getenv("HOST"), port=os.getenv("PORT"))

# Create the table if it doesn't exist
def create_table_if_not_exists(table_name: str = os.getenv("TABLE_NAME")):
    """Create the documents table if it doesn't exist."""
    try:
      conn = connect_db()
      cursor = conn.cursor()
    except Exception as e:
      print(f"Failed to connect to the database: {e}")
      return {"messages": [{"role": "system", "content": f"error: An error occured while trying to connect to DB in 'store_dataframe_to_db': {e}"}]}

    """
    # Here:
      - doc_name: is the document where the data have been extracted from
      - title: is the section/chapter/paragraph title
      - content: is the chunck content
      - retrieved: is a boolean to check if document has been retrieved in the past or not to create later on, a cache of that data so that it will improve retrieval if we get same kind of query (using special agent node to check that, and cache Redis with TTL and reset of that column so we need a function that does that, one that created the cache based on that column True and put in Redis with TTL and another which clears the cache everyday.week.month.. and reset that column to False)
    """
    # table is called `test_table` make sure to change name for application when tests are done
    try:
      cursor.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS {} (
            id UUID PRIMARY KEY,
            doc_name TEXT,
            title TEXT,
            content TEXT,
            retrieved BOOLEAN
        );
      """).format(sql.Identifier(table_name)),)
      cursor.execute(sql.SQL("UPDATE {} SET retrieved = 'TRUE' WHERE id = %s;").format(sql.Identifier(table_name)), [str(doc_id)])
      conn.commit()
      cursor.close()
      conn.close()
    except Exception as e:
      return {"messages": [{"role": "ai", "content": f"error while trying to check if table exists: {e}"}]}

create_table_if_not_exists(os.getenv("TABLE_NAME"))



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



# REDIS CACHE FUNC
# Connect to Redis
redis_client = redis.StrictRedis(host=os.getenv("REDIS_HOST"), port=os.getenv("REDIS_PORT"), db=0)

# cache doc in Redis function: qyeru_matching library have it's own cache_response so this function can be used as helper if needed
def cache_document(doc_id: uuid.UUID, content: str, ttl: int = int(os.getenv("TTL"))):
    """Cache the document content in Redis with a TTL (time-to-live)."""
    redis_client.setex(str(doc_id), timedelta(seconds=ttl), content)

# clear Redis cache function which sets the `retrieved` column to FALSE
def clear_cache(table_name: str):
    """Clear the Redis cache and reset the retrieved column in PostgreSQL."""
    conn = connect_db()
    cursor = conn.cursor()
    # Flush all keys in Redis
    redis_client.flushdb()
    # Reset the retrieved column to False in PostgreSQL
    cursor.execute(sql.SQL("UPDATE {} SET retrieved = 'FALSE' WHERE retrieved = 'TRUE';").format(sql.Identifier(table_name)))
    conn.commit()
    cursor.close()
    conn.close()

# sets the `retireved` column to TRUE. Can be used when retrieval is performed and maybe then also cache that using the function in `query_matching` library
def update_retrieved_status(table_name: str, doc_id: str):
    """Update the retrieved status of the document in PostgreSQL."""
    conn = connect_db()
    cursor = conn.cursor()
    # `doc_id` is a `uuid.UUID` and has to be passed as `str` for database sql query
    cursor.execute(sql.SQL("UPDATE {} SET retrieved = 'TRUE' WHERE id = %s;").format(sql.Identifier(table_name)), [str(doc_id)])
    conn.commit()
    cursor.close()
    conn.close()

# gets all documents from DB
def fetch_documents(table_name: str) -> List[Dict[str, Any]]:
    """Fetch documents from the PostgreSQL database."""
    conn = connect_db()
    cursor = conn.cursor()
    # here can customize using the extra columns that we have in the db and this will help create the object to embed
    #cursor.execute("SELECT id, doc_name, title, content FROM documents ORDER BY id")
    # Indexes for rows: 0-id 1-doc_name 2-title 3-content
    cursor.execute(sql.SQL("SELECT id, doc_name, title, content FROM {} ORDER BY id;").format(sql.Identifier(table_name)))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [{'id': row[0], 'content': row[3]} for row in rows]

#### EMBED DOC IN PGVECTOR
# Embeds the documents
def vector_db_create(doc: List[Document], collection: str, connection: str, embeddings: OllamaEmbeddings = embeddings) -> PGVector|dict:
  try:
    """Create and store embeddings in PGVector."""
    db_create = PGVector.from_documents(
        embedding=embeddings,
        documents=doc,
        collection_name=collection,
        connection_string=connection,
        distance_strategy=DistanceStrategy.COSINE,
        #distance_strategy="cosine", # can be "eucledian", "hamming", "cosine"EUCLEDIAN, COSINE, HAMMING
    )
    return db_create
  except Exception as e:
    print(f"An error occurred while trying to embed in vector db -> {e}")
    return {"error": f"An error occurred while trying to embed in vector db -> {e}"}

# function that loops through docs to embed those one by one using the embeding function 'vector_db_create'
def create_embedding_collection(all_docs: List[Document], collection_name: str, connection_string: str, embeddings: OllamaEmbeddings = embeddings) -> None|dict:
  """Create an embedding collection in PGVector."""
  try:
    vector_db_create(all_docs, collection_name, connection_string, embeddings)
    print("\n\nBatch doc embedding done!\n\n")
  except Exception as e:
    return {"error": f"An error occured while trying to create embeddings -> {e}"}

# function that created custom document embedding object. Can be used to embed the full database or part of it after web/pdf parsing 
def embed_all_db_documents(all_docs: List[Dict[str,Any]], collection_name: str, connection_string: str, embeddings: OllamaEmbeddings = embeddings) -> None|dict:
  # `all_docs` here is a parameter that is representing our chunk that we want to embed it has the inofrmationof several rows
  # `all_docs` is a `List[Dict[str,Any]]`
  docs = [Document(page_content=json.dumps(all_docs)),]
  print("DOCS: ", docs)

  # embed all those documents in the vectore db
  try:
    create_embedding_collection(docs, collection_name, connection_string, embeddings)
    return {"success": "Data has been successfully embedded."}
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
    
    # Can search using metadata `key` using: db.similarity_search_with_score(query, filter={"metadata.key": "value"}) or  if using postgresql table column insstead of collection(pgvector) to store metadata as JSONB, use `$` in front of metadata key `db.similarity_search_with_score(query, filter={"metadata.$key": "value"})`
    # Here `docs_and_similarity_score` returns a `List[Tuple[Documents, score]]`
    docs_and_similarity_score = db.similarity_search_with_score(query)
    print("Type docs_and_similarity_score: ", type(docs_and_similarity_score), "\nContent: ", docs_and_similarity_score)
    results = []

    # Iterate over each tuple in the result
    for doc, score in docs_and_similarity_score[:top_n]:
        # Parse the JSON content from the Document
        data = json.loads(doc.page_content)
        
        # Iterate through the parsed JSON data to extract relevant info
        for info in data:
            # Append the extracted info to the results list
            results.append({'UUID': info['UUID'], 'content': info['content'], 'score': score})
    
    return results

# use the UUID of the retirved doc to fetch the postgresql database row and get extra infos from it later on, it is also to verify the qulity of embedding and catch errors if the content is different for example (at the beginning of dev work then when we are sure of the code we get rid of the checks, but we will definetly check thatthe content is the same from embedding to database row content.).
def fetch_document_by_uuid(table_name: str, doc_id: str) -> Dict[str, Any]:
    """Fetch the table content by UUID from the PostgreSQL database."""
    conn = connect_db()
    cursor = conn.cursor()
    print("Doc id: ", doc_id, " , Type: ", type(doc_id))
    # doc_id is a uuid.UUID we need to pass in the query the `str` version of it
    cursor.execute(sql.SQL("SELECT * FROM {} WHERE id = %s;").format(sql.Identifier(table_name)), [str(doc_id)])
    row = cursor.fetchone()
    print("ROW Found in DB: ", row)
    cursor.close()
    conn.close()
    if row:
        return {'id': row[0], 'doc_name': row[1], 'title': row[2], 'content': row[3]}
    else:
        return {"Error": f"No row found in database for doc_id: {doc_id}"}

#### DELETE TABLE TO DELETE THE COLLECTION
# Delete All Rows from the Vector Table
def clear_vector_table(table_name: str) -> str:
    """
    Clear all data from a table storing vector embeddings.
    
    Args:
    table_name (str): The name of the table to clear.
    conn (psycopg2.extensions.connection): The connection object to the PostgreSQL database.
    
    Returns:
    str: A message indicating the result of the operation.
    """
    try:
        conn = connect_db()
        cursor = conn.cursor()
        # Delete all rows from the table
        cursor.execute(sql.SQL("DELETE FROM {};").format(sql.Identifier(table_name)))
        conn.commit()
        return f"All rows in table '{table_name}' have been successfully deleted."
    except Exception as e:
        conn.rollback()
        return f"Failed to clear table '{table_name}': {e}"
    finally:
        cursor.close()

# Delete table entirely : bye bye table
# function to delete table hat can be used as a tool
def delete_table(table_name: str) -> Dict[str, str]:
    """
    Delete a table from the PostgreSQL database using psycopg2.
    
    Args:
    table_name (str): The name of the table to delete.
    """
    # Establish a connection to the PostgreSQL database
    try:
        conn = connect_db()
        cursor = conn.cursor()
    except Exception as e:
        print(f"Failed to connect to the database: {e}")
        return {"error": f"An error occured while trying to connect to table in 'delete_table': {e}"}

    # Delete the table if it exists
    try:
        cursor.execute(sql.SQL("DROP TABLE IF EXISTS {};").format(sql.Identifier(table_name)))
        conn.commit()
        print(f"Table {table_name} deleted successfully.")
    except Exception as e:
        print(f"Failed to delete table {table_name}: {e}")
        conn.rollback()  # Rollback in case of error for the current transaction
        return {"error": f"An error occured while trying to drop the table {table_name}: {e}"}
    finally:
        # Close the cursor and the connection
        cursor.close()
        conn.close()
        return {"success": f"Table {table_name} have been deleted."}

#### BUSINESS LOGIC OF RETIREVAL: goes to db qith query and get relevance score and then goes to the postgresql db to getthe rest of the row content to form a nice context for quality answer.
def answer_retriever(table_name: str, query: str, relevance_score: float, top_n: int) -> List[Dict[str, Any]]:
    """Retrieve answers using PGVector"""
    relevant_vectors = retrieve_relevant_vectors(query, top_n)
    print("Type relevant_vectors: ", type(relevant_vectors), "\nContent: ", relevant_vectors)
    results = []
    for vector in relevant_vectors:
      if vector["score"] > relevance_score:
        print("Vector: ", vector, ", Vectore score: ", vector["score"], f" vs relevant score: {relevance_score}")
        #for doc in vector:
        print("Vector: ", vector)
        doc_id = vector['UUID']
        print("Doc_id before fetch data from DB: ", doc_id, "Type: ",type(doc_id))
        # here document will be a dict with keys: id, doc_name, title, content coming from the postgresql db, saved in the dict with key raw_data
        document = fetch_document_by_uuid(table_name, doc_id)
        results.append({
            'UUID': vector['UUID'],
            'score': vector['score'],
            'content': vector['content'],
            'row_data': document # document is a Dict with id, doc_name, title, content as key (full db row)
        })
    return results











