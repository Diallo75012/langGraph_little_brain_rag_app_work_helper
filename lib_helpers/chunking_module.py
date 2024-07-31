"""
Here we are going to create our own chunking process. As we are not satisfied by what is there at the moment.
We get the document data from each row and create  chuncks by grouping rows in one chunk.
The rule here is that we give a number for the chunk size, and it is going to fit rows until the chunk size is reached. The last row of each chunk is th efirst row of the next one this for the overlapping aspect.
We import the library embedding_an_retrieval and will embbed the chunks that we will create based on the content store in the postgresql database
which as been collected by cleaning the document and storing the auqlity content only in postgresql.
"""
import psycopg2
import os
import json
from dotenv import load_dotenv
from embedding_and_retrieval import connect_db, fetch_documents, vector_db_create, embeddings, CONNECTION_STRING, COLLECTION_NAME
from langchain.docstore.document import Document

load_dotenv()

def fetch_all_documents(conn):
    """Fetch all documents from the PostgreSQL database."""
    cursor = conn.cursor()
    cursor.execute("SELECT id, content FROM documents ORDER BY id")
    rows = cursor.fetchall()
    cursor.close()
    return rows

def create_chunks(rows, chunk_size):
    """Create chunks with overlapping content from the fetched rows."""
    chunks = []
    chunk = []
    current_size = 0

    for i, (doc_id, content) in enumerate(rows):
        doc_length = len(content)

        if current_size + doc_length > chunk_size and chunk:
            # Ensure the last row of the current chunk is the first row of the next chunk
            next_chunk = [{'UUID': chunk[-1]['UUID'], 'content': chunk[-1]['content']}]
            chunks.append(chunk)
            chunk = next_chunk
            current_size = len(chunk[-1]['content'])

        chunk.append({'UUID': str(doc_id), 'content': content})
        current_size += doc_length

    # Add the last chunk if it has content
    if chunk:
        chunks.append(chunk)

    return chunks

def embed_chunks(chunks):
    """Embed chunks using the embedding_and_retrieval module."""
    for chunk in chunks:
        chunk_text = json.dumps(chunk)
        document = Document(page_content=chunk_text)
        vector_db_create([document], COLLECTION_NAME, CONNECTION_STRING)

def main(chunk_size):
    """Main function to create and embed chunks."""
    conn = connect_db()
    rows = fetch_all_documents(conn)
    chunks = create_chunks(rows, chunk_size)
    embed_chunks(chunks)
    conn.close()

#if __name__ == "__main__":
 #   chunk_size = 1000  # Define your chunk size here
 #   main(chunk_size)
