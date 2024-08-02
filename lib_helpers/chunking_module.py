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
from embedding_and_retrieval import connect_db, fetch_documents, vector_db_create, embeddings, embed_all_db_documents,CONNECTION_STRING, COLLECTION_NAME
from langchain.docstore.document import Document
from typing import Dict, List


load_dotenv()

# function that have th elogic to create chunks from database content and will make sure an overlapping of one row being last in one chunk and firt in the next chunk
def create_chunks(rows: List[Dict[str, any]], chunk_size: int) -> List[List[Dict[str,any]]]:
    """Create chunks with overlapping content from the fetched rows."""
    chunks = []
    chunk = []
    current_size = 0

    for i, row in enumerate(rows):
        # we keep it simple but we could add the `doc_name` and `title` from the table as metadata of the chunk but need to change `fetch_documents` output to have all columns
        doc_id = row['id']
        content = row['content']
        doc_length = len(content)
        
        # Check if adding the current row exceeds the chunk size
        if current_size + doc_length > chunk_size and chunk:
            # Ensure the last row of the current chunk is the first row of the next chunk
            next_chunk = [{'UUID': chunk[-1]['UUID'], 'content': chunk[-1]['content']}]
            # append to the full chunk to final list of custom chunks
            chunks.append(chunk)
            # initalize the row accumulator with last row of previous chunk
            chunk = next_chunk
            # initalize length considering this previous row added for next row accumulation
            current_size = len(chunk[-1]['content'])
        
        # accumulated row data to build up chunk until it reaches chunk size
        chunk.append({'UUID': str(doc_id), 'content': content})
        current_size += doc_length

    # Add the last chunk if it has content
    if chunk:
        chunks.append(chunk)

    return chunks

