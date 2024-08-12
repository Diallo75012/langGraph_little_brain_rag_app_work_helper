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
from langchain.docstore.document import Document
from typing import Dict, List, Any
import requests
from bs4 import BeautifulSoup


load_dotenv()

### Create Chunks From Database Data
# function that have th elogic to create chunks from database content and will make sure an overlapping of one row being last in one chunk and firt in the next chunk
def create_chunks(rows: List[Dict[str, Any]], chunk_size: int) -> List[List[Dict[str,Any]]]:
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




#### Create chunks from webpage content
# chunk `text` key and preserve `section` (that will be created later on with a summary title of the `text` and `section`) and `url`
def chunk_text_data(data: Dict[str, Any], chunk_size: int) -> List[Dict[str, Any]]:
    """Chunks the 'text' content while keeping 'section' and 'url' intact."""
    
    print("DATA: ", data, type(data))

    # Initialize variables
    chunks = []
    text = data['text']
    section = data['section']
    url = data['url']

    # Split the text into smaller chunks
    words = text.split()
    current_chunk = []
    current_size = 0

    for word in words:
        word_length = len(word) + 1  # including the space

        # If adding the next word exceeds the chunk size, finalize the current chunk
        if current_size + word_length > chunk_size:
            chunks.append({
                'text': ' '.join(current_chunk),
                'section': section,
                'url': url
            })
            current_chunk = []
            current_size = 0

        # Add the word to the current chunk
        current_chunk.append(word)
        current_size += word_length

    # Add the last chunk if there's any remaining content
    if current_chunk:
        chunks.append({
            'text': ' '.join(current_chunk),
            'section': section,
            'url': url
        })

    return chunks

# put here data scrapped from webpage `{'text': text.strip(),'section': current_title.strip(),'url': url}` and choose chunk size
def create_chunks_from_data(data: List[Dict[str, Any]], chunk_size: int) -> List[Dict[str, Any]]:
    """Creates chunks from the entire dataset."""
    all_chunks = []
    for row in data:
        chunks = chunk_text_data(row, chunk_size)
        all_chunks.extend(chunks)

    return all_chunks

