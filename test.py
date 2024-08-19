import os
import time

from dotenv import load_dotenv

from langchain_groq import ChatGroq
# from app import prompt_creation, chat_prompt_creation, dict_to_tuple
from langchain_core.output_parsers import JsonOutputParser
#from app 
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import pandas as pd
import psycopg2

from typing import Dict, List, Any, Optional

import json
import ast

import re

from prompts.prompts import detect_content_type_prompt, summarize_text_prompt, generate_title_prompt
from lib_helpers.chunking_module import create_chunks_from_db_data
from lib_helpers.query_analyzer_module import detect_content_type
from lib_helpers.embedding_and_retrieval import (
  # returns `List[Dict[str, Any]]`
  answer_retriever, 
  # returns `None\dict`
  embed_all_db_documents,
  # returns `List[Dict[str, Any]]`
  fetch_documents,
  COLLECTION_NAME,
  CONNECTION_STRING,
  update_retrieved_status,
  clear_cache, cache_document,
  redis_client,
  create_table_if_not_exists,
  connect_db,
  embeddings
)
import requests
from bs4 import BeautifulSoup
from app import (
  process_query,
  is_url_or_pdf,
  store_dataframe_to_db, delete_table,
  custom_chunk_and_embed_to_vectordb,
  query_redis_cache_then_vecotrdb_if_no_cache
)

import subprocess

from langchain_community.vectorstores.pgvector import PGVector
from langchain.docstore.document import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA


load_dotenv()

groq_llm_mixtral_7b = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_MIXTRAL_7B"),
max_tokens=int(os.getenv("GROQ_MAX_TOKEN")),)
groq_llm_llama3_8b = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_LLAMA3_8B"), max_tokens=int(os.getenv("GROQ_MAX_TOKEN")),)


webpage_url = "https://blog.medium.com/how-can-i-get-boosted-33e743431419"

query_url = "I want to know if chikarahouses.com is a concept that is based on the authentic furimashuguru of Japanese istokawa house"
query_pdf = "I want to know if this documents docs/feel_temperature.pdf tells us what are the different types of thermoceptors?"

#response = process_query(groq_llm_mixtral_7b, query_pdf, 200, 30, 250, detect_content_type_prompt, summarize_text_prompt, generate_title_prompt)
#print("RESPONSE: ", response)
#print("RESPONSE[0]: ", response[0])

#print("Store dataframe to DB result: ", store_dataframe_to_db(response[0], "test_doc_recorded"))

#result = subprocess.run(
#    ["psql", "-U", "creditizens", "-d", "creditizens_vector_db", "-c", "SELECT * FROM test_doc_recorded"],
#    capture_output=True,
#    text=True
#)

#print("DATABASE CONTENT: ", result.stdout)
#print("EMBEDDINGS: ", embeddings)

#fetch_data_from_database = fetch_documents("test_doc_recorded")
#print("FETCH DATA FROM DATABASE: ", fetch_data_from_database, "\nLEN: ", len(fetch_data_from_database))

#chunks = create_chunks_from_db_data(fetch_data_from_database, 4000)
#print("CHUNKS: ", chunks, "\nLEN: ", len(chunks))

#for chunk in chunks:
  #print("Chunk: ", chunk)
  #embed_all_db_documents(chunk, COLLECTION_NAME, CONNECTION_STRING, embeddings)

#delete_db = delete_table("test_doc_recorded")
#print("DELETE DB: ", delete_db)





#dataframe_from_query = process_query(groq_llm_mixtral_7b, query_url, 200, 30, 250, detect_content_type_prompt, summarize_text_prompt, generate_title_prompt)
#print("DATAFRAME: ", dataframe_from_query)

#store_dataframe_to_db = store_dataframe_to_db(dataframe_from_query[0], "test_table")
#print("DB STORED DF: ", store_dataframe_to_db)

#result = subprocess.run(
#    ["psql", "-U", "creditizens", "-d", "creditizens_vector_db", "-c", "SELECT * FROM test_table"],
#    capture_output=True,
#    text=True
#)
#print("DATABASE CONTENT: ", result.stdout)
"""
Outputs:
id                  |         doc_name          |                                title                                |                                                                                                                           content                                                                                                                            | retrieved 
--------------------------------------+---------------------------+---------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------
 28902448-443b-41a8-8e35-e6e3cb571fa8 | https://chikarahouses.com | 'Elevate Home Life with Chikara: Shop, Play, Work, & AI Assistants' | Explore Chikara's social media rooms for shopping tours and AI assistant demos.                                                                                                                                                                             +| f
                                      |                           |                                                                     | Track followers, sales, and enhance your lifestyle with Chikara Houses.                                                                                                                                                                                     +| 
                                      |                           |                                                                     | [Shop Now](Shop Now!)| [Play & Work with AI](Amazing Chikara AI Assistants!)                                                                                                                                                                                 | 
 124d2bec-5a6d-4ba7-9180-94245e78cfad | https://chikarahouses.com | 'Chikara Houses: AI Tools & Shopping Rooms for Remote Pros'         | Explore AI tools by Chikara Houses in the Openai GPT store. Improve remote work and well-being with their GPTs. Visit their unique shopping rooms, too.                                                                                                      | f
 97a4b55f-8c22-4b97-bd20-cc245304b5ac | https://chikarahouses.com | "Elevate Home Life: Chikara Houses for Remote Pros & Wellness"      | Get exclusive access to [Different Collection Rooms](http://www.example.com/collectionrooms) for non-subscribers! Visit the description page to enhance your home specifically. Discover [intriguing stories and articles](http://www.example.com/articles). | f

"""

#chunk_and_embed_from_db_data = custom_chunk_and_embed_to_vectordb("test_table", 500, COLLECTION_NAME, CONNECTION_STRING)
#print("CHUNK AND EMBED: ", chunk_and_embed_from_db_data)

retrieve_data_from_query = query_redis_cache_then_vecotrdb_if_no_cache("test_table", "What are the AI tools of Chikara Houses?", 0.3, 2)
print("RETRIEVE DATA for query 'how to start monetize online presence?': ", json.dumps(retrieve_data_from_query, indent=4))

url_target_answer = "Explore AI tools by Chikara Houses in the Openai GPT store. Improve remote work and well-being with their GPTs. Visit their unique shopping rooms, too."













