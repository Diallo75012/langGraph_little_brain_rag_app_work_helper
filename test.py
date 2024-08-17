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
from app import process_query, is_url_or_pdf, store_dataframe_to_db, delete_table

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
print("EMBEDDINGS: ", embeddings)

fetch_data_from_database = fetch_documents("test_doc_recorded")
print("FETCH DATA FROM DATABASE: ", fetch_data_from_database, "\nLEN: ", len(fetch_data_from_database))

chunks = create_chunks_from_db_data(fetch_data_from_database, 4000)
print("CHUNKS: ", chunks, "\nLEN: ", len(chunks))

for chunk in chunks:
  print("Chunk: ", chunk)
  embed_all_db_documents(chunk, COLLECTION_NAME, CONNECTION_STRING, embeddings)

#delete_db = delete_table("test_doc_recorded", conn)
#print("DELETE DB: ", delete_db)














