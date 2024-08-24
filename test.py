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
  store_dataframe_to_db,
  delete_table,
  custom_chunk_and_embed_to_vectordb,
  query_redis_cache_then_vecotrdb_if_no_cache,
  # tool function
  internet_research_user_query,
  # graph conditional adge helper function
  decide_next_step
)

import subprocess

from langchain_community.vectorstores.pgvector import PGVector
from langchain.docstore.document import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA

from app_states.app_graph_states import StateCustom
# for graph creation and management
from langgraph.checkpoint import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

# display drawing of graph
from IPython.display import Image, display

# tools
from app_tools.app_tools import internet


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



def get_user_input(state: MessagesState):
  user_input =  input("Do you need any help? any PDF doc or webpage to analyze? ")

  messages = state['messages']
  
  return {"messages": [user_input]}

def answer_user(state: MessagesState):
  messages = state['messages']
  print("Message state: ", messages)
  last_message = messages[-1].content
  return {"messages": [{"role": "ai", "content": last_message}]}

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

#chunk_and_embed_from_db_data = custom_chunk_and_embed_to_vectordb("test_table", 500, COLLECTION_NAME, CONNECTION_STRING)
#print("CHUNK AND EMBED: ", chunk_and_embed_from_db_data)

#retrieve_data_from_query = query_redis_cache_then_vecotrdb_if_no_cache("test_table", "What are the AI tools of Chikara Houses?", 0.3, 2)
#print("RETRIEVE DATA for query 'how to start monetize online presence?': ", json.dumps(retrieve_data_from_query, indent=4))

#url_target_answer = "Explore AI tools by Chikara Houses in the Openai GPT store. Improve remote work and well-being with their GPTs. Visit their unique shopping rooms, too."



# Initialize states
workflow = StateGraph(MessagesState)

# each node will have one function so one job to do
workflow.add_node("get_user_input", get_user_input)
workflow.add_node("dataframe_from_query", process_query)
workflow.add_node("answer_user", answer_user)
workflow.add_node("internet_search", internet_research_user_query)
workflow.add_node("store_dataframe_to_db", store_dataframe_to_db)

#workflow.add.node("store_dataframe_to_db", store_dataframe_to_db)
#workflow.add.node("chunk_and_embed_from_db_data", custom_chunk_and_embed_to_vectordb)
#workflow.add.node("retrieve_data_from_query", query_redis_cache_then_vecotrdb_if_no_cache)
#workflow.add.node("answer_user", final_answer)

workflow.set_entry_point("get_user_input")
workflow.add_edge("get_user_input", "dataframe_from_query")
workflow.add_conditional_edges(
    "dataframe_from_query",  # Node where the decision is made
    decide_next_step,        # Function that makes the decision
    {
        "do_df_storage": "store_dataframe_to_db",  # Node to store dataframe in DB
        "do_internet_search": "internet_search",            # Node to perform internet search
    }
)
workflow.add_edge("dataframe_from_query", "internet_search")
#workflow.add_edge("dataframe_from_query", "store_dataframe_to_db")
#workflow.add_edge("store_dataframe_to_db", "chunk_and_embed_from_db_data")
#workflow.add_edge("chunk_and_embed_from_db_data", "retrieve_data_from_query")
#workflow.add_edge("retrieve_data_from_query", "answer_user")

workflow.add_edge("dataframe_from_query", "answer_user")
workflow.add_edge("internet_search", "answer_user")
workflow.add_edge("answer_user", END)

checkpointer = MemorySaver()

app = workflow.compile(checkpointer=checkpointer)




final_state = app.invoke(
  #{ "query": UserInput.user_initial_input },
  {"messages": [HumanMessage(content="initialize messages")]},
  config={"configurable": {"thread_id": 11}}
)

# Get the final message
final_message = final_state["messages"][-1].content
print("Final Message:", final_message)
# query = "I am looking for japanese furniture and want to know if chikarahouses.com have those"

# display graph drawing
graph_image = app.get_graph().draw_png()
with open("graph.png", "wb") as f:
    f.write(graph_image)





