import os
import time

from dotenv import load_dotenv

from langchain_groq import ChatGroq
# from app import prompt_creation, chat_prompt_creation, dict_to_tuple
from langchain_core.output_parsers import JsonOutputParser
#from app 
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import Tool, tool

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


# NODE FUNCTIONS
def get_user_input(state: MessagesState):
  user_input =  input("Do you need any help? any PDF doc or webpage to analyze? ")

  #messages = state['messages']
  
  return {"messages": [user_input]}

def answer_user(state: MessagesState):
  messages = state['messages']
  #print("Message state: ", messages)
  last_message = messages[-1].content
  return {"messages": [{"role": "ai", "content": last_message}]}

def error_handler(state: MessagesState):
  messages = state['messages']
  last_message = messages[-1].content
  
  with open("./logs/conditional_edge_logs.log", "a", encoding="utf-8") as conditional:
      conditional.write(f"\n\nerror handler called: {last_message}\n\n")

  return {"messages": [{"role": "ai", "content": f"An error occured, error message: {last_message}"}]}


# CONDITIONAL EDGES FONCTIONS
# check if message is path or not, specially for the process query returned dataframe path or text state message. will be used in the conditional edge
def is_path_or_text(input_string: str) -> str:
    """
    Determines if the input string is a valid file path or just a text string.

    Args:
    input_string (str): The string to be checked.

    Returns:
    str: 'path' if the input is a valid file path, 'text' otherwise.
    """
    # Normalize the path to handle different OS path formats
    normalized_path = os.path.normpath(input_string)
    
    # Check if the normalized path exists or has a valid directory structure
    if os.path.exists(normalized_path) or os.path.isdir(os.path.dirname(normalized_path)):
        return "path"
    else:
        return "text"

# `dataframe_from_query` conditional edge
'''
    {
        "path": "store_dataframe_to_db",
        "text": "answer_user", # maybe replace by internet research first so use agent with tool
        "error": "error_handler"
    }
'''
def dataframe_from_query_conditional_edge_decision(state: MessagesState):
  messages = state['messages']
  last_message = messages[-1].content
  path_or_text = is_path_or_text(last_message)
  with open("./logs/conditional_edge_logs.log", "a", encoding="utf-8") as conditional:
    conditional.write("\ndataframe_from_query_conditional_edge_decision:\n")
    conditional.write(f"- last message: {last_message}")
    conditional.write(f"- path_or_text: {path_or_text}\n\n")
  
  if path_or_text == "path":
    with open("./logs/conditional_edge_logs.log", "a", encoding="utf-8") as conditional:
      conditional.write(f"- returned: 'path'\n\n")
    return "store_dataframe_to_db"
  
  if path_or_text == "text":
    with open("./logs/conditional_edge_logs.log", "a", encoding="utf-8") as conditional:
      conditional.write(f"- returned: 'text'\n\n")
    return "answer_user"
  
  else:
    with open("./logs/conditional_edge_logs.log", "a", encoding="utf-8") as conditional:
      conditional.write(f"- returned: 'error'\n\n")
    return "error_handler"

# `store_dataframe_to_db` conditional edge
'''
    {
        "next_node": "chunk_and_embed_from_db_data",
        "answer_query": "answer_user",  # maybe replace by internet research first so use agent with tool
        "error": "error_handler"
    }
'''
def store_dataframe_to_db_conditional_edge_decision(state: MessagesState):
  messages = state['messages']
  last_message = messages[-1].content
  with open("./logs/conditional_edge_logs.log", "a", encoding="utf-8") as conditional:
    conditional.write("\nstore_dataframe_to_db_conditional_edge_decision:\n")
    conditional.write(f"- last message: {last_message}")

  if "error" in last_message:
    with open("./logs/conditional_edge_logs.log", "a", encoding="utf-8") as conditional:
      conditional.write(f"- error: {last_message}\n\n")
    return "error_handler"

  elif "success" in last_message:
    with open("./logs/conditional_edge_logs.log", "a", encoding="utf-8") as conditional:
      conditional.write(f"- success: {last_message}\n\n")
    #return "chunk_and_embed_from_db_data"
    return "answer_user"

  elif "text" in last_message:
    with open("./logs/conditional_edge_logs.log", "a", encoding="utf-8") as conditional:
      conditional.write(f"- text: {last_message}\n\n\n\n")
    return "answer_user"

  else:
    with open("./logs/conditional_edge_logs.log", "a", encoding="utf-8") as conditional:
      conditional.write(f"- error last: {last_message}\n\n")
    return "error_handler"

# `chunk_and_embed_from_db_data` conditional edge
'''
    {
        "success": "answer_user",  # maybe replace by internet research first so use agent with tool
        "error": "error_handler"
    }
'''
def chunk_and_embed_from_db_data_conditional_edge_decision(state: MessagesState):
  messages = state['messages']
  last_message = messages[-1].content
  with open("./logs/conditional_edge_logs.log", "a", encoding="utf-8") as conditional:
    conditional.write("\nchunk_and_embed_from_db_data_conditional_edge_decision:\n")
    conditional.write(f"- last message: {last_message}")

  if "error" in last_message:
    with open("./logs/conditional_edge_logs.log", "a", encoding="utf-8") as conditional:
      conditional.write(f"- error: {last_message}\n\n")
    return "error_handler"

  elif "success" in last_message:
    with open("./logs/conditional_edge_logs.log", "a", encoding="utf-8") as conditional:
      conditional.write(f"- succes: {last_message}\n\n")
    return "answer_user"
  else:
    return "error_handler"
   
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
workflow.add_node("error_handler", error_handler) # will be used to end the graph returning the app system error messages
workflow.add_node("get_user_input", get_user_input)
workflow.add_node("dataframe_from_query", process_query)
workflow.add_node("store_dataframe_to_db", store_dataframe_to_db)
#workflow.add_node("chunk_and_embed_from_db_data", custom_chunk_and_embed_to_vectordb)
workflow.add_node("answer_user", answer_user)
#workflow.add_node("internet_search", internet_research_user_query)


workflow.set_entry_point("get_user_input")
workflow.add_edge("get_user_input", "dataframe_from_query")

# `dataframe_from_query` conditional edge
workflow.add_conditional_edges(
    "dataframe_from_query",
    dataframe_from_query_conditional_edge_decision,
)
#workflow.add_edge("dataframe_from_query", "store_dataframe_to_db")
#workflow.add_edge("dataframe_from_query", "answer_user")
#workflow.add_edge("dataframe_from_query", "error_handler")

# `store_dataframe_to_db` conditional edge
workflow.add_conditional_edges(
    "store_dataframe_to_db",
    store_dataframe_to_db_conditional_edge_decision,
)
#workflow.add_edge("store_dataframe_to_db", "chunk_and_embed_from_db_data")
#workflow.add_edge("store_dataframe_to_db", "answer_user")
#workflow.add_edge("store_dataframe_to_db", "error_handler")

# `chunk_and_embed_from_db_data` conditional edge
#workflow.add_conditional_edges(
#    "chunk_and_embed_from_db_data",
#    chunk_and_embed_from_db_data_conditional_edge_decision,
#)
#workflow.add_edge("chunk_and_embed_from_db_data", "answer_user")


#workflow.add_edge("dataframe_from_query", "store_dataframe_to_db")
#workflow.add_edge("store_dataframe_to_db", "chunk_and_embed_from_db_data")
#workflow.add_edge("chunk_and_embed_from_db_data", "retrieve_data_from_query")
#workflow.add_edge("retrieve_data_from_query", "answer_user")


#workflow.add_edge("internet_search", "answer_user")
workflow.add_edge("error_handler", "answer_user")
workflow.add_edge("answer_user", END)

checkpointer = MemorySaver()

app = workflow.compile(checkpointer=checkpointer)



'''
# using: INVOKE
final_state = app.invoke(
  #{ "query": UserInput.user_initial_input },
  {"messages": [HumanMessage(content="initialize messages")]},
  config={"configurable": {"thread_id": 11}}
)

# Get the final message
final_message = final_state["messages"][-1].content
print("Final Message:", final_message)
# query = "I am looking for japanese furniture and want to know if chikarahouses.com have those"
'''

def message_to_dict(message):
    if isinstance(message, (AIMessage, HumanMessage, SystemMessage, ToolMessage)):
        return {
            "content": message.content,
            "additional_kwargs": message.additional_kwargs,
            "response_metadata": message.response_metadata if hasattr(message, 'response_metadata') else None,
            "tool_calls": message.tool_calls if hasattr(message, 'tool_calls') else None,
            "usage_metadata": message.usage_metadata if hasattr(message, 'usage_metadata') else None,
            "id": message.id,
            "role": getattr(message, 'role', None),
        }
    return message

def convert_to_serializable(data):
    if isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    elif isinstance(data, dict):
        return {k: convert_to_serializable(v) for k, v in data.items()}
    elif isinstance(data, (AIMessage, HumanMessage, SystemMessage, ToolMessage)):
        return message_to_dict(data)
    return data

def beautify_output(data):
    serializable_data = convert_to_serializable(data)
    return json.dumps(serializable_data, indent=4)

# using STREAM
# we can maybe get the uder input first and then inject it as first message of the state: `{"messages": [HumanMessage(content=user_input)]}`
count = 0
for step in app.stream(
    {"messages": [SystemMessage(content="Graph Embedding Webpage or PDF")]},
    config={"configurable": {"thread_id": 42}}):
    count += 1
    if "messages" in step:
        print(f"Step {count}: {beautify_output(step['messages'][-1].content)}")
    else:
        print(f"Step {count}: {beautify_output(step)}")

# display graph drawing
graph_image = app.get_graph().draw_png()
with open("graph.png", "wb") as f:
    f.write(graph_image)





'''
internet_search_tool = DuckDuckGoSearchRun()
tool_internet = Tool(
    name="duckduckgo_search",
    description="Search DuckDuckGO for recent results.",
    func=internet_search_tool.run,
)

tool_node = ToolNode([tool_internet])

class InternetSearchStructuredOutput(BaseModel):
    """Internet research Output Response"""
    internet_search_answer: str = Field(description="The result of the internet research about user query")

structured_internet_research_llm_tool = llm_with_internet_search_tool.with_structured_output(InternetSearchStructuredOutput, include_raw=True)

class InternetSearch(BaseModel):
  query: str = Field(description="user query to be searched through the internet")
  state: MessagesState = Field(description="the state storing all messages.")

def internet_search(query: str = "latest restaurant opened in Azabu Juuban?", state: MessagesState = MessagesState()) -> str:
  """
    This tool will search in the internet about the answer to the user query
    
    Parameter: 
    query str : 'User query to be searched through the internet'
    
    Returns: 
    str message about report output quality status: 'ok' for good reports and another message different from 'ok' for report that needs to be redone.
  """
  response = internet_search_tool.run(query)
  return {"messages": [response]}

internet_research_tool = StructuredTool.from_function(
  func=internet_search,
  name="internet research tool",
  description=  """
    This tool will search in the internet about the answer to the user query
  """,
  args_schema=InternetSearch,
  # return_direct=True, # returns tool output only if no TollException raised
  # coroutine= ... <- you can specify an async method if desired as well
  # callback==callback_function # will run after task is completed
)

def internet_research_user_query(state: MessagesState):
    query = state["messages"][-1].content  # Extract the last user query

    # Create the prompt for the LLM
    system_message = SystemMessage(content="You are an expert search engine and use the tool available to answer to user query.")
    human_message = HumanMessage(content=f"Find the most relevant information about: {query}")

    # Use the tool with the constructed prompt
    agent_executor = AgentExecutor.from_agent_and_tools(agent=llm_with_internet_search_tool, tools=[tool_internet])

    try:
        # Pass the list of BaseMessages directly
        result_agent = agent_executor.invoke(human_message)
        print("Result Agent:", result_agent)
        return {"messages": [{"role": "assistant", "content": result_agent}]}
    except Exception as e:
        print("Error during invocation:", e)
        return {"messages": [{"role": "assistant", "content": "An error occurred during the internet search."}]}
'''
@tool
def get_proverb(query: str, state: MessagesState = MessagesState()):
  """Will transform user query into a funny proverb"""
  system_message = SystemMessage(content="You are an expert in creating funny short proverbs from any query.")
  human_message = HumanMessage(content=f"Create a funny proverb please: {query}")
  response = groq_llm_mixtral_7b.invoke([system_message, human_message])
  return {"messages": [response]}

@tool
def find_link_story(query: str, state: MessagesState = MessagesState()):
  """Will find a link between user query and the planet Mars"""
  system_message = SystemMessage(content="You are an expert in finding links between Mars planet and any query.")
  human_message = HumanMessage(content=f"{query}")
  response = groq_llm_mixtral_7b.invoke([system_message, human_message])
  return {"messages": [response]}


import os
from langgraph.graph import StateGraph, MessagesState, END
from langchain.agents import AgentExecutor
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import Tool, tool
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import StructuredTool

# Load environment variables
load_dotenv()

# Initialize LLM
groq_llm_mixtral_7b = ChatGroq(
    temperature=float(os.getenv("GROQ_TEMPERATURE")),
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-groq-70b-8192-tool-use-preview",
    max_tokens=int(os.getenv("GROQ_MAX_TOKEN")),
)

# Define the search tool
internet_search_tool = DuckDuckGoSearchRun()
tool_internet = Tool(
    name="duckduckgo_search",
    description="Search DuckDuckGO for recent results.",
    func=internet_search_tool.run,
)

@tool
def search(query: str, state: MessagesState = MessagesState()):
    """Call to surf the web."""
    search_results = internet_search_tool.run(query)
    return {"messages": [search_results]}

tool_search_node = ToolNode([get_proverb, find_link_story])

# Bind the tool to the LLM
llm_with_internet_search_tool = groq_llm_mixtral_7b.bind_tools([get_proverb, find_link_story])



# Define helper node functions
def get_user_input(state: MessagesState):
  user_input =  input("Do you need any help? any PDF doc or webpage to analyze? ")

  #messages = state['messages']
  
  return {"messages": [user_input]}


def answer_user(state: MessagesState):
  messages = state['messages']
  #print("Message state: ", messages)
  last_message = messages[-1].content
  return {"messages": [{"role": "ai", "content": last_message}]}

def call_model(state: MessagesState):
    messages = state['messages']

    # print("messages from call_model func: ", messages)
    response = llm_with_internet_search_tool.invoke(messages[-1].content)
    # print("response from should_continue func: ", response)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# please search online about the latest restaurant opened in Azabu Juuban?


# Example of how to use this function
count = 0
for step in app.stream(
    {"messages": [HumanMessage(content="message initialization")]},
    config={"configurable": {"thread_id": 42}}):
    count += 1
    if "messages" in step:
        print(f"Step {count}: {beautify_output(step['messages'][-1].content)}")
    else:
        print(f"Step {count}: {beautify_output(step)}")


'''
# example of execution INVOKE()
final_state = app.invoke(
    {"messages": [HumanMessage(content="What is the biggest city in Asia?")]},
    config={"configurable": {"thread_id": 11}}
)

print("Final State:", final_state)


count = 0
for step in app.stream(
    {"messages": [HumanMessage(content="message initialization")]},
    config={"configurable": {"thread_id": 42}}):
    count += 1
    if "messages" in step:
      print(f"Step {count}: {step['messages'][-1].content}")
    else:
      print(f"Step {count}: {step}")
'''

# agent tool print png
# display graph drawing
graph_image = app.get_graph().draw_png()
with open("agent_tool_call_visualization.png", "wb") as f:
    f.write(graph_image)

# test the tool independently
#search_results = internet_search_tool.run("latest restaurant opened in Azabu Juuban")
#print("Direct search results: ", search_results)


#response = structured_internet_research_llm_tool.invoke([SystemMessage(content="You are an AI with access to various tools. Use the best tool to respond to the user's query."), HumanMessage(content="please search online about the latest restaurant opened in Azabu Juuban?")])
#print("Final Response: ", {"messages": [{"role": "assistant", "content": response}]})




'''
{
  'messages': [
    HumanMessage(content='what is the weather in sf', id='bd9c0deb-9d88-425e-adf8-62b016e68444'), 
    AIMessage(content='',
      additional_kwargs={
        'tool_calls': [
          {
            'id': 'call_q6av', 
            'function': {
              'arguments': '{"query":"current weather in San Francisco"}', 'name': 'search'
            }, 
            'type': 'function'
          }
        ]
      },
      response_metadata={
        'token_usage': {
          'completion_tokens': 46, # output
          'prompt_tokens': 903, # input
          'total_tokens': 949,
          'completion_time': 0.143823147,
          'prompt_time': 0.058552886,
          'queue_time': 0.0043955680000000025,
          'total_time': 0.202376033
        },
        'model_name': 'llama3-70b-8192',
        'system_fingerprint': 'fp_753a4aecf6',
        'finish_reason': 'tool_calls',
        'logprobs': None
      },
      id='run-82ff5790-33d4-4d0a-9e62-0bf4ddc34e7a-0',
      tool_calls=[
        {'name': 'search',
        'args': {
          'query': 'current weather in San Francisco'
        },
        'id': 'call_q6av', 'type': 'tool_call'
        }
      ],
      usage_metadata={
        'input_tokens': 903,
        'output_tokens': 46,
        'total_tokens': 949
      }
    ),
    ToolMessage(
      content="It's 60 degrees and foggy.",
      name='search',
      id='7d209e17-9612-4439-8e0f-6dbfca0a583b',
      tool_call_id='call_q6av'
    ),
    AIMessage(
      content='The weather in San Francisco is 60 degrees and foggy.',
      response_metadata={
        'token_usage': {
          'completion_tokens': 14,
          'prompt_tokens': 977,
          'total_tokens': 991,
          'completion_time': 0.043690019,
          'prompt_time': 0.054365208,
          'queue_time': 0.004546837000000005,
          'total_time': 0.098055227
        },
        'model_name': 'llama3-70b-8192',
        'system_fingerprint': 'fp_753a4aecf6',
        'finish_reason': 'stop',
        'logprobs': None
      },
      id='run-85b6ee46-684f-4cfa-8ea9-55000b3d8da1-0',
      usage_metadata={
        'input_tokens': 977,
        'output_tokens': 14,
        'total_tokens': 991
      }
    )
  ]
}
'''
















