import os
import json
# LLM chat AI, Human, System
from langchain_core.messages import (
  AIMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage
)
from typing import Dict, List, Any, Optional, Union
# Prompts LAngchain and Custom
from langchain_core.prompts import PromptTemplate
# Tools
from app_tools.app_tools import (
  # internet node & internet llm binded tool
  tool_search_node,
  llm_with_internet_search_tool
)
# Node Functions from App.py 
"""
Maybe will have to move those functions OR to a file having all node functions OR to the corresponding graph directly
"""
from app_utils import (
  store_dataframe_to_db,
  custom_chunk_and_embed_to_vectordb
)
# LLMs
from llms.llms import (
  groq_llm_mixtral_7b,
  groq_llm_llama3_8b,
  groq_llm_llama3_8b_tool_use,
  groq_llm_llama3_70b,
  groq_llm_llama3_70b_tool_use,
  groq_llm_gemma_7b,
)
# for graph creation and management
from langgraph.checkpoint import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
# display drawing of graph
from IPython.display import Image, display
# env vars
from dotenv import load_dotenv, set_key


# load env vars
load_dotenv(dotenv_path='.env', override=False)
load_dotenv(dotenv_path=".vars.env", override=True)

# HELPER FUNCTION
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

# NODE FUNCTIONS
def inter_graph_node(state: MessagesState):
  messages = state['messages']
  last_message = messages[-1].content
  print("Inter Graph Node from Embedding Graph: ", last_message)
  
  # save reuslt to .var env file
  set_key(".vars.env", "EMBEDDING_GRAPH_RESULT", last_message)
  load_dotenv(dotenv_path=".vars.env", override=True)
  
  return {"messages": [{"role": "ai", "content": last_message}]}

def get_user_input(state: MessagesState):
  messages = state['messages']
  last_message_parquet_path = messages[-1].content
  return {"messages": [{"role": "ai", "content": last_message_parquet_path}]}

# answer user different functions
def final_answer_user(state: MessagesState):
  messages = state['messages']
  #print("Message state: ", messages)
  last_message = {"first_graph_message": messages[0].content, "second_graph_message": messages[1].content, "last_graph_message": messages[-1].content}
  return {"messages": [{"role": "ai", "content": last_message}]}

def error_handler(state: MessagesState):
  messages = state['messages']
  last_message = messages[-1].content
  
  with open("./logs/conditional_edge_logs.log", "a", encoding="utf-8") as conditional:
      conditional.write(f"\n\nerror handler called: {last_message}\n\n")

  return {"messages": [{"role": "ai", "content": f"An error occured, error message: {last_message}"}]}

# Internet search Agent Node
def internet_search_agent(state: MessagesState):
    messages = state['messages']
    print("message state -1: ", messages[-1].content, "\nmessages state -2: ", messages[-2].content)
    # print("messages from call_model func: ", messages)
    response = llm_with_internet_search_tool.invoke(messages[-1].content)
    if ("success_hash" or "success_semantic" or "success_vector_retrieved_and_cached") in messages[-1].content:
      print(f"\nAnswer retrieved, create schema for tool choice of llm, last message: {messages[-1].content}")
      response = llm_with_internet_search_tool.invoke(f"to the query {messages[-2].content} we found response in organization internal documents with content and source id: {messages[-1].content}. Analyze thouroughly the answer retrieved. Correlate the question to the answer retrieved. Find extra information by making an internet search about the content retrieved to answer the question the best.")
    # print("response from should_continue func: ", response)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

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


# `store_dataframe_to_db` conditional edge
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
    return "chunk_and_embed_from_db_data"

  elif "text" in last_message:
    with open("./logs/conditional_edge_logs.log", "a", encoding="utf-8") as conditional:
      conditional.write(f"- text: {last_message}\n\n\n\n")
    return "internet_search_agent"

  else:
    with open("./logs/conditional_edge_logs.log", "a", encoding="utf-8") as conditional:
      conditional.write(f"- error last: {last_message}\n\n")
    return "error_handler"

# `chunk_and_embed_from_db_data` conditional edge
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
    return "inter_graph_node"
  else:
    return "error_handler"


# Initialize states
workflow = StateGraph(MessagesState)

# each node will have one function so one job to do
workflow.add_node("error_handler", error_handler)
workflow.add_node("get_user_input", get_user_input)
workflow.add_node("store_dataframe_to_db", store_dataframe_to_db)
workflow.add_node("chunk_and_embed_from_db_data", custom_chunk_and_embed_to_vectordb)
workflow.add_node("inter_graph_node", inter_graph_node) # create the function to save the state and have the next graph launched or not
workflow.add_node("internet_search_agent", internet_search_agent)
workflow.add_node("tool_search_node", tool_search_node)
workflow.add_node("answer_user", final_answer_user)

# edges
workflow.set_entry_point("get_user_input")
workflow.add_edge("get_user_input", "store_dataframe_to_db")
# `store_dataframe_to_db` conditional edge
workflow.add_conditional_edges(
    "store_dataframe_to_db",
    store_dataframe_to_db_conditional_edge_decision,
)
#workflow.add_edge("store_dataframe_to_db", "chunk_and_embed_from_db_data")
#workflow.add_edge("store_dataframe_to_db", "internet_search_agent")
#workflow.add_edge("store_dataframe_to_db", "error_handler")
# `chunk_and_embed_from_db_data` conditional edge
workflow.add_conditional_edges(
    "chunk_and_embed_from_db_data",
    chunk_and_embed_from_db_data_conditional_edge_decision,
)
#workflow.add_edge("chunk_and_embed_from_db_data", "inter_graph_node")
# tools
workflow.add_edge("internet_search_agent", "tool_search_node")
workflow.add_edge("tool_search_node", "inter_graph_node")
# answer user if error and stop graph or update intermediary state to activate next graph
workflow.add_edge("error_handler", "answer_user")
workflow.add_edge("answer_user", END)
workflow.add_edge("inter_graph_node", END)

checkpointer = MemorySaver()
embedding_graph = workflow.compile(checkpointer=checkpointer)


'''
# using: INVOKE
final_state = embedding_subgraph.invoke(
  #{ "query": UserInput.user_initial_input },
  {"messages": [HumanMessage(content="initialize messages")]},
  config={"configurable": {"thread_id": 11}}
)

# Get the final message
final_message = final_state["messages"][-1].content
print("Final Message:", final_message)
# query = "I am looking for japanese furniture and want to know if chikarahouses.com have those"
'''

# using STREAM
# we can maybe get the uder input first and then inject it as first message of the state: `{"messages": [HumanMessage(content=user_input)]}`

def embedding_subgraph(parquet_file_path):
  print("Embedding Graph")
  count = 0
  for step in embedding_graph.stream(
    {"messages": [SystemMessage(content=parquet_file_path)]},
    config={"configurable": {"thread_id": int(os.getenv("THREAD_ID"))}}):
    count += 1
    if "messages" in step:
      print(f"Step {count}: {beautify_output(step['messages'][-1].content)}")
    else:
      print(f"Step {count}: {beautify_output(step)}")


  # display graph drawing
  graph_image = embedding_graph.get_graph().draw_png()
  with open("embedding_subgraph.png", "wb") as f:
    f.write(graph_image)
  
  if "success" in os.getenv("EMBEDDING_GRAPH_RESULT"):
    # tell to start `retrieval` graph
    return "retrieval"
  return "error"

'''
if __name__ == "__main__":
  load_dotenv(dotenv_path=".vars.env", override=True)
  parquet_file_path = os.getenv("PARQUET_FILE_PATH")
  embedding_subgraph(parquet_file_path)
'''



