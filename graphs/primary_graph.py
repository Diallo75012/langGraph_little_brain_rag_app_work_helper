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
from prompts.prompts import answer_user_with_report_from_retrieved_data_prompt
# Node Functions from App.py 
"""
Maybe will have to move those functions OR to a file having all node functions OR to the corresponding graph directly
"""
# Tools
from app_tools.app_tools import (
  # internet node & internet llm binded tool
  tool_search_node,
  llm_with_internet_search_tool
)
from app_utils import (
  process_query,
  is_path_or_text,
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


# Helper functions
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
  # get last message
  messages = state['messages']
  last_message = messages[-1].content
  print("Last Message: ", last_message)
  
  # save parquet file path to .var env file
  set_key(".vars.env", "PARQUET_FILE_PATH", last_message)
  load_dotenv(dotenv_path=".vars.env", override=True)

  return {"messages": [{"role": "ai", "content": last_message}]}

def internet_search_agent(state: MessagesState):
    messages = state['messages']
    print("message state -1: ", messages[-1].content, "\nmessages state -2: ", messages[-2].content)
    # print("messages from call_model func: ", messages)
    response = llm_with_internet_search_tool.invoke(messages[-1].content)
    return {"messages": [response]}

# NODE FUNCTIONS
def get_user_input(state: MessagesState):
  messages = state['messages']
  last_message = messages[-1].content
  """
   see if here we need a template to have an llm called to make the final report based on that message
  """
  return {"messages": [last_message]}

# answer user different functions
def final_answer_user(state: MessagesState):
  messages = state['messages']
  last_message = messages[-1].content
  """
    Maybe create the report and save it to a file
  """
  return {"messages": [{"role": "ai", "content": last_message}]}

def error_handler(state: MessagesState):
  messages = state['messages']
  last_message = messages[-1].content
  
  with open("./logs/conditional_edge_logs.log", "a", encoding="utf-8") as conditional:
      conditional.write(f"\n\nerror handler called: {last_message}\n\n")

  return {"messages": [{"role": "ai", "content": f"An error occured, error message: {last_message}"}]}

# CONDITIONAL EDGES FUNCTIONS
# `dataframe_from_query` conditional edge
def dataframe_from_query_conditional_edge_decision(state: MessagesState):
  messages = state['messages']
  last_message = messages[-1].content
  
  # check if path or text returned using helper function `path_or_text`
  path_or_text = is_path_or_text(last_message)
  
  with open("./logs/conditional_edge_logs.log", "a", encoding="utf-8") as conditional:
    conditional.write("\ndataframe_from_query_conditional_edge_decision:\n")
    conditional.write(f"- last message: {last_message}")
    conditional.write(f"- path_or_text: {path_or_text}\n\n")
  
  if path_or_text == "path":
    with open("./logs/conditional_edge_logs.log", "a", encoding="utf-8") as conditional:
      conditional.write(f"- returned: 'path'\n\n")
    return "inter_graph_node"
  
  if path_or_text == "text":
    with open("./logs/conditional_edge_logs.log", "a", encoding="utf-8") as conditional:
      conditional.write(f"- returned: 'text'\n\n")
    # use internet search to answer to query and answer to user
    return "internet_search_agent"
  
  else:
    with open("./logs/conditional_edge_logs.log", "a", encoding="utf-8") as conditional:
      conditional.write(f"- returned: 'error'\n\n")
    return "error_handler"

# Initialize states
workflow = StateGraph(MessagesState)

# nodes
workflow.add_node("error_handler", error_handler)
workflow.add_node("get_user_input", get_user_input)
workflow.add_node("dataframe_from_query", process_query)
workflow.add_node("internet_search_agent", internet_search_agent)
workflow.add_node("tool_search_node", tool_search_node)
workflow.add_node("inter_graph_node", inter_graph_node) # function to be created
workflow.add_node("answer_user", final_answer_user)

# edges
workflow.set_entry_point("get_user_input")
workflow.add_edge("get_user_input", "dataframe_from_query")
# `dataframe_from_query` conditional edge
workflow.add_conditional_edges(
    "dataframe_from_query",
    dataframe_from_query_conditional_edge_decision,
)
# tool edges
workflow.add_edge("internet_search_agent", "tool_search_node")
# answer user
workflow.add_edge("error_handler", "answer_user")
workflow.add_edge("tool_search_node", "answer_user")
workflow.add_edge("answer_user", END)
workflow.add_edge("inter_graph_node", END)

# compile
checkpointer = MemorySaver()
user_query_processing_stage = workflow.compile(checkpointer=checkpointer)


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





"""
"""

def primary_graph(user_query):
  print("Primary Graph")
  count = 0
  for step in user_query_processing_stage.stream(
    {"messages": [SystemMessage(content=user_query)]},
    config={"configurable": {"thread_id": int(os.getenv("THREAD_ID"))}}):
    count += 1
    if "messages" in step:
      output = beautify_output(step['messages'][-1].content)
      print(f"Step {count}: {output}")
    else:
      output = beautify_output(step)
      print(f"Step {count}: {output}")
  
  # subgraph drawing
  graph_image = user_query_processing_stage.get_graph().draw_png()
  with open("primary_subgraph.png", "wb") as f:
    f.write(graph_image)

  if ".parquet" in os.getenv("PARQUET_FILE_PATH"):
    # tell to start `embeddings` graph
    return "embeddings"
  return "error"

'''
if __name__ == "__main__":

  # to test it standalone
  primary_graph()
'''


