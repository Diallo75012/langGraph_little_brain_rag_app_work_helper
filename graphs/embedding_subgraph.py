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
from app import (
  process_query,
  store_dataframe_to_db,
  custom_chunk_and_embed_to_vectordb
)
# LLMs
from llms import (
  groq_llm_mixtral_7b,
  groq_llm_llama3_8b,
  groq_llm_llama3_8b_tool_use,
  groq_llm_llama3_70b,
  groq_llm_llama3_70b_tool_use,
  groq_llm_gemma_7b,
)
# Custom States
from app_states.app_graph_states import GraphStatePersistFlow
# for graph creation and management
from langgraph.checkpoint import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
# display drawing of graph
from IPython.display import Image, display
# env vars
from dotenv import load_dotenv


# load env vars
load_dotenv(dotenv_path='.env', override=False)
laod_dotenv(dotenv_path=",vars", override=True)

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
def get_user_input(state: MessagesState):
  user_input =  input("Do you need any help? any PDF doc or webpage to analyze? ").strip()

  #messages = state['messages']
  
  return {"messages": [user_input]}

# answer user different functions
def final_answer_user(state: MessagesState):
  messages = state['messages']
  #print("Message state: ", messages)
  last_message = {"first_graph_message": messages[0].content, "second_graph_message": messages[1].content, "last_graph_message": messages[-1].content}
  return {"messages": [{"role": "ai", "content": last_message}]}

def answer_user_with_report(state: MessagesState):
  messages = state['messages']
  #print("Message state: ", messages)

  # check if tool have been called or not. if it haven't been called the -2 will be the final answer else we keep it the same -1 is the final answer
  if messages[-2].tool_calls == []:
    try:
      last_message = messages[-1].content
      response = groq_llm_mixtral_7b.invoke("I need a detailed report about. Put your report answer between markdown tags ```markdown ```: {last_message}")
      formatted_answer = response.content.split("```")[1].strip("markdown").strip()
    except IndexError as e:
      formatted_answer = response.content
      print(f"We found an error. answer returned by llm withotu markdown tags: {e}")
    return {"messages": [{"role": "ai", "content": formatted_answer}]}
  # otherwise we return -1 message as it is the tool answer
  try:
    last_message = messages[-2].content
    response = groq_llm_mixtral_7b.invoke("I need a detailed report about. Put your report answer between markdown tags ```markdown ```: {last_message}")
    formatted_answer = response.content.split("```")[1].strip("markdown").strip()
  except IndexError as e:
    formatted_answer = response.content
    print(f"We found an error. answer returned by llm withotu markdown tags: {e}")
  #formatted_answer_structured_output = response
  return {"messages": [{"role": "ai", "content": formatted_answer}]}

def answer_user_with_report_from_retrieved_data(state: MessagesState):
  messages = state['messages']
  #print("Message state: ", messages)
  # # -4 user input, -3 data retrieved, -2 schema internet tool, -1 internet search result
  internet_search_result = messages[-1].content
  info_data_retrieved = messages[-3].content
  question = messages[-4].content
  prompt = answer_user_with_report_from_retrieved_data_prompt["human"]["template"]
  prompt_human_with_input_vars_filled = eval(f'f"""{prompt}"""')
  print(f"\nprompt_human_with_input_vars_filled: {prompt_human_with_input_vars_filled}\n")
  system_message = SystemMessage(content=prompt_human_with_input_vars_filled)
  human_message = HumanMessage(content=prompt_human_with_input_vars_filled)
  messages = [system_message, human_message]
  response = groq_llm_mixtral_7b.invoke(messages)
  #formatted_answer = response.content.split("```")[1].strip("markdown").strip()

  # couldn't get llm to answer in markdown tags???? so just getting content saved to states
  return {"messages": [{"role": "ai", "content": response.content}]}


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
    return "store_dataframe_to_db"
  
  if path_or_text == "text":
    with open("./logs/conditional_edge_logs.log", "a", encoding="utf-8") as conditional:
      conditional.write(f"- returned: 'text'\n\n")
    #return "internet_search_agent"
    return "answer_user"
  
  else:
    with open("./logs/conditional_edge_logs.log", "a", encoding="utf-8") as conditional:
      conditional.write(f"- returned: 'error'\n\n")
    return "error_handler"

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
    #return "answer_user"

  elif "text" in last_message:
    with open("./logs/conditional_edge_logs.log", "a", encoding="utf-8") as conditional:
      conditional.write(f"- text: {last_message}\n\n\n\n")
    return "answer_user"

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
    return "answer_user"
  else:
    return "error_handler"


# Initialize states
workflow = StateGraph(MessagesState)

# each node will have one function so one job to do
workflow.add_node("error_handler", error_handler)
workflow.add_node("get_user_input", get_user_input)
workflow.add_node("dataframe_from_query", process_query)
workflow.add_node("store_dataframe_to_db", store_dataframe_to_db)
workflow.add_node("chunk_and_embed_from_db_data", custom_chunk_and_embed_to_vectordb)
workflow.add_node("internet_search_agent", internet_search_agent)
workflow.add_node("tool_search_node", tool_search_node)
# special andwser report fetching user query, database retrieved data and internet search result
workflow.add_node("answer_user_with_report_from_retrieved_data", answer_user_with_report_from_retrieved_data) # -4 user input, -3 data retrieved, -2 schema internet tool, -1 internet search result
workflow.add_node("answer_user_with_report", answer_user_with_report)
workflow.add_node("answer_user", final_answer_user)

workflow.set_entry_point("get_user_input")
workflow.add_edge("get_user_input", "dataframe_from_query")

# `dataframe_from_query` conditional edge
workflow.add_conditional_edges(
    "dataframe_from_query",
    dataframe_from_query_conditional_edge_decision,
)

workflow.add_edge("dataframe_from_query", "store_dataframe_to_db")
workflow.add_edge("dataframe_from_query", "answer_user")
workflow.add_edge("dataframe_from_query", "error_handler")

# `store_dataframe_to_db` conditional edge
workflow.add_conditional_edges(
    "store_dataframe_to_db",
    store_dataframe_to_db_conditional_edge_decision,
)

workflow.add_edge("store_dataframe_to_db", "chunk_and_embed_from_db_data")
workflow.add_edge("store_dataframe_to_db", "answer_user")
workflow.add_edge("store_dataframe_to_db", "error_handler")

# `chunk_and_embed_from_db_data` conditional edge
workflow.add_conditional_edges(
    "chunk_and_embed_from_db_data",
    chunk_and_embed_from_db_data_conditional_edge_decision,
)
workflow.add_edge("chunk_and_embed_from_db_data", "answer_user")
workflow.add_edge("dataframe_from_query", "store_dataframe_to_db")

# TOOLS EDGES
workflow.add_edge("internet_search_agent", "answer_user")
workflow.add_edge("internet_search_agent", "tool_search_node")
workflow.add_edge("tool_search_node", "answer_user_with_report")

workflow.add_edge("error_handler", "answer_user")
workflow.add_edge("answer_user", END)
workflow.add_edge("answer_user_with_report", END)


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

# using STREAM
# we can maybe get the uder input first and then inject it as first message of the state: `{"messages": [HumanMessage(content=user_input)]}`

def embedding_subgraph():
  count = 0
  for step in app.stream(
    {"messages": [SystemMessage(content="Graph Embedding Webpage or PDF")]},
    config={"configurable": {"thread_id": int(os.getenv("THREAD_ID"))}}):
    count += 1
    if "messages" in step:
      output = beautify_output(step['messages'][-1].content
      # update custom state graph messages list
      GraphStatePersistFlow.subgraph_messages.append(str(output))
      print(f"Step {count}: {beautify_output(step['messages'][-1].content)}")
    else:
      output = beautify_output(step)
      # update custom state graph messages list
      GraphStatePersistFlow.subgraph_messages.append(str(output))
      print(f"Step {count}: {beautify_output(step)}")


  # display graph drawing
  graph_image = app.get_graph().draw_png()
  with open("embedding_subgraph.png", "wb") as f:
    f.write(graph_image)

  return "subgraph embedding job done"

if __name__ == "__main__":

  embedding_subgraph()




