import os
import time
import json
# For str to dict
import ast
import re
# for web request
import requests
# for DB
import psycopg2
from psycopg2 import sql
# for subprocesses
import subprocess
# for dataframe
import pandas as pd
from uuid import uuid4
# for typing func parameters and outputs and states
from typing import Literal, TypedDict, Dict, List, Tuple, Any, Optional
from pydantic import BaseModel
# for llm call with func or tool and prompts formatting
from langchain_groq import ChatGroq
# one is @tool decorator and the other Tool class
from langchain_core.tools import tool, Tool
from langchain_community.tools import (
  # Run vs Results: Results have more information
  DuckDuckGoSearchRun,
  DuckDuckGoSearchResults
) 
from langchain_core.messages import (
  AIMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage
)
from langchain.prompts import (
  PromptTemplate,
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate,
  AIMessagePromptTemplate
)
from app_tools.app_tools import (
  #import tool_internet, internet_search_tool, internet_search_query
  # internet node & internet llm binded tool
  tool_search_node,
  llm_with_internet_search_tool
)
from langchain_core.output_parsers import JsonOutputParser
# for graph creation and management
from langgraph.checkpoint import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
# langchain agent
from langchain.agents import AgentExecutor
from langgraph.prebuilt.tool_executor import ToolExecutor
# for env. vars
from dotenv import load_dotenv

#### MODULES #####
# PROMPTS
from prompts.prompts import (
  detect_content_type_prompt,
  summarize_text_prompt,
  generate_title_prompt,
  generate_from_empty_prompt
)
# USER QUERY ANALYSIS AND TRANSFORMATION
from lib_helpers.query_analyzer_module import detect_content_type # returns `str`
# INITIAL DOC PARSER, STORER IN POSTGRESL DB TABLE
from lib_helpers.pdf_parser import pdf_to_sections # returns `list`
from lib_helpers.webpage_parser import scrape_website # returns `Dict[str, Any]`
# CUSTOM CHUNKING
from lib_helpers.chunking_module import (
  # returns `List[Dict[str, Any]]`
  create_chunks_from_webpage_data,
  # returns `List[List[Dict[str,Any]]]`
  create_chunks_from_db_data
)
# REDIS CACHE RETRIEVER
from lib_helpers.query_matching import handle_query_by_calling_cache_then_vectordb_if_fail # returns `Dict[str, Any]`
# DB AND VECTORDB EMBEDDINGS AND RETRIEVAL
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
  clear_cache,
  delete_table,
  cache_document,
  redis_client,
  create_table_if_not_exists,
  connect_db,
  embeddings
)
from lib_helpers.query_matching import handle_query_by_calling_cache_then_vectordb_if_fail
# STATES Persistence between graphs
from app_states.app_graph_states import GraphStatePersistFlow
# DOCKER REMOTE CODE EXECUTION
# eg.: print(run_script_in_docker("test_dockerfile", "./app.py"))
from docker_agent.execution_of_agent_in_docker_script import run_script_in_docker # returns `Tuple[str, str]` stdout,stderr
# to run next graphs
from graphs.graphs import subgraph


# load env vars
load_dotenv(dotenv_path='.env', override=False)

groq_llm_mixtral_7b = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_MIXTRAL_7B"),
max_tokens=int(os.getenv("GROQ_MAX_TOKEN")),)
groq_llm_llama3_8b = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_LLAMA3_8B"), max_tokens=int(os.getenv("GROQ_MAX_TOKEN")),)
groq_llm_llama3_8b_tool_use = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_LLAMA3_8B_TOOL_USE"), max_tokens=int(os.getenv("GROQ_MAX_TOKEN")),)
groq_llm_llama3_70b = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_LLAMA3_70B"), max_tokens=int(os.getenv("GROQ_MAX_TOKEN")),)
groq_llm_llama3_70b_tool_use = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_LLAMA3_70B_TOOL_USE"), max_tokens=int(os.getenv("GROQ_MAX_TOKEN")),)
groq_llm_gemma_7b = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_GEMMA_7B"), max_tokens=int(os.getenv("GROQ_MAX_TOKEN")),)


collection_name = COLLECTION_NAME
connection_string = CONNECTION_STRING

# Connect to the PostgreSQL database (all imported from `embedding_and_retrieval`)
conn = connect_db()
create_table_if_not_exists()


# initialized vars:
'''
UserInput.user_initial_input = input("How can i help you? ")
FuncInputParams.maximum_content_length = 200
FuncInputParams.maximum_title_length = 30
FuncInputParams.chunk_size_df = 250
FuncInputParams.detect_prompt_template = detect_content_type_prompt
FuncInputParams.text_summary_prompt = summarize_text_prompt
FuncInputParams.title_summary_prompt = generate_title_prompt
FuncOutputs.df_final = pd.DataFrame()
'''

# Define the tools for the agent to use
# this is a dummy tool just for the sake of testing langgraph
@tool
def search(query: str):
    """Call to surf the web."""
    # This is a placeholder, but don't tell the LLM that...
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."


tools = [search]
# print("Tools: ", tools)

tool_node = ToolNode(tools)
# print("tool_node: ", tool_node)

model = groq_llm_llama3_70b.bind_tools(tools)
# print("Model with bind_tools: ", model)


# Define the function that determines whether to continue or not
def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    # print("messages from should_continue func: ", messages)
    last_message = messages[-1]
    # print("last message from should_continue : ", last_message)
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        # print("Tool called!")
        return "tools"
    # Otherwise, we stop (reply to the user)func
    # print("Tool not called returning answer to user.")
    return END


# Define the function that calls the model
def call_model(state: MessagesState):
    messages = state['messages']
    # print("messages from call_model func: ", messages)
    response = model.invoke(messages)
    # print("response from should_continue func: ", response)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define a new graph
workflow = StateGraph(MessagesState)
# print("workflow: ", workflow)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
# print("Workflow add node 'agent': ", workflow)
workflow.add_node("tools", tool_node)
# print("Workflow add node 'tools': ", workflow)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent")
# print("Workflow set entry point 'agent': ", workflow)

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)
# print("Workflow add conditional edge 'agent' -> should_continue func: ", workflow)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("tools", "agent")
# print("Workflow add edge 'tools' -> 'agent': ", workflow)

# Initialize memory to persist state between graph runs
checkpointer = MemorySaver()
# print("MEmory checkpointer: ", checkpointer)

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable.
# Note that we're (optionally) passing the memory when compiling the graph
app = workflow.compile(checkpointer=checkpointer)
# print("App compiled with checkpointer: ", app)

# Use the Runnable
final_state = app.invoke(
    {"messages": [HumanMessage(content="what is the weather in sf")]},
    config={"configurable": {"thread_id": 42}}
)
# print("Final State = answer: ", final_state)

final_state["messages"][-1].content
# print("Final state last message content: ", final_state["messages"][-1].content)


#########################################################################################
"""
### LOGIC FOR THE MOMENT

# workflow:

  `Use agent prompting and conditional edge of langgraph to direct the llm to use internet search if didn't found information or if needed extra information because of info poorness`
 - better use lots of mini tools than one huge and let the llm check. We will just prompting the agent to tell it what is the workflow. So revise functions here to be decomposed in mini tools
   `list of tools to create to be planned and helps for nodes creation as well so it is a mix of tools and nodes definition titles for the moment: [internet_search_tool, redis_cache_search_tool, embedding_vectordb_search_tool, user_query_analyzer_rephrasing_tool, node_use_tool_or_not_if_not_answer_query, save_initial_query_an_end_outcome_to_key_value_db_and_create_redis_cache_with_long_term_ttl, node_judge_answer_for_new_iteration_or_not]`
 - then build all those mini tools
 - Have a function node that will just check the date and if 24h have passed it will reset the column retrieved in the database to False (for the moment we work with a TTL of 24h can be extended if needed so put the TTL as a function parameter that will be the same as in Redis). `from embedding_and_retrieval import clear_cache
"""
### HELPERS
# function to beautify output for an ease of human creditizens reading
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

# to format prompts
def dict_to_tuple(d: Dict[str, str]) -> Tuple[Tuple[str, str], ...]:
    """
    Transforms a dictionary into a tuple of key-value pairs.

    Args:
    d (Dict[str, str]): The dictionary to transform.

    Returns:
    Tuple[Tuple[str, str], ...]: A tuple containing the dictionary's key-value pairs.
    """
    return tuple(d.items())

# function needed to get the STR dict representation returned by the query analyzer and be able to fetch values as a dict
def string_to_dict(string: str) -> Dict[str, Any]:
    """
    Converts a string representation of a dictionary to an actual dictionary.
    
    Args:
    string (str): The string representation of a dictionary.
    
    Returns:
    Dict[str, Any]: The corresponding dictionary.
    """
    try:
        # Safely evaluate the string as a Python expression
        dictionary = ast.literal_eval(string)
        if isinstance(dictionary, dict):
            return dictionary
        else:
            raise ValueError("The provided string does not represent a dictionary.")
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"Error converting string to dictionary: {e}")

# if needed this will tell if the string is url or pdf
def is_url_or_pdf(input_string: str) -> str:
    """
    Checks if the given string is a URL or a PDF document.

    Args:
    input_string (str): The input string to check.

    Returns:
    str: "url" if the string is a URL, "pdf" if it's a PDF document, and "none" if it's neither.
    """
    try:
        # Regular expression pattern for URLs
        url_pattern = re.compile(
            r'^(https?:\/\/)?'                # http:// or https:// (optional)
            r'([a-z0-9]+[.-])*[a-z0-9]+\.[a-z]{2,6}'  # domain name
            r'(:[0-9]{1,5})?'                 # port (optional)
            r'(\/.*)?$',                      # path (optional)
            re.IGNORECASE
        )

        # Check if it's a URL
        #if input_string.lower().startswith("https://") or input_string.lower().startswith("http://"):
        if url_pattern.match(input_string):
            print("It was an URL!")
            return "url"

        # Check if it's a .pdf document URL or file path
        if input_string.lower().endswith(".pdf"):
            print("It was a PDF!")
            return "pdf"

        # Neither URL nor PDF
        return "none"

    except Exception as e:
        # Catch any unexpected errors and return "none"
        print(f"An error occurred: {e}")
        return "none"

# Function to summarize text
def summarize_text(llm, row, maximum, prompt):
    # Add length constraints directly to the prompt
    prompt_text = eval(f'f"""{prompt}"""')
    print("PROMPT_TEXT_CONTENT: ", prompt_text)
    response = llm.invoke(prompt_text)
    print("LLM RESPONSE: ", response)
    print("response content: ", response.content, "len response content: ", len(response.content))
    try:
      summary = response.content.split("```")[1].strip("markdown").strip()
      #summary = response['choices'][0]['text'].strip()
      print("Summary: ", summary, "len summary: ", len(summary))
    except IndexError as e:
      if len(response.content) != 0:
        summary = response.content
    return summary

# Function to summarize title
def generate_title(llm, row, maximum, prompt):
    prompt_text = eval(f'f"""{prompt}"""')
    print("PROMPT_TEXT_TITLE: ", prompt_text)
    response = llm.invoke(prompt_text)
    print("LLM RESPONSE: ", response)
    print("response content: ", response.content, "len response content: ", len(response.content))
    try:
      title = response.content.split("```")[1].strip("markdown").strip()
      print("title: ", title, "len title: ", len(title))
    except IndexError as e:
      if len(response.content) != 0:
        title = response.content
    return title

def get_final_df(llm: ChatGroq, is_url: bool, url_or_doc_path: str, maximum_content_length: int, maximum_title_length: int, chunk_size: Optional[int], text_summary_prompt: str, title_summary_prompt: str):
  
  if is_url:
    # parse webpage content `dict`
    webpage_data = scrape_website(url_or_doc_path)
    print("Webpage_data", webpage_data)
    # create the chunks. make sure it is a list of data passed in [dict]
    chunks =  create_chunks_from_webpage_data([webpage_data], chunk_size)
    # put content in a pandas dataframe. make sure it is a list of dict `[dict]` and not a 'dict'. chunks returned is a list[dict]
    """
     # limit the df for the moment  just to test, when app works fine we can release this constraint:
       `df = pd.DataFrame(chunks)[0:12]`
    """
    df = pd.DataFrame(chunks)[0:3] 
    print("DF: ", df.head())
  else:
    # Parse the PDF and create a DataFrame
    pdf_data = pdf_to_sections(url_or_doc_path)
    print("PDF DATA: ", pdf_data)
    df = pd.DataFrame(pdf_data)
    print("DF: ", df.head(10))

  ## CREATE FINAL DATAFRAME
  
  # generate summary of content text, we send row and will fetch there row[text]
  df['summary'] = df.apply(lambda row: summarize_text(llm, row, maximum_content_length, text_summary_prompt["system"]["template"]), axis=1) # here use llm to make the summary

  # Generate titles using section and text (we send the row and we will fetch there row[text], row[section])
  df['title'] = df.apply(lambda row: generate_title(llm, row, maximum_title_length, title_summary_prompt["system"]["template"]), axis=1)

  # Generate metadata and add UUID and retrieved fields
  df['id'] = [str(uuid4()) for _ in range(len(df))]
  
  if is_url:
    df['doc_name'] = df['url']
  else:
    df['doc_name'] = df['document']
  
  df['content'] = df['summary']
  df['retrieved'] = False

  # Select only the necessary columns
  df_final = df[['id', 'doc_name', 'title', 'content', 'retrieved']]
  print("Df Final from library: ", df_final, type(df_final))
  
  # create csv files as log of what have been produced for demonstration and human verification to later improve prompts or workflow quality data extraction
  # the normal flow won't use those logs but just use `df_final` to save it as `parquet` for efficiency in size and then saved to db later on
  if is_url:
    with open("./dataframes_csv/test_url_parser_df_output.csv", "w", encoding="utf-8") as f:
      df_final.to_csv(f, index=False)
  else:
    with open("./dataframes_csv/test_pdf_parser_df_output.csv", "w", encoding="utf-8") as f:
      df_final.to_csv(f, index=False)
  
  return df_final

# save dataframe to file path parquet formatted
def save_dataframe(df, directory="parsed_dataframes"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_id = str(uuid4())
    file_path = os.path.join(directory, f"{file_id}.parquet")
    df.to_parquet(file_path)
    return file_path

# load dataframe from file path (from a parquet formatted data)
def load_dataframe(file_path):
    return pd.read_parquet(file_path)

# delete files at certain path
def delete_parquet_file(file_path: str) -> None:
  """
  Deletes the specified parquet file from the filesystem.

  Args:
  file_path (str): The path to the parquet file to be deleted.
  """
  try:
    if os.path.exists(file_path):
      os.remove(file_path)
      print(f"File {file_path} has been deleted.")
    else:
      print(f"File {file_path} does not exist.")
  except Exception as e:
    print(f"Error occurred while deleting file {file_path}: {e}")

# function to decide next step from process_query to dtaframe storage or internet research
def decide_next_step(state: MessagesState):
    last_message = state['messages'][-1].content
    if isinstance(last_message, str) and last_message.endswith(".parquet"):  # Assuming the path is returned as a string
        print("Last Message to Decide Next Step (db storage): ", last_message)
        return "do_df_storage"
    else:
        print("Last Message to Decide Next Step (internet search): ", last_message)
        return "do_internet_search"

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
        return 'path'
    else:
        return 'text'

# function that gets different important messages and query llm to make a report
# maybe here we can see if those instead of getting vars from `MessagesState` we might want to use our custom state for that so other graphs can use those import information
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




# ********************* ----------------------------------------------- *************************************
##### QUERY & EMBEDDINGS #####

## 1- IDENTIFY IF QUERY IS FOR A PFD OR A WEB PAGE
# Main function to process the user query and return pandas dataframe and the user query dictionary structured by llm. here df_final will have chunks already done! cool!
#def process_query(llm: ChatGroq, query: str, maximum_content_length: int, maximum_title_length: int, chunk_size: Optional[int], detect_prompt_template: Dict = detect_content_type_prompt, text_summary_prompt: str = summarize_text_prompt, title_summary_prompt: str = generate_title_prompt) -> Tuple | Dict:
#def process_query(llm: ChatGroq = groq_llm_mixtral_7b, query: str = UserInput.user_initial_input, maximum_content_length: int = FuncInputParams.maximum_content_length, maximum_title_length: int = FuncInputParams.maximum_title_length, chunk_size: Optional[int] = FuncInputParams.chunk_size_df, detect_prompt_template: Dict = detect_content_type_prompt, text_summary_prompt: str = summarize_text_prompt, title_summary_prompt: str = generate_title_prompt, final_df: pd.DataFrame = FuncOutputs.df_final) -> Tuple | Dict:
def process_query(state: MessagesState) -> Dict:
    # get last state message
    messages = state['messages']
    print("messages from should_continue func {process_query}: ", messages)
    last_message = messages[-1].content
    if "nothing_in_cache_nor_vectordb" in last_message:
      last_message = last_message.split(":")[1].strip()
    print("Last Message: ", last_message, type(last_message))
    
    # variables
    query = last_message
    maximum_content_length = 200
    maximum_title_length = 30
    chunk_size = 250
    detect_prompt_template = detect_content_type_prompt
    text_summary_prompt = summarize_text_prompt
    title_summary_prompt = generate_title_prompt

    print("Query: ", query)

    # Default values in the function
    llm = groq_llm_mixtral_7b

    # use of detect_content_type from `query_analyzer_module`
    content_str = detect_content_type(llm, query, detect_prompt_template)
    content_dict = string_to_dict(content_str)

    # print content to see how it looks like, will it getthe filename if pdf or the full correct url if url...
    print("content_dict: ", content_dict)

    if content_dict["pdf"] and is_url_or_pdf(content_dict["pdf"].strip()) == "pdf":
        is_url = False
        # functions from `lib_helpers.pdf_parser`(for pdf we increase a bit the content lenght accepted and the chunk size)
        df_final = get_final_df(llm, is_url, content_dict["pdf"].strip(), maximum_content_length + 200, maximum_title_length, chunk_size + 150, text_summary_prompt, title_summary_prompt)
    elif content_dict["url"] and is_url_or_pdf(content_dict["url"].strip()) == "url":
        is_url = True
        # just for text to see if `is_url_or_pdf` func works as expected
        #return is_url
        # functions from `lib_helpers.webpage_parser`
        df_final = get_final_df(llm, is_url, content_dict["url"].strip(), maximum_content_length, maximum_title_length, chunk_size, text_summary_prompt, title_summary_prompt)
    else:
        # update state no_doc_but_text (thismeans that it is just a questiont hat can be answers combining llm answer and internet search for example)
        if content_dict["question"]:
          query_reformulated_in_question = content_dict["question"]      
          return {"messages": [{"role": "system", "content": query_reformulated_in_question}]}
        else:
          query_text = content_dict["text"]      
          return {"messages": [{"role": "system", "content": query_text}]}
 

    df_saved_to_path = save_dataframe(df_final, "parsed_dataframes")
    print("DF SAVED TO PATH: ", df_saved_to_path)
    # update state
    return {"messages": [{"role": "system", "content": df_saved_to_path}]}


## 2- STORE DOCUMENT DATA TO POSTGRESQL DATABASE
# Function to store cleaned data in PostgreSQL
"""
store dataframe to DB by creating a custom table and by activating pgvector and insering the dataframe rows in it
so here we will have different tables in the database (`conn`) so we will have the flexibility to delete the table entirely if not needed anymore
"""
def store_dataframe_to_db(state: MessagesState) -> Dict[str, Any]:
  """
  Store a pandas DataFrame to a PostgreSQL table using psycopg2, avoiding duplicate records.
  Ask to user when document already exist if user want to update records or not.
  Deletes the `.parquet` file for storage efficiency, when the dataframe is saved to db or user decides to not update the db records.
    
  Args:
  state MessagesState: The Graph stored messages from one node to another
  
  Returns:
  messages Dict0[str, Any]: a dictionary that updates the state messages by adding a new one reflecting this node function result passed to next node.  
  """

  # get last message from states
  messages = state['messages']
  df_path = messages[-1].content
  # get latest message from state and deserialize it
  df_final = load_dataframe(df_path)
  print("DF DESERIALIZED: ", df_final, type(df_final))
  # get table name from virtual env
  table_name = os.getenv("TABLE_NAME")
  print("TABLE NAME: ", table_name)
    
  if type(df_final) == pd.DataFrame:
    # Establish a connection to the PostgreSQL database
    try:
      conn = connect_db()
      cursor = conn.cursor()
    except Exception as e:
      print(f"Failed to connect to the database: {e}")
      return {"messages": [{"role": "system", "content": f"error: An error occured while trying to connect to DB in 'store_dataframe_to_db': {e}"}]}

    # Ensure the table exists, if not, create it
    try:
      cursor.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS {} (
          id UUID PRIMARY KEY,
          doc_name TEXT,
          title TEXT,
          content TEXT,
          retrieved BOOLEAN
        );
      """).format(sql.Identifier(table_name)))
      conn.commit()
    except Exception as e:
      print(f"Failed to create table {table_name}: {e}")
      cursor.close()
      conn.close()
      return {"messages": [{"role": "system", "content": f"error: An error occured while trying to 'create table if exist': {e}"}]}

    # Ensure pgvector extension is available
    try:
      cursor.execute(sql.SQL("CREATE EXTENSION IF NOT EXISTS vector;"))
      conn.commit()
    except Exception as e:
      print(f"Failed to enable pgvector extension: {e}")
      cursor.close()
      conn.close()
      return {"messages": [{"role": "system", "content": f"error: An error occured while trying to 'create extention vector in db': {e}"}]}

    # Check if the document already exists in the database
    try:
      cursor.execute(sql.SQL("""
        SELECT COUNT(*) FROM {}
        WHERE doc_name = {};
      """).format(sql.Identifier(table_name)), [str(df_final['doc_name'][0])])
      doc_exists = cursor.fetchone()[0] > 0

      if doc_exists:
        user_input = input("Document already present in the database. Do you want to update it? (yes/no): ").strip().lower()
        if user_input != 'yes':
          print("Document already in the database, user chose not to update. Can now perform retrieval on document information.")
          # delete the dataframe stored as parquet file
          delete_parquet_file(df_path)
          return {"messages": [{"role": "system", "content": "success: Document already in the database, user chose not to update. Retrieval can be performed"}]}

    except Exception as e:
      print(f"Error checking for existing document: {e}")
      cursor.close()
      conn.close()
      return {"messages": [{"role": "system", "content": f"error: An error occurred while checking for existing document in DB: {e}"}]}

    # Insert data into the table row by row, avoiding duplicates
    for index, row in df_final.iterrows():
      try:
        # Check if the content already exists in the table
        cursor.execute(sql.SQL("""
          SELECT COUNT(*) FROM {}
          WHERE doc_name = {} AND content = {};
        """).format(sql.Identifier(table_name)), [str(row["doc_name"]), str(row["content"])])
            
        count = cursor.fetchone()[0]
            
        if count == 0:
          # If no duplicate found, insert the row
          cursor.execute(sql.SQL("""
            INSERT INTO {} (id, doc_name, title, content, retrieved)
            VALUES (%s, %s, %s, %s, %s)
          """).format(sql.Identifier(table_name)), [str(row["id"]), str(row["doc_name"]), str(row["title"]), str(row["content"]), str(row["retrieved"])])
          conn.commit()  # Commit the transaction if successful
        else:
          print(f"Duplicate found for doc_name: {row['doc_name']}, content: {row['content']}. Skipping insertion.")

      except Exception as e:
        print(f"Error inserting row {index}: {e}")
        conn.rollback()  # Rollback in case of error for the current transaction
        return {"messages": [{"role": "system", "content": f"error: An error occured while trying to insert data in db: {e}"}]}

    # Close the cursor and the connection
    cursor.close()
    conn.close()

    print("DataFrame successfully stored in the database.")
    # delete the dataframe stored as parquet file
    delete_parquet_file(df_path)

    return {"messages": [{"role": "system", "content": f"success: Document quality data saved to PostgreSQL database table: {table_name}"}]}

  else:
    no_pdf_or_webpage_get_query_text = messages[-1].content
    print("No pdf or webpage therefore use query reformulated or raw query: ", no_pdf_or_webpage_get_query_text)
    return {"messages": [{"role": "system", "content": f"text: {no_pdf_or_webpage_get_query_text}"}]}


## 3- EMBED ALL DATABASE SAVED DOC DATA TO VECTORDB
# `COLLECTION_NAME` and `CONNECTION_STRING` should be recognized as it comes from 'lib_helpers.embedding_and_retrieval'
def custom_chunk_and_embed_to_vectordb(state: MessagesState) -> Dict[str, Any]:

  # vars
  # doc_name_or_url  3 maybe add here document name or url by saving it to dynamic env en loading it here so that we embed only the document concerning user needs
  table_name: str = os.getenv("TABLE_NAME") # "test_table"
  chunk_size: int = 500
  collection_name: str = COLLECTION_NAME
  connection_string: str = CONNECTION_STRING
  
  # Embed all documents in the database
  # function from module `lib_helpers.embedding_and_retrieval`
  # here we get List[Dict[str,Any]]
  try:  
    rows = fetch_documents(table_name)
  except Exception as e:
    conn.close()
    # update state
    return {"messages": [{"role": "system", "content": f"error: An error occured while trying to fetch rows from db -> {e}"}]}
  # here we get List[List[Dict[str,Any]]]
  try:
    # chunk will be formatted only with uuid and content: `{'UUID': str(doc_id), 'content': content}` but here it is a `List[List[Dict[str,Any]]]`
    chunks = create_chunks_from_db_data(rows, chunk_size)
  except Exception as e:
    conn.close()
    # update state
    return {"messages": [{"role": "system", "content": f"error: An error occured while trying to create chunks -> {e}"}]}
  # here we create the custom document and embed it
  try:
    for chunk_list in chunks:
      # here chunk_list is a `List[Dict[str,Any]]`
      embed_all_db_documents(chunk_list, collection_name, connection_string) 
  except Exception as e:
    conn.close()
    # update state
    return {"messages": [{"role": "system", "content": f"error: An error occured while trying to create custom docs and embed it -> {e}"}]}
  
  conn.close()
  # update state
  
  return {"messages": [{"role": "system", "content": "success: data chunks fully embedded to vectordb"}]}
   




# ********************* ----------------------------------------------- *************************************
###### QUERY AND RETRIEVAL ######

## 4- FETCH QUERY FROM REDIS CACHE, IF NOT FOUND ONLY THEN DO A VECTOR RETRIEVAL FROM VECTORDB

#def query_redis_cache_then_vecotrdb_if_no_cache(table_name: str, query: str, score: float, top_n: int) -> List[Dict[str,Any]] | str | Dict[str,str]:
def query_redis_cache_then_vecotrdb_if_no_cache(state: MessagesState) -> Dict[str,Any]:

  messages = state['messages']
  last_message = messages[-1].content
  print("Last Message: ", last_message) 
 
  table_name: str = os.getenv("TABLE_NAME")
  query: str = last_message
  score: float = 0.3 
  top_n: int = 2

  try:
    response = handle_query_by_calling_cache_then_vectordb_if_fail(table_name, query, score, top_n)
    print("RESPONSE CACHE OR VDB: ", json.dumps(response, indent=4))
    
    if response:


      # here the redis response from hashed query matching
      if "exact_match_search_response_from_cache" in response: 
        exact_match_response = response["exact_match_search_response_from_cache"]
        # List[Dict[str,Any]]
        #return exact_match_response
        return {"messages": [{"role": "ai", "content": f"success_exact_match: {exact_match_response}"}]}

      # here the redis response from semantic search on stored embeddings as no hashed query matched to get an answer
      elif "semantic_search_response_from_cache" in response: 
        semantic_response = response["semantic_search_response_from_cache"]
        # List[Dict[str,Any]]
        #return semantic_response
        return {"messages": [{"role": "ai", "content": f"success_semantic: {semantic_response}"}]}

      # here the vectordb retrieved answer and also cached in redis for next search as for this one nothign was found in cache
      elif "vector_search_response_after_cache_failed_to_find" in response:
        vector_response = response["vector_search_response_after_cache_failed_to_find"]
        # List[Dict[str,Any]]
        #return vector_response
        return {"messages": [{"role": "ai", "content": f"success_vector: {vector_response}"}]}

      # here a message that suggest to perform internet search on the query as nothing has been found in redis and vectordb
      elif "message" in response:
        print(response["message"])
        # str
        str_response = response["message"]
        #return str_response
        return {"messages": [{"role": "ai", "content": f"success_str_response: {str_response}"}]}

      # Here to catch any errors
      elif "error" in response:
        #raise Exception(response["error"])
        response_error = response["error"]
        return {"messages": [{"role": "ai", "content": f"error_response: {response_error}"}]}
    
  except Exception as e:
    # Dict[str,str]
    #return {"error": f"An error occured while trying to handle query by calling cache then vectordb if nothing found: {e}"}
    return {"messages": [{"role": "ai", "content": f"error_exception: An error occured while trying to handle query by calling cache then vectordb if nothing found: {e}"}]}

"""

"""
"""
    # here this means that the cache search and vector search did fail to find relevant information
    # we need then to do an internet search using another node or to have this in a conditional eadge
"""

###### INTERNET SEARCH TOOL ######
## -5 PERFORM INTERNET SEARCH IF NO ANSWER FOUND IN REDIS OR PGVECTOR

# function that check states to know if we perform internet search, should handle all the cases in which internet search can be called

# search through internet and get 5 search results
def internet_research_user_query(state: MessagesState):

    query = state["messages"][-1].content  # Extract the last user query
    
    # Fill the prompt template
    generate_from_empty_prompt["system"]["template"] = "You are an expert search engine and use the tool available to answer to user query."
    generate_from_empty_prompt["human"]["template"] = f"Find the most relevant information about: {query}"
    generate_from_empty_prompt["ai"]["template"] = ""
    #generate_from_empty_prompt["human"]["input_variables"] = {"query": query}
    
    # Construct the prompt for the LLM
    system_message = generate_from_empty_prompt["system"]["template"]
    print("System Message: ", system_message)
    human_message = generate_from_empty_prompt["human"]["template"]
    print("Human Message: ", human_message)
    ai_message = generate_from_empty_prompt["ai"]["template"]

    # Create the messages list
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=human_message),
        #AIMessage(content=ai_message),
    ]
    print("Messages: ", messages, "\nLen Messages: ", len(messages))
    
    # Use the tool with the constructed prompt
    # first way of doing it asking llm only
    #search_results = groq_llm_mixtral_7b.invoke(messages)
    #print("Result Search: ", search_results)
    
    # second way to do it using duckduckgo result
    #results2 = internet_search_tool.run(query)
    #print("Results2: ", results2)
    
    # third way using an agent?????
    # Create an agent executor
    internet = [tool_internet]
    agent_executor = AgentExecutor.from_agent_and_tools(agent=llm_with_internet_search_tool, tools=internet)
    
    tool_executor = ToolExecutor(internet)
    print("TYPE QUERY: ", type(query))
    result_agent = tool_executor.invoke({"input":query})
    print("Result Agent: ", result_agent)

    # Test the AgentExecutor directly
    human_message = HumanMessage(content=query)

    try:
        result_agent = agent_executor(messages)
        print("Result Agent:", result_agent)
        # Return the formatted message
        return {"messages": [{"role": "assistant", "content": result_agent}]}
    except Exception as e:
        print("Error during invocation:", e)



'''
# Initialize states
workflow = StateGraph(DetectContentState, ParseDocuments, StoreDftodbState, ChunkAndEmbedDbdataState, RetrievedRedisVectordb)

# each node will have one function so one job to do
workflow.add_node("internet_tool", internet_tool_node)

# each conditional edge will have an agent to go from and a function that perform task to determine which is the next agent called.
workflow.add_conditional_edges(
    # here define the start node for this conditional edge. This means these are the edges taken after the `agent` node is called. so new routes
    "<determine_after_which_agent_we_call_this_edge>",
    # Next, we pass in the function that will determine which node will be reached next. so this function should return END or next node name
    <put_func_name>,
    {
        "search": "web_search", # if output == "search" use web_search node
        "generate": "generate",  # if output == "generate" use generate node
    },
)

workflow.add_edge("web_search", "generate") 
'''

##### PROMPTS CREATION TO EXPLAIN TO LLM WHAT TO DO AND CONDITIONS IF ANY #####
## 6- CREATE PROMPT TO INSTRUCT LLM
# Use internet search if there is nothing retireved to get more info
"""
Langfuse can also be used here to store prompts that we will just call in the graph when needed...
But we will first here create prompts in the conventional way before testing the langfuse way (to have the need of less dependencies and jsut use langchain and langgraph)
"""

# create prompts format, we can pass in as many kwargs as wanted it will unfold and place it properly
# we have two schema one for normal question/answer prompts and another for chat_prompts system/human/ai

def call_chain(model: ChatGroq, prompt: PromptTemplate, prompt_input_vars: Optional[Dict], prompt_chat_input_vars: Optional[Dict]):
  # only to concatenate all input vars for the system/human/ai chat
  chat_input_vars_dict = {}
  for k, v in prompt_chat_input_vars.items():
    for key, value in v.items():
      chat_input_vars_dict[key] = value
  print("Chat input vars: ",  chat_input_vars_dict)

  # default question/answer
  if prompt and prompt_input_vars:
    print("Input variables found: ", prompt_input_vars)
    chain = ( prompt | model )
    response = chain.invoke(prompt_input_vars)
    print("Response: ", response, type(response))
    return response.content.split("```")[1].strip("markdown").strip()
    
  # special for chat system/human/ai
  elif prompt and prompt_chat_input_vars:
    print("Chat input variables found: ", prompt_chat_input_vars)
    chain = ( prompt | model )
    response = chain.invoke(chat_input_vars_dict)
    print("Response: ", response, type(response))
    return response.content.split("```")[1].strip("markdown").strip()

  print("Chat input variables NOT found or missing prompt!")
  chain = ( prompt | model )
  response = chain.invoke(input={})
  print("Response: ", response, type(response))
  return response.content.split("```")[1].strip("markdown").strip()

def make_normal_or_chat_prompt_chain_call(llm_client, prompt_input_variables_part: Dict, prompt_template_part: Optional[Dict], chat_prompt_template: Optional[Dict]):
  
  # default prompt question/answer
  if prompt_template_part:
    prompt = (
      PromptTemplate.from_template(prompt_template_part)
    )
    response = call_chain(llm_client, prompt, prompt_input_variables_part, {})
    return response
  
  # chat prompts question/answer system/human/ai
  elif chat_prompt_template:
    prompt = (
      SystemMessage(content=chat_prompt_template["system"]["template"]) + HumanMessage(content=chat_prompt_template["human"]["template"]) + AIMessage(content=chat_prompt_template["ai"]["template"])
    )
    response = call_chain(llm_client, prompt, {}, {"system": chat_prompt_template["system"]["input_variables"], "human": chat_prompt_template["human"]["input_variables"], "ai": chat_prompt_template["ai"]["input_variables"]})
    return response
  return {'error': "An error occured while trying to create prompts. You must provide either: `prompt_template_part` with `prompt_input_variables_part` or `chat_prompt_template` - in combinaison with llm_client which is always needed." }

# just to test
from prompts.prompts import test_prompt_tokyo, test_prompt_siberia, test_prompt_monaco, test_prompt_dakar

#print("SIBERIA: \n", make_normal_or_chat_prompt_chain_call(groq_llm_llama3_8b, test_prompt_siberia["input_variables"], test_prompt_siberia["template"], {}))
#time.sleep(0.5)
#print("TOKYO: \n", make_normal_or_chat_prompt_chain_call(groq_llm_llama3_8b, test_prompt_tokyo["input_variables"], test_prompt_tokyo["template"], {}))
#time.sleep(0.5)
#print("DAKAR: \n", make_normal_or_chat_prompt_chain_call(groq_llm_llama3_8b, {}, {}, test_prompt_dakar))
#time.sleep(0.5)
#print("MONACO: \n", make_normal_or_chat_prompt_chain_call(groq_llm_llama3_8b, {}, {}, test_prompt_monaco))



"""
1- import the right prompt template normal question/response or chat special system/human/aione
2- use the template and the function argument to create the prompt
3- use make_normal_or_chat_prompt_chain_call(llm_client, prompt_input_variables_part: Dict, prompt_template_part: Optional[Dict], chat_prompt_template: Optional[Dict])
4- use internet search tool if nothing is found or go to the internet search node
"""


###### CREATE WORKFLOW TO PLAN NODE AND EDGES NEEDS #######
## 7- CREATE WORKFLOW
"""
Here we will create hypothetical desired workflow for the graph

--- START 
    0--- ask user query input
    1--- Analyze User Query To Get the Webpage or Document or just text (to know if we need to perform webpage parsing or pdf doc parsing) 
         --- save to state 'webpage' or 'pdf' or 'text'
        1.1--- If/Else conditional edge 'db_doc_check':
           --- output = 'webpage/pdf title IN db'
           1.1.1--- If webpage title or pdf title in db --- If/Else conditional edge 'cache/vectordb query answer check'
                                                        1.1.1.1--- If answer in cache/vectordb --- answer user query --- END
                                                        1.1.1.2--- Else Internet Search about subject
                                                                   --- Embed query/internet_answer and cache it
                                                                       --- answer user query with internet_search result
                                                                           --- END
           --- output = 'webpage/pdf title NOT IN db'
           1.1.2--- Else --- If/Else conditional edge 'webpage or pdf processing':
                         --- output = 'url'
                         1.1.2.1--- If webpage --- parse webpage
                                                   --- store to database
                                                       --- create chunks from database rows
                                                           --- embed chunks
                                                               --- save to state 'webpage embedded'
                                                                   --- go to 1.1.1.1 flow which will search in cache and won't found it and make a vector search and answer-> END
                         --- output = 'pdf'
                         1.1.2.2--- If pdf     --- parse pdf
                                                   --- store to database
                                                       --- create chunks from database rows
                                                           --- embed chunks
                                                               --- save to state 'pdf embedded'
                                                                   --- go to 1.1.1.1 flow which will search in cache and won't found it and make a vector search and answer-> END
                         --- output = 'text'
                         1.1.2.3--- If text    --- perform internet search to get response of `text only` query
                                                   --- format internet result and answer -> END

    2--- Analyze User Query to extract the question from it and rephrase the question to optimize llm information research/retrieval --- save to state 'query/question'
         3--- Retrieve answer from Query in embedded collection
              --- save to state answer_retrieved
                  4--- Internet Search user query
                       --- save to state internet search_answer
                           5--- answer to user in markdown report format using both states answer_retrieved/search_answer
"""

## 8- CREATE NODES, EDGES AND CONDITIONAL EDGES

"""
creditizens_doc_report_flow = StateGraph(MessagesState)
creditizens_doc_report_flow.set_entry_point("<node_name>")
creditizens_doc_report_flow.add_node("<node_name>", <function_associated_to_node_action>)
# condition adge = conditional route from one node to a function output which will determine next node
creditizens_doc_report_flow.add_conditional_edges(
    # From which node
    "<node_name>",
    # function that will determine which node is called next
    <function_with_condition>,
    # function called output will determine next node
    {
      "output1": "<node_name>", # if output == "output1" go to specific node
      "output2": "<node_name>",  # if output == "output2" go to specific node
    },
)
# edge are route so from node to another
creditizens_doc_report_flow.add_edge("<node_name1>", "<node_name2>")
# checkpointer for memory of graph and compile the graph
checkpointer = MemorySaver()
creditizens_graph_flow_app = workflow.compile(checkpointer=checkpointer)
# inject user query in the creditizens_doc_report_flow: "can maybe here think of an app that ask user input to collect it and use it to launch the creditizens_graph_flow_app"
user_query_to_answer_using_creditizens_graph_flow = creditizens_graph_flow_app.invoke(
    {"messages": [HumanMessage(content="what is the weather in sf")]},
    config={"configurable": {"thread_id": 42}}
)
user_query_to_answer_using_creditizens_graph_flow["messages"][-1].content

________________________________________________________________________________________
creditizens_doc_report_flow = StateGraph(MessagesState)

Nodes:
creditizens_doc_report_flow.add_node("ask_user_input", <function_associated_to_node_action>)
creditizens_doc_report_flow.add_node("webpage_or_pdf_embedding_or_vector_search", query_analyzer_module.detect_content_type)
creditizens_doc_report_flow.add_node("user_query_analyzer_type_of_doc_extraction", <function_associated_to_node_action>)
creditizens_doc_report_flow.add_node("user_query_extract_question_and_rephrase", <function_associated_to_node_action>)
creditizens_doc_report_flow.add_node("check_if_document_in_db", doc_db_search)
creditizens_doc_report_flow.add_node("cache_vectordb_search", query_matching.handle_query_by_calling_cache_then_vectordb_if_fail)
creditizens_doc_report_flow.add_node("answer_user_query", <function_associated_to_node_action>) # this will END
creditizens_doc_report_flow.add_node("internet_search", <function_associated_to_node_action>)
# can add also save query and internet search to DB and put in title a llm generated title for this query/answer, and have doc_name being =internet search
creditizens_doc_report_flow.add_node("save_to_db", <function_associated_to_node_action>)
creditizens_doc_report_flow.add_node("embed", <function_associated_to_node_action>)
creditizens_doc_report_flow.add_node("webpage_parser", webpage_parser.get_df_final)
creditizens_doc_report_flow.add_node("pdf_parser", pdf_parser.get_final_df)
creditizens_doc_report_flow.add_node("create_chunk", <func>)
creditizens_doc_report_flow.add_node("markdown_report_creation", <func>)

0              creditizens_doc_report_flow.set_entry_point("ask_user_input")
1              creditizens_doc_report_flow.add_edge("user_query_analyzer_type_of_doc_extraction", "webpage_or_pdf_embedding_or_vector_search")
               creditizens_doc_report_flow.add_conditional_edges(
                 "webpage_or_pdf_embedding_or_vector_search",
                 query_analyzer_module.detect_content_type,
                 {
                   "url": "check_if_document_in_db", 
                   "pdf": "check_if_document_in_db",
                   "text": "cache_vectordb_search", # this will go to the flow 2
                 },
               )
1.1            creditizens_doc_report_flow.add_edge("webpage_or_pdf_embedding_or_vector_search", "check_if_document_in_db")
               creditizens_doc_report_flow.add_conditional_edges(
                 # From which node
                 "check_if_document_in_db",
                 # function that will determine which node is called next
                 doc_db_search,
                 # function called output will determine next node
                 {
                   "title_found": "cache_vectordb_search", # we go search in cache/vectordb
                   "title_not_found": "webpage_or_pdf_embedding_or_vector_search",
                 },
               )
1.1.1          creditizens_doc_report_flow.add_edge("check_if_document_in_db", "cache_vectordb_search")
               creditizens_doc_report_flow.add_conditional_edges(
                 "cache_vectordb_search",
                 query_matching.handle_query_by_calling_cache_then_vectordb_if_fail,
                 {
                   "answer_found": "answer_user_query", # this will answer and END
                   
                   "answer_not_found": "internet_search", # here we will start a long process of nodes and edges
                 },
               )
1.1.1.1        creditizens_doc_report_flow.add_edge("cache_vectordb_search", "answer_user_query") # this will END

1.1.1.2        creditizens_doc_report_flow.add_edge("cache_vectordb_search", "internet_search")
               # save to db here query and internet search result, llm generated title for this query/answer, and have doc_name being =internet search
               creditizens_doc_report_flow.add_edge("internet_search", "save_to_db")
               creditizens_doc_report_flow.add_edge("save_to_db", "embed")
               creditizens_doc_report_flow.add_edge("embed", "answer_user_query") # this will END
               
1.1.2          creditizens_doc_report_flow.add_edge("check_if_document_in_db", "webpage_or_pdf_embedding_or_vector_search")
               creditizens_doc_report_flow.add_conditional_edges(
                 "webpage_or_pdf_embedding_or_vector_search",
                 query_analyzer_module.detect_content_type,
                 {
                   "url": "webpage_parser",
                   "pdf": "pdf_parser",
                   "text": "user_query_extract_question_and_rephrase", # this will go to the flow 2
                 },
               )
1.1.2.1        creditizens_doc_report_flow.add_edge("webpage_or_pdf_embedding_or_vector_search", "webpage_parser")
               creditizens_doc_report_flow.add_edge("webpage_parser", "save_to_db")
               creditizens_doc_report_flow.add_edge("save_to_db", "create_chunks")
               creditizens_doc_report_flow.add_edge("create_chunks", "embed")
               creditizens_doc_report_flow.add_edge("embed", cache_vectordb_search)
               creditizens_doc_report_flow.add_edge("cache_vectordb_search", "answer_user_query") # this will END

1.1.2.2        creditizens_doc_report_flow.add_edge("webpage_or_pdf_embedding_or_vector_search", "pdf_parser")
               creditizens_doc_report_flow.add_edge("pdf_parser", "save_to_db")
               creditizens_doc_report_flow.add_edge("save_to_db", "create_chunks")
               creditizens_doc_report_flow.add_edge("create_chunks", "embed")
               creditizens_doc_report_flow.add_edge("embed", cache_vectordb_search)
               creditizens_doc_report_flow.add_edge("cache_vectordb_search", "answer_user_query") # this will END

2              creditizens_doc_report_flow.add_edge("user_query_extract_question_and_rephrase", "webpage_or_pdf_embedding_or_vector_search")
               creditizens_doc_report_flow.add_edge("webpage_or_pdf_embedding_or_vector_search", "cache_vectordb_search")
               creditizens_doc_report_flow.add_edge("cache_vectordb_search", "internet_search")
3              creditizens_doc_report_flow.add_edge("internet_search", "markdown_report_creation")
4              creditizens_doc_report_flow.add_edge("markdown_report_creation", END)

For all conditional edge to make it simpler for some, save to state the value of the function return and get just the state value in the conditional edge
"""
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
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import StructuredTool

import pandas as pd
import psycopg2

from typing import Dict, List, Any, Optional, Union

import json
import ast

import re

from prompts.prompts import (
  detect_content_type_prompt,
  summarize_text_prompt,
  generate_title_prompt,
  answer_user_with_report_from_retrieved_data_prompt,
  structured_outpout_report_prompt
)
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
from lib_helpers.query_matching import handle_query_by_calling_cache_then_vectordb_if_fail
from app_tools.app_tools import (
  # internet node & internet llm binded tool
  tool_search_node,
  llm_with_internet_search_tool
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
  decide_next_step,
  # delete parquet file after db storage
  delete_parquet_file
)

import subprocess

from langchain_community.vectorstores.pgvector import PGVector
from langchain.docstore.document import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
#for structured output setup
from langchain_core.pydantic_v1 import BaseModel, Field
from typing_extensions import Annotated, TypedDict
#from langchain.tools.render import format_tool_to_openai_function
#from langchain_core.utils.function_calling import convert_pydantic_to_openai_function
from langchain_core.utils.function_calling import convert_to_openai_function # use this one to replace both `format_tool_to_openai_function` and `convert_pydantic_to_openai_function`
from langchain_core.output_parsers.pydantic import PydanticOutputParser


from app_states.app_graph_states import StateCustom
# for graph creation and management
from langgraph.checkpoint import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

# display drawing of graph
from IPython.display import Image, display
# TOOLS
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
    #return {"messages": [search_results]}
    return search_results

tool_search_node = ToolNode([search])

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

# STRUCTURED OUTPUT LLM BINDED WITH TOOL
structured_internet_research_llm_tool = groq_llm_mixtral_7b.with_structured_output(InternetSearchStructuredOutput)


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

def handle_query_by_calling_cache_then_vectordb_if_fail_conditional_edge_decision(state: MessagesState):
  messages = state['messages']
  last_message = messages[-1].content
  with open("./logs/conditional_edge_logs.log", "a", encoding="utf-8") as conditional:
    conditional.write("\nhandle_query_by_calling_cache_then_vectordb_if_fail_conditional_edge_decision:\n")
    conditional.write(f"- last message: {last_message}")

  if ("error_hash" or "error_semantic" or "error_vector_retrieved_and_cached" or "error_vector" or "error_cache_and_vector_retrieval") in last_message:
    with open("./logs/conditional_edge_logs.log", "a", encoding="utf-8") as conditional:
      conditional.write(f"- error: {last_message}\n\n")
    return "error_handler"

  elif "success_hash" in last_message:
    with open("./logs/conditional_edge_logs.log", "a", encoding="utf-8") as conditional:
      conditional.write(f"- success_hash: {last_message}\n\n")
    #return "chunk_and_embed_from_db_data"
    #return "internet_search_agent"
    return "answer_user"

  elif "success_semantic" in last_message:
    with open("./logs/conditional_edge_logs.log", "a", encoding="utf-8") as conditional:
      conditional.write(f"- success_semantic: {last_message}\n\n\n\n")
    #return "internet_search_agent"
    return "answer_user"

  elif "success_vector_retrieved_and_cached" in last_message:
    with open("./logs/conditional_edge_logs.log", "a", encoding="utf-8") as conditional:
      conditional.write(f"- success_vector_retrieved_and_cached: {last_message}\n\n")
    #return "internet_search_agent"
    return "answer_user"

  elif "nothing_in_cache_nor_vectordb" in last_message:
    with open("./logs/conditional_edge_logs.log", "a", encoding="utf-8") as conditional:
      conditional.write(f"- nothing_in_cache_nor_vectordb: {last_message}\n\n\n\n")
    # will process user query and start the dataframe creation flow and embedding storage of that df
    return "dataframe_from_query"
    #return

  else:
    with open("./logs/conditional_edge_logs.log", "a", encoding="utf-8") as conditional:
      conditional.write(f"- error last: {last_message}\n\n")
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

'''
1 - check if user have document or url if it is in present in DB or not
    - if in doc/url DB, query the cache
      - if not in cache we try vector retrieval of answer
      - if not in vector we perform internet search
        - then we answer to user with internet search response and cache the query/internet_answer
    - if doc/url not in DB we `process_query` so create the dataframe and follow the flow
      - store_dataframe_to_db
        - custom_chunk_and_embed_to_vectordb
          - 'query_redis_cache_then_vecotrdb_if_no_cache' which is gonna retrieve infor from db as redis have been checked before and nothing was there
            - answer user with the retrieved vector response
            - we can also perform internet search and get llm to combine vector retrieval if any with internet result to provide a formated answer

                
'''

# Initialize states
workflow = StateGraph(MessagesState)

# each node will have one function so one job to do
workflow.add_node("error_handler", error_handler) # will be used to end the graph returning the app system error messages
workflow.add_node("get_user_input", get_user_input)
workflow.add_node("handle_query_by_calling_cache_then_vectordb_if_fail", handle_query_by_calling_cache_then_vectordb_if_fail)
workflow.add_node("dataframe_from_query", process_query)
workflow.add_node("store_dataframe_to_db", store_dataframe_to_db)
workflow.add_node("chunk_and_embed_from_db_data", custom_chunk_and_embed_to_vectordb)
workflow.add_node("internet_search_agent", internet_search_agent) # -2 user input, -1 data retrieved
workflow.add_node("tool_search_node", tool_search_node) # -3 user input, -2 data retrieved, -1 schema internet tool
# special andwser report fetching user query, database retrieved data and internet search result
workflow.add_node("answer_user_with_report_from_retrieved_data", answer_user_with_report_from_retrieved_data) # -4 user input, -3 data retrieved, -2 schema internet tool, -1 internet search result
workflow.add_node("answer_user_with_report", answer_user_with_report)
workflow.add_node("answer_user", final_answer_user)
#workflow.add_node("", internet_research_user_query)

workflow.set_entry_point("get_user_input")
workflow.add_edge("get_user_input", "handle_query_by_calling_cache_then_vectordb_if_fail")

# `handle_query_by_calling_cache_then_vectordb_if_fail` edge
workflow.add_conditional_edges(
    "handle_query_by_calling_cache_then_vectordb_if_fail",
    handle_query_by_calling_cache_then_vectordb_if_fail_conditional_edge_decision, 
    #"dataframe_from_query"
)

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

# No NEED AS `handle_query_by_calling_cache_then_vectordb_if_fail` NODE MANAGES RETRIEVAL
#workflow.add_edge("chunk_and_embed_from_db_data", "retrieve_data_from_query")
#workflow.add_edge("retrieve_data_from_query", "answer_user")


# TOOLS EDGES
#workflow.add_edge("internet_search_agent", "answer_user")
#workflow.add_edge("internet_search_agent", "tool_search_node")
#workflow.add_edge("tool_search_node", "answer_user_with_report")

workflow.add_edge("error_handler", "answer_user")
workflow.add_edge("answer_user", END)
#workflow.add_edge("answer_user_with_report", END)


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
## 10- CREATE CONFIGS FOR GRAPH INTERRUPTION OR OTHER NEEDS












