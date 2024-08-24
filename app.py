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
  SystemMessage
)
from langchain.prompts import (
  PromptTemplate,
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate,
  AIMessagePromptTemplate
)
from app_tools.app_tools import internet
from langchain_core.output_parsers import JsonOutputParser
# for graph creation and management
from langgraph.checkpoint import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
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
# STATES
from app_states.app_graph_states import StateCustom
# DOCKER REMOTE CODE EXECUTION
# eg.: print(run_script_in_docker("test_dockerfile", "./app.py"))
from docker_agent.execution_of_agent_in_docker_script import run_script_in_docker # returns `Tuple[str, str]` stdout,stderr



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
    # print("last message from should_continue func: ", last_message)
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        # print("Tool called!")
        return "tools"
    # Otherwise, we stop (reply to the user)
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
workflow.add_edge("tools", 'agent')
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
  if is_url:
    with open("test_url_parser_df_output.csv", "w", encoding="utf-8") as f:
      df_final.to_csv(f, index=False)
  else:
    with open("test_pdf_parser_df_output.csv", "w", encoding="utf-8") as f:
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

# function to decide enxt step from process_query to dtaframe storage or internet research
def decide_next_step(state: MessagesState):
    last_message = state['messages'][-1].content
    if isinstance(last_message, str) and last_message.endswith(".parquet"):  # Assuming the path is returned as a string
        print("Last Message to Decide Next Step (db storage): ", last_message)
        return "do_df_storage"
    else:
        print("Last Message to Decide Next Step (internet search): ", last_message)
        return "do_internet_search"

# ********************* ----------------------------------------------- *************************************
##### QUERY & EMBEDDINGS #####

## 1- IDENTIFY IF QUERY IS FOR A PFD OR A WEB PAGE
# Main function to process the user query and return pandas dataframe and the user query dictionary structured by llm. here df_final will have chunks already done! cool!
#def process_query(llm: ChatGroq, query: str, maximum_content_length: int, maximum_title_length: int, chunk_size: Optional[int], detect_prompt_template: Dict = detect_content_type_prompt, text_summary_prompt: str = summarize_text_prompt, title_summary_prompt: str = generate_title_prompt) -> Tuple | Dict:
#def process_query(llm: ChatGroq = groq_llm_mixtral_7b, query: str = UserInput.user_initial_input, maximum_content_length: int = FuncInputParams.maximum_content_length, maximum_title_length: int = FuncInputParams.maximum_title_length, chunk_size: Optional[int] = FuncInputParams.chunk_size_df, detect_prompt_template: Dict = detect_content_type_prompt, text_summary_prompt: str = summarize_text_prompt, title_summary_prompt: str = generate_title_prompt, final_df: pd.DataFrame = FuncOutputs.df_final) -> Tuple | Dict:
def process_query(state: MessagesState) -> Dict:
    # get last state message
    messages = state['messages']
    print("messages from should_continue func: ", messages)
    last_message = messages[-1].content
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
          query_text = content_dict["question"]      
          return {"messages": [{"role": "system", "content": query_text}]}
 
    
    #return df_finaldf_final, content_dict
    
    # update state

    df_saved_to_path = save_dataframe(df_final, "parsed_dataframes")
    print("DF SAVED TO PATH: ", df_saved_to_path)
    print("LOAD DF: ", load_dataframe(df_saved_to_path))

    return {"messages": [{"role": "system", "content": df_saved_to_path}]}


## 2- STORE DOCUMENT DATA TO POSTGRESQL DATABASE
# Function to store cleaned data in PostgreSQL
"""
store dataframe to DB by creating a custom table and by activating pgvector and insering the dataframe rows in it
so here we will have different tables in the database (`conn`) so we will have the flexibility to delete the table entirely if not needed anymore
"""
def store_dataframe_to_db(state: MessagesState) -> Dict[str, str]:
    """
    Store a pandas DataFrame to a PostgreSQL table using psycopg2, avoiding duplicate records.
    
    Args:
    df_final (pd.DataFrame): The DataFrame containing the data to store.
    table_name (str): The name of the table where data should be stored.
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
        StoreDftodbState.df_stored = False
        return {"messages": [{"role": "system", "content": {"error": f"An error occured while trying to connect to DB in 'store_dataframe_to_db': {e}"}}]}

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
        StoreDftodbState.df_stored = False
        return {"messages": [{"role": "system", "content": {"error": f"An error occured while trying to 'create table if exist': {e}"}}]}

      # Ensure pgvector extension is available
      try:
        cursor.execute(sql.SQL("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()
      except Exception as e:
        print(f"Failed to enable pgvector extension: {e}")
        cursor.close()
        conn.close()
        StoreDftodbState.df_stored = False
        return {"messages": [{"role": "system", "content": {"error": f"An error occured while trying to 'create extention vector in db': {e}"}}]}

      # Insert data into the table row by row, avoiding duplicates
      for index, row in df_final.iterrows():
        try:
            # Check if the content already exists in the table
            cursor.execute(sql.SQL("""
                SELECT COUNT(*) FROM {}
                WHERE doc_name = {} AND content = {};
            """).format(sql.Identifier(table_name), sql.Literal(row["doc_name"]), sql.Literal(row["content"])))
            
            count = cursor.fetchone()[0]
            
            if count == 0:
                # If no duplicate found, insert the row
                cursor.execute(sql.SQL("""
                    INSERT INTO {} (id, doc_name, title, content, retrieved)
                    VALUES ({}, {}, {}, {}, {})
                """).format(sql.Identifier(table_name), sql.Literal(row["id"]), sql.Literal(row["doc_name"]), sql.Literal(row["title"]), sql.Literal(row["content"]), sql.Literal(row["retrieved"])))
                conn.commit()  # Commit the transaction if successful
            else:
                print(f"Duplicate found for doc_name: {row['doc_name']}, content: {row['content']}. Skipping insertion.")

        except Exception as e:
            print(f"Error inserting row {index}: {e}")
            conn.rollback()  # Rollback in case of error for the current transaction
            StoreDftodbState.df_stored = False
            return {"messages": [{"role": "system", "content": {"error": f"An error occured while trying to insert data in db: {e}"}}]}

      # Close the cursor and the connection
      cursor.close()
      conn.close()

      print("DataFrame successfully stored in the database.")
      StoreDftodbState.df_stored = True
      return {"messages": [{"role": "system", "content": f"Document quality data saved to PostgreSQL database table: {table_name}"}]}

    else:
      no_pdf_or_webpage_get_query_text = messages[-1].content
      print("No pdf or webpage therefore use query reformulated or raw query: ", no_pdf_or_webpage_get_query_text)
      return {"messages": [{"role": "system", "content": f"{no_pdf_or_webpage_get_query_text.content.query}"}]}


## 3- EMBED ALL DATABASE SAVED DOC DATA TO VECTORDB
# `COLLECTION_NAME` and `CONNECTION_STRING` should be recognized as it comes from 'lib_helpers.embedding_and_retrieval'
def custom_chunk_and_embed_to_vectordb(table_name: str, chunk_size: int, COLLECTION_NAME: str, CONNECTION_STRING: str) -> Dict[str, str]:
  # Embed all documents in the database
  # function from module `lib_helpers.embedding_and_retrieval`
  # here we get List[Dict[str,Any]]
  try:  
    rows = fetch_documents(table_name)
  except Exception as e:
    conn.close()
    # update state
    StateCustom.df_data_chunked_and_embedded = False
    return {"error": f"An error occured while trying to fetch rows from db -> {e}"}
  # here we get List[List[Dict[str,Any]]]
  try:
    # chunk will be formatted only with uuid and content: `{'UUID': str(doc_id), 'content': content}` but here it is a `List[List[Dict[str,Any]]]`
    chunks = create_chunks_from_db_data(rows, chunk_size)
  except Exception as e:
    conn.close()
    # update state
    StateCustom.df_data_chunked_and_embedded = False
    return {"error": f"An error occured while trying to create chunks -> {e}"}
  # here we create the custom document and embed it
  try:
    for chunk_list in chunks:
      # here chunk_list is a `List[Dict[str,Any]]`
      embed_all_db_documents(chunk_list, COLLECTION_NAME, CONNECTION_STRING) 
  except Exception as e:
    conn.close()
    # update state
    StateCustom.df_data_chunked_and_embedded = False
    return {"error": f"An error occured while trying to create custom docs and embed it -> {e}"}
  
  conn.close()
  # update state
  state.df_data_chunked_and_embedded = True
  
  return {"success": "database data fully embedded to vectordb"}
    




# ********************* ----------------------------------------------- *************************************
###### QUERY AND RETRIEVAL ######

## 4- FETCH QUERY FROM REDIS CACHE, IF NOT FOUND ONLY THEN DO A VECTOR RETRIEVAL FROM VECTORDB

def query_redis_cache_then_vecotrdb_if_no_cache(table_name: str, query: str, score: float, top_n: int) -> List[Dict[str,Any]] | str | Dict[str,str]:

  try:
    response = handle_query_by_calling_cache_then_vectordb_if_fail(table_name, query, score, top_n)
    print("RESPONSE CACHE OR VDB: ", json.dumps(response, indent=4))
    
    if response:


      # here the redis response from hashed query matching
      if "exact_match_search_response_from_cache" in response: 
        exact_match_response = response["exact_match_search_response_from_cache"]
        # List[Dict[str,Any]]
        StateCustom.query_hash_retrieved = exact_match_response
        return exact_match_response

      # here the redis response from semantic search on stored embeddings as no hashed query matched to get an answer
      elif "semantic_search_response_from_cache" in response: 
        semantic_response = response["semantic_search_response_from_cache"]
        # List[Dict[str,Any]]
        StateCustom.query_vector_hash_retrieved = semantic_response
        return semantic_response

      # here the vectordb retrieved answer and also cached in redis for next search as for this one nothign was found in cache
      elif "vector_search_response_after_cache_failed_to_find" in response:
        vector_response = response["vector_search_response_after_cache_failed_to_find"]
        # List[Dict[str,Any]]
        StateCustom.vectorbd_retrieved = vector_response
        return vector_response

      # here a message that suggest to perform internet search on the query as nothing has been found in redis and vectordb
      elif "message" in response:
        print(response["message"])
        # str
        StateCustom.nothing_retrieved = True
        return response["message"]

      # Here to catch any errors
      elif "error" in response:
        raise Exception(response["error"])
    
  except Exception as e:
    # Dict[str,str]
    return {"error": f"An error occured while trying to handle query by calling cache then vectordb if nothing found: {e}"}

"""

"""
"""
    # here this means that the cache search and vector search did fail to find relevant information
    # we need then to do an internet search using another node or to have this in a conditional eadge
"""

###### INTERNET SEARCH TOOL ######
## -5 PERFORM INTERNET SEARCH IF NO ANSWER FOUND IN REDIS OR PGVECTOR

# function that check states to know if we perform internet search, should handle all the cases in which internet search can be called

# will be used as `llm_with_internet_search_tool(query)`
llm_with_internet_search_tool = groq_llm_mixtral_7b.bind_tools(internet)
# OR `internet_tool_node = ToolNode(internet)`

# search through internet and get 5 search results
def internet_research_user_query(state: MessagesState):

    query = state["messages"][-1].content  # Extract the last user query
    
    # Fill the prompt template
    generate_from_empty_prompt["system"]["template"] = "You are an expert search engine."
    generate_from_empty_prompt["human"]["template"] = "Find the most relevant information about: {query}"
    generate_from_empty_prompt["human"]["input_variables"] = {"query": query}
    
    # Construct the prompt for the LLM
    system_message = generate_from_empty_prompt["system"]["template"]
    human_message = generate_from_empty_prompt["human"]["template"].format(**generate_from_empty_prompt["human"]["input_variables"])
    ai_message = generate_from_empty_prompt["ai"]["template"]
    
    # Use the tool with the constructed prompt
    search_results = llm_with_internet_search_tool([system_message, human_message, ai_message])
    print("Result Search: ", search_results)
    
    # Process the search results (e.g., select the top result or combine summaries)
    summary = search_results[:5]  # Take the top 5 results
    
    # Return the formatted message
    return {"messages": [{"role": "assistant", "content": "\n".join(summary)}]}

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
                                                                   --- Save query/internet_answer to DB
                                                                       --- Embed query/internet_answer
                                                                           --- answer user query and cache it
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
## 10- CREATE CONFIGS FOR GRAPH INTERRUPTION OR OTHER NEEDS












