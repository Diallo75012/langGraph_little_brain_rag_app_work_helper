import os
import json
# For str to dict
import ast
import re
# for dataframe
import pandas as pd
from uuid import uuid4
# postgresql
import psycopg2
from psycopg2 import sql
# for typing func parameters and outputs and states
from typing import Dict, List, Tuple, Any, Optional
# for llm call with func or tool and prompts formatting
from langchain_groq import ChatGroq
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
# for graph creation and management
from langgraph.graph import MessagesState
#### MODULES #####
# PROMPTS
from prompts.prompts import (
  detect_content_type_prompt,
  summarize_text_prompt,
  generate_title_prompt,
  rewrite_or_create_api_code_script_prompt
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
# DB AND VECTORDB EMBEDDINGS AND RETRIEVAL
from lib_helpers.embedding_and_retrieval import (
  # returns `None\dict`
  embed_all_db_documents,
  # returns `List[Dict[str, Any]]`
  fetch_documents,
  CONNECTION_STRING,
  create_table_if_not_exists,
  connect_db
)
# to run next graphs
from llms.llms import (
  groq_llm_mixtral_7b,
  groq_llm_llama3_8b,
  groq_llm_llama3_8b_tool_use,
  groq_llm_llama3_70b,
  groq_llm_llama3_70b_tool_use,
  groq_llm_gemma_7b,
)
# for env. vars
from dotenv import load_dotenv, set_key


# load env vars
load_dotenv(dotenv_path='.env', override=False)
load_dotenv(dotenv_path=".vars", override=True)

# pggVector
collection_name = os.getenv("COLLECTION_NAME")
connection_string = CONNECTION_STRING

# Connect to the PostgreSQL database (all imported from `embedding_and_retrieval`)
conn = connect_db()
create_table_if_not_exists(os.getenv("TABLE_NAME"))

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
# creation of prompts
def prompt_creation(target_prompt_human_system_or_ai: Dict[str, Any], **kwargs: Any) -> str: #-> PromptTemplate:
    input_variables = target_prompt_human_system_or_ai.get("input_variables", [])

    prompt = PromptTemplate(
        template=target_prompt_human_system_or_ai["template"],
        input_variables=input_variables
    )

    formatted_template = prompt.format(**kwargs) if input_variables else target_prompt_human_system_or_ai["template"]
    print("formatted_template: ", formatted_template)
    return formatted_template
    #return PromptTemplate(
    #    template=formatted_template,
    #    input_variables=[]
    #)

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
    df = pd.DataFrame(chunks)#[0:3]
    print("DF: ", df.head())
  else:
    # Parse the PDF and create a DataFrame
    pdf_data = pdf_to_sections(url_or_doc_path)
    print("PDF DATA: ", pdf_data)
    df = pd.DataFrame(pdf_data)
    print("DF: ", df.head())

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
  system_message = SystemMessage(content=answer_user_with_report_from_retrieved_data_prompt["system"]["template"])
  human_message = HumanMessage(content=prompt_human_with_input_vars_filled)
  messages = [system_message, human_message]
  response = groq_llm_mixtral_7b.invoke(messages)
  #formatted_answer = response.content.split("```")[1].strip("markdown").strip()

  # couldn't get llm to answer in markdown tags???? so just getting content saved to states
  return {"messages": [{"role": "ai", "content": response.content}]}




# ********************* ----------------------------------------------- *************************************
##### QUERY & EMBEDDINGS #####

## 1- IDENTIFY IF QUERY IS FOR A PFD OR A WEB PAGE
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
    maximum_content_length: int = int(os.getenv("MAXIMUM_CONTENT_LENGTH"))
    maximum_title_length: int = int(os.getenv("MAXIMUM_TITLE_LENGTH"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE_FOR_DB"))
    detect_prompt_template = detect_content_type_prompt
    text_summary_prompt = summarize_text_prompt
    title_summary_prompt = generate_title_prompt

    print("Query: ", query)

    # Default values in the function
    llm = groq_llm_mixtral_7b

    # use of detect_content_type from `query_analyzer_module`
    content_str = detect_content_type(llm, query, detect_prompt_template)
    content_dict = string_to_dict(content_str)

    # update intermediary state as we need query for retrieval
    if content_dict["question"]:
      set_key(".vars.env", "QUERY_REFORMULATED", content_dict["question"])
      load_dotenv(dotenv_path=".vars.env", override=True)
    else:
      set_key(".vars.env", "QUERY_REFORMULATED", content_dict["text"])
      load_dotenv(dotenv_path=".vars.env", override=True)

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
  print("DOC NAME: ", df_final['doc_name'][0])
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
        WHERE doc_name = %s;
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
          WHERE doc_name = %s AND content = %s;
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
  chunk_size: int = int(os.getenv("CHUNK_SIZE_FOR_EMBEDDINGS"))
  collection_name: str = os.getenv("COLLECTION_NAME")
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


## JDUGE DOCUMENTATION WRITTEN BY AGENT TO MAKE API CALL
# The function will decide if we need to rewrite the documentation or if we can start writing the code, then the code will be sent to different llms to create scripts in parallel using conditional edges that can run different branches together
def rewrite_or_create_api_code_script(state: MessagesState, apis: Dict[str,str]) -> str:
  """
    Here we will judge if the agent written documentation is valid or need to be re-written.
    If the documentation is valid, we will then ask different agents to write scripts corresponding to those instructions
  """
  # vars
  # documentation written by agent
  messages = state["messages"]
  last_message = messages[-1].content
  # user inital query
  user_initial_query = os.getenv("USER_INITIAL_QUERY")
  # api chosen to satisfy query
  api_choice = os.getenv("API_CHOICE")
  # links for api calls for each apis existing in our choices of apis
  apis_links = apis
  '''
    Here check if we can use structures output format to access what we need
    - decision: rewrite or generate
    - reason: reason why you have taken that decision
    - stage: the stage to return to if decision is rewrite: internet or rewrite
  '''
  prompt = rewrite_or_create_api_code_script_prompt["human"]["template"]
  prompt_human_with_input_vars_filled = eval(f'f"""{prompt}"""')
  print(f"\nprompt_human_with_input_vars_filled: {prompt_human_with_input_vars_filled}\n")
  system_message = SystemMessage(content=rewrite_or_create_api_code_script_prompt["system"]["template"])
  human_message = HumanMessage(content=prompt_human_with_input_vars_filled)
  messages = [system_message, human_message]
  response = groq_llm_mixtral_7b.invoke(messages)
  if response.content["decision"] == "rewrite":
    return {"messages": [{"role": "ai", "content": f"disagree:rewrite,{response.content['reason']}"}]} # use spliting technique to get what is wanted 
  elif response.content["decision"] == "generate":
    return {"messages": [{"role": "ai", "content": f"success:generate"}]}
  else:
    return {"messages": [{"role": "ai", "content": f"error:{response.content}"}]}

  
























