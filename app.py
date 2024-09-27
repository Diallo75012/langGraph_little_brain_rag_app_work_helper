import os
import sys
# for typing func parameters and outputs and states
from typing import Dict, List, Tuple, Any, Optional

# for graph creation and management
from langgraph.graph import MessagesState
#### MODULES #####
# DB AND VECTORDB EMBEDDINGS AND RETRIEVAL
from lib_helpers.embedding_and_retrieval import (
  create_table_if_not_exists,
  connect_db
)
# STATES Persistence between graphs
from app_states.app_graph_states import GraphStatePersistFlow
# to run next graphs
from graphs.embedding_subgraph import embedding_subgraph
from graphs.retrieval_subgraph import retrieval_subgraph
from graphs.report_creation_subgraph import report_creation_subgraph
from graphs.primary_graph import primary_graph
from graphs.code_execution_graph import code_execution_graph
from llms.llms import (
  groq_llm_mixtral_7b,
  groq_llm_llama3_8b,
  groq_llm_llama3_8b_tool_use,
  groq_llm_llama3_70b,
  groq_llm_llama3_70b_tool_use,
  groq_llm_gemma_7b,
)
# structured output
from structured_output.structured_output import (
  # class created to structure the output
  ReportAnswerCreationClass,
  # function taking in class output structured and the query, returns dict
  structured_output_for_agent
)
# for env. vars
from dotenv import load_dotenv


# load env vars
load_dotenv(dotenv_path='.env', override=False)
load_dotenv(dotenv_path=".vars.env", override=True)

# Connect to the PostgreSQL database (all imported from `embedding_and_retrieval`)
conn = connect_db()
create_table_if_not_exists(os.getenv("TABLE_NAME"))

#########################################################################################


if __name__ == "__main__":
  # I want to know if this documents docs/feel_temperature.pdf tells us what are the different types of thermoceptors?
  '''
    We will use raise Exception(...) but if having a Webui we can see how to handle it
  '''
  custom_state = GraphStatePersistFlow()
  embedding_flow = ""
  retrieval_flow = ""
  # get user query
  user_query =  input("How can I help you today? ").strip()
  # save query to sdtate
  custom_state.user_initial_query = user_query
  print("custom graph user query field value: ", custom_state.user_initial_query)

  ###################################### PDF or URL RAG DOCUMENT ANALYSIS AND REPORT CREATION #########################################
  """
    QUERY ANALYSIS FOR CODE EXECUTION: DOCUMENTATION CREATION > SCRIPT CREATION > CODE EXECUTION IN DOCKER  or STARTING NEXT GRAPH
  """
  # start code execution graph
  print("Starting Code Execution Graph")
  try:
    code_execution_flow =   code_execution_graph(custom_state.user_initial_query)
    if "error" in code_execution_flow.lower():
      raise Exception(f"An error occured while running 'code_execution_flow'")
  except Exception as e:
    raise Exception(f"An error occured while running 'code_execution_flow': {e}")

  # stop the graph here if it was just code generation and documentation creation only otherwise the next graph will be started
  if os.getenv("DOCUMENTATION_AND_CODE_GENERATION_ONLY") == "true" and code_execution_flow == "done":
    # see here if we print all the env vars having where the files can be found
    print("Code execution done in safe docker sandbox environment, documentation have been created.")
    sys.exit()

  ###################################### PDF or URL RAG DOCUMENT ANALYSIS AND REPORT CREATION #########################################
  """
    QUERY ANALYSIS: REPORT DIRECTLY OR RAG > CACHE > REPORT
  """
  # start primary graph
  print("Starting Primary Graph")
  try:
    # if the code_execution_graph returns the signal to start the query analysis we start the flow of graphs
    if code_execution_flow == "primary_flow" and os.getenv("REPORT_NEEDED") == "true":
      primary_flow =  primary_graph(os.getenv("USER_INITIAL_QUERY"))
      if "error" in primary_flow.lower():
        raise Exception(f"An error occured while running 'primary_flow'")
  except Exception as e:
    raise Exception(f"An error occured while running 'primary_flow': {e}")

  # catch here the answer directly and stop the graph if no documents have been provided and returnt the path of the report path where the answer will be stored
  if os.getenv("PRIMARY_GRAPH_NO_DOCUMENTS_NOR_URL") == "true" and primary_flow == "done":
    print(f"No PDF or URL as been given. Please find the response at: {os.getenv('REPORT_PATH')}")
  

  """
    EMBEDDINGS
  """
  # Get primary graph parquet path fromt he .var env file
  load_dotenv(dotenv_path=".vars.env", override=True)
  parquet_file_path = os.getenv("PARQUET_FILE_PATH")
  print("1: ", parquet_file_path)

  # embedding graph
  if primary_flow == "embeddings" and parquet_file_path:
    print("Starting Embedding Graph")
    try:
      embedding_flow = embedding_subgraph(parquet_file_path)
      if "error" in embedding_flow.lower():
        raise Exception(f"An error occured while running 'embedding_flow'")
    except Exception as e:
      raise Exception("An error occured while running 'embedding_flow': {e}")



  embedding_graph_result = os.getenv("EMBEDDING_GRAPH_RESULT")
  print("2: ", embedding_graph_result)

  """
    RETRIEVAL
  """
  # Get the reformulated initial user query for the retrieval graph
  load_dotenv(dotenv_path=".vars.env", override=True)
  reformulated_query = os.getenv("QUERY_REFORMULATED")

  # retrieval graph
  """
  just for the test embedding_flow = "retrieval" don't forget to get rid of it when it works fine
  embedding_flow = "retrieval"
  """

  if embedding_flow == "retrieval" and "success" in embedding_graph_result:
    print("Starting Retrieval")
    try:
      retrieval_flow = retrieval_subgraph(reformulated_query)
      if "error" in retrieval_flow.lower():
        raise Exception(f"An error occured while running 'retrieval_flow'")
    except Exception as e:
      raise Exception(f"An error occured while running 'retrieval_flow': {e}")

  retrieval_graph_result = os.getenv("RETRIEVAL_GRAPH_RESULT")
  print("2: ", embedding_graph_result)

  """
    REPORT
  """
  # Get the reformulated initial user query for the retrieval graph
  load_dotenv(dotenv_path=".vars.env", override=True)
  reformulated_query = os.getenv("QUERY_REFORMULATED")

  # report creation graph
  if retrieval_flow == "report" and "success" in retrieval_graph_result:
    print("Starting Report Creation")
    try:
      retrieval_graph_result = retrieval_graph_result.split(":")[-1].strip()
      report_creation_flow = report_creation_subgraph(retrieval_graph_result)
      if "error" in report_creation_flow.lower():
        raise Exception(f"An error occured while running 'retrieval_flow': {report_creation_flow.lower()}")
    except Exception as e:
      raise Exception(f"An error occured while running 'retrieval_flow': {e}")
  elif retrieval_graph_result:
    if "report" in retrieval_graph_result:
      report_path = retrieval_graph_result.split(":")[-1].strip()
      with open(report_path, "r", encoding="utf-8") as report:
        report_content = report.read()
      print(f"Report content (markdown) present at {os.getenv('REPORT_PATH')}: \n", report_content)

  '''
  create here condition to get the report graph starting if no code execution is needed in user query and there is no link address or pdf documents
  '''



