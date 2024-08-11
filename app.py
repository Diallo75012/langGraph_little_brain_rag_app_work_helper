import os
import time
from typing import Annotated, Literal, TypedDict
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import (
  PromptTemplate,
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate,
  AIMessagePromptTemplate
)
from langchain_core.output_parsers import JsonOutputParser
from langgraph.checkpoint import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
import psycopg2
from typing import TypedDict, Dict, List, Any, Optional
#### MODULES #####
from prompts import prompts
# USER QUERY ANALYSIS AND TRANSFORMATION
from lib_helpers.query_analyzer_module import *
# INITIAL DOC PARSER, STORER IN POSTGRESL DB TABLE
from lib_helpers.pdf_parser import *
from lib_helpers.webpage_parser import *
# CUSTOM CHUNKING
from lib_helpers.chunking_module import *
# REDIS CACHE RETRIEVER
from lib_helpers.query_matching import *
# VECTORDB EMBEDDINGS AND RETRIEVAL
from lib_helpers.embedding_and_retrieval import *


# load env vars
load_dotenv(dotenv_path='.env', override=False)

groq_llm_mixtral_7b = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_MIXTRAL_7B"),
max_tokens=int(os.getenv("GROQ_MAX_TOKEN")),)
groq_llm_llama3_8b = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_LLAMA3_8B"), max_tokens=int(os.getenv("GROQ_MAX_TOKEN")),)
groq_llm_llama3_8b_tool_use = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_LLAMA3_8B_TOOL_USE"), max_tokens=int(os.getenv("GROQ_MAX_TOKEN")),)
groq_llm_llama3_70b = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_LLAMA3_70B"), max_tokens=int(os.getenv("GROQ_MAX_TOKEN")),)
groq_llm_llama3_70b_tool_use = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_LLAMA3_70B_TOOL_USE"), max_tokens=int(os.getenv("GROQ_MAX_TOKEN")),)
groq_llm_gemma_7b = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_GEMMA_7B"), max_tokens=int(os.getenv("GROQ_MAX_TOKEN")),)


# Connect to the PostgreSQL database
def connect_db() -> psycopg2.extensions.connection:
    return psycopg2.connect(
        database=os.getenv("DATABASE"), 
        user=os.getenv("USER"), 
        password=os.getenv("PASSWORD"),
        host=os.getenv("HOST"), 
        port=os.getenv("PORT")
    )

def create_table_if_not_exists():
    """Create the documents table if it doesn't exist."""
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id UUID PRIMARY KEY,
            doc_name TEXT,
            title TEXT,
            content TEXT,
            retrieved BOOLEAN
        );
    """)
    conn.commit()
    cursor.close()
    conn.close()

create_table_if_not_exists()



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


"""def analyze_user_query():"""
##### QUERY & EMBEDDINGS #####
## 1- IDENTIFY IF QUERY IS FOR A PFD OR A WEB PAGE
# Main function to process the user query and return pandas dataframe
def process_query(query: str) -> None:
    # use of detect_content_type from `query_analyzer_module`
    content_type, content = detect_content_type(query)
    # print content to see how it looks like, will it getthe filename if pdf or the full correct url if url...
    print("content: ", content)
    if content_type == 'pdf':
        # functions from `lib_helpers.pdf_parser`
        df_final = get_final_df(content)
    elif content_type == 'url':
        # functions from `lib_helpers.webpage_parser`
        df_final = get_df_final(content)
    else:
        return f"No PDF nor URL found in the query. Content: {content}"
    
    return df_final

from query_analyzer_module, to get response object, maybe better than detect_content_type(query) other function from same library:
{
  'url': <url in user query make sure that it always has https. and get url from the query even if user omits the http>,
  'pdf': <pdf full name or path in user query>,
  'text': <if text only without any webpage url or pdf file .pdf path or file just put here the user query>,
  'question': <if quesiton identified in user query rephrase it in an easy way to retrieve data from vector database search without the filename and in a good english grammatical way to ask question>
}
function to be improved as it should use a template to call llm and other utility functions and accept llm as parameter so that we can change it on the fly
llm_call(query: str, prompt_llm_call: List[Tuple[str,str]])

## 2- STORE DOCUMENT DATA TO POSTGRESQL DATABASE
# Function to store cleaned data in PostgreSQL
def store_data_to_postgresql(df_final, conn):
    conn = connect_db()
    cursor = conn.cursor()
    for _, row in df.iterrows():
        cursor.execute(
            "INSERT INTO documents (id, doc_name, title, content, retrieved) VALUES (%s, %s, %s, %s, %s)",
            (row['id'], row['doc_name'], row['title'], row['content'], row['retrieved'])
        )
    conn.commit()
    cursor.close()
    conn.close()
    
    return "document quality data saved to postgresql database table"

## 3- EMBED ALL DATABASE SAVED DOC DATA TO VECTORDB
# `COLLECTION_NAME` and `CONNECTION_STRING` should be recognized as it comes from 'lib_helpers.embedding_and_retrieval'
def custom_chunk_and_embed_to_vectordb(chunk_size: int, COLLECTION_NAME: str, CONNECTION_STRING: str) -> dict:
  # Embed all documents in the database
  # function from module `lib_helpers.embedding_and_retrieval`
  conn = connect_db()
  # here we get List[Dict[str,Any]]
  try:  
    rows = fetch_documents()
  except Exception as e:
    conn.close()  
    return {"error": f"An error occured while trying to fetch rows from db -> {e}"}
  # here we get List[List[Dict[str,Any]]]
  try:
    chunks = create_chunks(rows, chunk_size)
  except Exception as e:
    conn.close()
    return {"error": f"An error occured while trying to create chunks -> {e}"}
  # here we create the custom document and embed it
  try:
    for chunk_list in chunks:
      embed_all_db_documents(chunk_list, COLLECTION_NAME, CONNECTION_STRING) 
  except Exception as e:
    conn.close()
    return {"error": f"An error occured while trying to create custom docs and embed it -> {e}"}
  
  conn.close()    
  return {"success": "database data fully embedded to vectordb"}
    

###### QUERY AND RETRIEVAL ######
## 4- FETCH QUERY FROM REDIS CACHE, IF NOT FOUND ONLY THEN DO A VECTOR RETRIEVAL FROM VECTORDB

def query_redis_cache_then_vecotrdb_if_no_cache(query: str, score: float) -> List[Dict[str,Any]] | str:
  response = handle_query_by_calling_cache_then_vectordb_if_fail(query, score)
  print(json.dumps(response, indent=4))
  if response["exact_match_search_response_from_cache"]: 
    exact_match_response = json.loads(response["exact_match_search_response_from_cache"])
    return exact_match_response
  elif response["semantic_search_response_from_cache"]: 
    semantic_response = json.loads(response["semantic_search_response_from_cache"])
    return semantic_response
  elif response["vector_search_response_after_cache_failed_to_find"]:
    vector_response = json.loads(response["vector_search_response_after_cache_failed_to_find"])
    return vector_response
  elif response["message"]:
    print(response["message"])
    return response["message"]
    """
    # here this means that the cache search and vector search did fail to find relevant information
    # we need then to do an internet search using another node or to have this in a conditional eadge
    """

###### INTERNET SEARCH TOOL ######
## -5 PERFORM INTERNET SEARCH IF NO ANSWER FOUND IN REDIS OR PGVECTOR
from langchain_core.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults # Run vs Results: Results have more information 
internet_search_tool = DuckDuckGoSearchRun()
tool_internet = Tool(
    name="duckduckgo_search",
    description="Search DuckDuckGO for recent results.",
    func=internet_search_tool.run,
)
internet = [tool_internet]
llm_with_internet_searhc_tool = groq_llm_mixtral_7b.bind_tools(internet)

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
    return response.content.split("```")[1].strip("python").strip()
    
  # special for chat system/human/ai
  elif prompt and prompt_chat_input_vars:
    print("Chat input variables found: ", prompt_chat_input_vars)
    chain = ( prompt | model )
    response = chain.invoke(chat_input_vars_dict)
    print("Response: ", response, type(response))
    return response.content.split("```")[1].strip("python").strip()

  print("Chat input variables NOT found or missing prompt!")
  chain = ( prompt | model )
  response = chain.invoke(input={})
  print("Response: ", response, type(response))
  return response.content.split("```")[1].strip("python").strip()

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

print("SIBERIA: \n", make_normal_or_chat_prompt_chain_call(groq_llm_llama3_8b, test_prompt_siberia["input_variables"], test_prompt_siberia["template"], {}))
time.sleep(0.5)
print("TOKYO: \n", make_normal_or_chat_prompt_chain_call(groq_llm_llama3_8b, test_prompt_tokyo["input_variables"], test_prompt_tokyo["template"], {}))
time.sleep(0.5)
print("DAKAR: \n", make_normal_or_chat_prompt_chain_call(groq_llm_llama3_8b, {}, {}, test_prompt_dakar))
time.sleep(0.5)
print("MONACO: \n", make_normal_or_chat_prompt_chain_call(groq_llm_llama3_8b, {}, {}, test_prompt_monaco))



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












