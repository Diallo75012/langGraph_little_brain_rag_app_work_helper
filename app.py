import os
from typing import Annotated, Literal, TypedDict
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.checkpoint import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
import psycopg2


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
print("Tools: ", tools)

tool_node = ToolNode(tools)
print("tool_node: ", tool_node)

model = groq_llm_llama3_70b.bind_tools(tools)
print("Model with bind_tools: ", model)


# Define the function that determines whether to continue or not
def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    print("messages from should_continue func: ", messages)
    last_message = messages[-1]
    print("last message from should_continue func: ", last_message)
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        print("Tool called!")
        return "tools"
    # Otherwise, we stop (reply to the user)
    print("Tool not called returning answer to user.")
    return END


# Define the function that calls the model
def call_model(state: MessagesState):
    messages = state['messages']
    print("messages from call_model func: ", messages)
    response = model.invoke(messages)
    print("response from should_continue func: ", response)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define a new graph
workflow = StateGraph(MessagesState)
print("workflow: ", workflow)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
print("Workflow add node 'agent': ", workflow)
workflow.add_node("tools", tool_node)
print("Workflow add node 'tools': ", workflow)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent")
print("Workflow set entry point 'agent': ", workflow)

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)
print("Workflow add conditional edge 'agent' -> should_continue func: ", workflow)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("tools", 'agent')
print("Workflow add edge 'tools' -> 'agent': ", workflow)

# Initialize memory to persist state between graph runs
checkpointer = MemorySaver()
print("MEmory checkpointer: ", checkpointer)

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable.
# Note that we're (optionally) passing the memory when compiling the graph
app = workflow.compile(checkpointer=checkpointer)
print("App compiled with checkpointer: ", app)

# Use the Runnable
final_state = app.invoke(
    {"messages": [HumanMessage(content="what is the weather in sf")]},
    config={"configurable": {"thread_id": 42}}
)
print("Final State = answer: ", final_state)

final_state["messages"][-1].content
print("Final state last message content: ", final_state["messages"][-1].content)


#########################################################################################
"""
### LOGIC FOR THE MOMENT

# workflow:

 - after that we are ready to use here the  library `query_matching` to get user/llm query and perform cahce search first and then vector search if not found in cache
  `from query_matching import *`
 - need to implement a tool internet search in the agent logic to perform internet search only if this fails
  `Use agent prompting and conditional edge of langgraph to direct the llm to use internet search if didn't found information or if needed extra information because of info poorness`
 - better use lots of mini tools than one huge and let the llm check. We will just prompting the agent to tell it what is the workflow. So revise functions here to be decomposed in mini tools
   `list of tools to create to be planned and helps for nodes creation as well so it is a mix of tools and nodes definition titles for the moment: [internet_search_tool, redis_cache_search_tool, embedding_vectordb_search_tool, user_query_analyzer_rephrasing_tool, node_use_tool_or_not_if_not_answer_query, save_initial_query_an_end_outcome_to_key_value_db_and_create_redis_cache_with_long_term_ttl, node_judge_answer_for_new_iteration_or_not]`
 - then build all those mini tools

"""
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
  # here we get List[Dict[str,any]]
  try:  
    rows = fetch_documents()
  except Exception as e:
    conn.close()  
    return {"error": f"An error occured while trying to fetch rows from db -> {e}"}
  # here we get List[List[Dict[str,any]]]
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
"""
Here will need to granularly check how the objects are fetched from cache to be able to extra the data that we want
"""
def query_redis_cache_then_vecotrdb_if_no_cache(query: str, score: float) -> dict:
  response = handle_query_by_calling_cache_then_vectordb_if_fail(query, score)
  print(json.dumps(response, indent=4))
  return response
