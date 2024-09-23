import os
import json
# for typing func parameters and outputs and states
from typing import Dict, List, Tuple, Any, Optional
# LLM chat AI, Human, System
from langchain_core.messages import (
  AIMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage
)
# structured output
from structured_output.structured_output import (
  # class for document quality judgement
  CodeDocumentionEvaluation,
  # function for documentation evaluation
  structured_output_for_agent_doc_evaluator
)
from prompts.prompts import structured_outpout_report_prompt
from typing import Dict, List, Any, Optional, Union
# Prompts LAngchain and Custom
from langchain_core.prompts import PromptTemplate
# Node Functions from App.py 
"""
Maybe will have to move those functions OR to a file having all node functions OR to the corresponding graph directly
"""
# utils
from app_utils import creation_prompt
# prompts
from prompts.prompts import rewrite_or_create_api_code_script_prompt
# Tools
from app_tools.app_tools import (
  # internet node & internet llm binded tool
  tool_search_node,
  llm_with_internet_search_tool,
  tool_agent_decide_which_api_node,
  llm_api_call_tool_choice
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

apis = {
  "joke": "https://official-joke-api.appspot.com/random_joke",
  "agify": "https://api.agify.io?name=[name]",
  "dogimage": "https://dog.ceo/api/breeds/image/random",
}

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

# api selection
def tool_api_choose_agent(state: MessagesState):
    messages = state['messages']
    print("message state -1: ", messages[-1].content, "\nmessages state -2: ", messages[-2].content)
    # print("messages from call_model func: ", messages)
    response = llm_api_call_tool_choice.invoke(messages[-1].content)
    return {"messages": [response]}

# NODE FUNCTIONS
def get_user_input(state: MessagesState):
  messages = state['messages']
  last_message = messages[-1].content
  set_key(".var.env", "USER_INITIAL_QUERY", last_message)
  load_dotenv(dotenv_path=".vars.env", override=True)
  return {"messages": [last_message]}

def error_handler(state: MessagesState):
  messages = state['messages']
  last_message = messages[-1].content
  
  with open("./logs/conditional_edge_logs.log", "a", encoding="utf-8") as conditional:
      conditional.write(f"\n\nerror handler called: {last_message}\n\n")

  return {"messages": [{"role": "ai", "content": f"An error occured, error message: {last_message}"}]}

def find_documentation_online_agent(APIs: Dict[str,str] = apis, state: MessagesState):
    messages = state['messages']
    # should be the tool that have been picked by the agent to satisfy user request
    last_message = messages[-1].content
    # create environment variable with the API that have been chosen for next nodes to have access to it if needed
    sek_key(".var.env", "API_CHOICE", last_message)
    load_dotenv(dotenv_path=".vars.env", override=True)
    print("message state -1: ", messages[-1].content)
    user_initial_query = os.getenv("USER_INITIAL_QUERY")
    prompt = f"We have to find online how to make a Python script to make a simple API call and get the response in mardown to this: {last_message}. Here is the different APIs urls that we have: {APIs}. Select just the one corresponding accourdingly to user intent: {user_initial_query}. And search how to make a Python script to call that URL and return the response using Python."
    response = llm_with_internet_search_tool.invoke(prompt)
    return {"messages": [response]}

def documentation_writer(apis: Dict[str,str] = apis, state: MessagesState):
  messages = state['messages']
  
  #### VARS ####
  
  # this message should be the internet search result telling how to create the code to make api call to the chosen api
  last_message = messages[-1].content
  # save last_message which is the internet search result so that we can access it from other nodes in case of a loop back in a future node so not coming back here
  sek_key(".var.env", "DOCUMENTATION_FOUND_ONLINE", last_message)
  load_dotenv(dotenv_path=".vars.env", override=True)
  # online doc found
  documentation_found_online = os.getenv("DOCUMENTATION_FOUND_ONLINE")
  # agent choosen api to satisfy user api call code generation
  api_choice = os.getenv("API_CHOICE")
  # user initial query
  user_initial_query = os.getenv("USER_INITIAL_QUERY")
  
  llm = groq_llm_mixtral_7b
  prompt = PromptTemplate.from_template("{query}")
  chain = prompt | llm
  '''
    See if here we can use structured output and also save this prompt in the prompt file and use the python `eval()` to fill the fields or langchain `prompt_template.from_template`
  '''
  result = chain.invoke({"query": f"User wanted a Python script in markdown to call an API: {user_initial_query}. Our agent chosen this api to satisfy user request: {api_choice}; and found some documentation online: <doc>{documentation_found_online}</doc>. Can you write in markdown format detailed documentation in how to write the script that will call the API chosen by user which you can get the reference from: <api links>{apis}</api links>. Calling and returning the formatted response. We need instruction like documentation so that llm agent called will understand and provide the code. So just ouput the documentation with all steps for Python developer to understand how to write the script. Therefore, DO NOT write the script, just the documentation and guidance in how to do it in markdown format."})
  
  # write the documentation to a file
  with open(os.getenv("CODE_DOCUMENTATION_FILE_PAH"), "w", encoding="utf-8") as code_doc:
    code_doc.write("""
      '''
      This file contains the documentation in how to write a mardown Python script to make API to satisfy user request.
      Make sure that the script is having the right imports, indentations and do not have any error before starting writing it.
      '''
    """\n\n)
    code_doc.write(result.content)
  
  return {"messages": [result.content]}

def llama_3_8b_script_creator(state: MessagesState):
  messages = state['messages']
  last_message = messages[-1].content
  
  system_message = SystemMessage(content=last_message)
  human_message = HumanMessage(content="Create a Python script to call the API following the instructions. Make sure that it is in markdown format and have the right indentations and imports.")
  messages = [system_message, human_message]
  response = groq_llm_llama3_8b.invoke(messages)  
  
  return {"messages": [{"role": "ai", "content": f"llama_3_8b:{response.content}"}]}

def llama_3_70b_script_creator(state: MessagesState):
  messages = state['messages']
  last_message = messages[-1].content
  
  system_message = SystemMessage(content=last_message)
  human_message = HumanMessage(content="Create a Python script to call the API following the instructions. Make sure that it is in markdown format and have the right indentations and imports.")
  messages = [system_message, human_message]
  response = groq_llm_llama3_70b.invoke(messages)  
  
  return {"messages": [{"role": "ai", "content": f"llama_3_70b:{response.content}"}]}

def gemma_3_7b_script_creator(state: MessagesState):
  messages = state['messages']
  last_message = messages[-1].content
  
  system_message = SystemMessage(content=last_message)
  human_message = HumanMessage(content="Create a Python script to call the API following the instructions. Make sure that it is in markdown format and have the right indentations and imports.")
  messages = [system_message, human_message]
  response = groq_llm_gemma_7b.invoke(messages)  
  
  return {"messages": [{"role": "ai", "content": f"gemma_3_7b:{response.content}"}]}

# judging documentation created agent
def documentation_steps_evaluator_and_doc_judge(state: MessagesState):
  """
    Will jodge the documentation and return 'rewrite' to rewrite the documentation or 'generate' to start generating the script
  """
  # documentation written by agent
  messages = state["messages"]
  documentation = messages[-1].content
  # user inital query
  user_initial_query = os.getenv("USER_INITIAL_QUERY")
  # api chosen to satisfy query
  api_choice = os.getenv("API_CHOICE")
  # links for api calls for each apis existing in our choices of apis
  apis_links = apis
  
  # we need to fill this template with the input variables to create the query form 'human' side of the rpompt template
  query = prompt_creation(rewrite_or_create_api_code_script_prompt["human"], documentation=documentation, user_initial_query=user_initial_query, api_choice=api_choice, apis_links=apis_links)
  # get the answer as we want it using structured output
  try:
    decision = structured_output_for_agent(ReportAnswerCreationClass, query, rewrite_or_create_api_code_script_prompt["system"]["template"])
    print("\n\n\nDECISION: ", decision, type(decision))
  """
  # Object returned by the structured output llm call
  { 
    "DECISION": response.decision,
    "REASON": response.reason,
    "STAGE": response.stage,
  }
    """
    if decision.content["DECISION"] == "rewrite":
      return {"messages": [{"role": "ai", "content": json.dumps({"disagree":decision.content})}]} # use spliting technique to get what is wanted 
    elif decision.content["DECISION"] == "generate":
      return {"messages": [{"role": "ai", "content": json.dumps({"success":"generate"})}]}
    else:
      return {"messages": [{"role": "ai", "content": json.dumps({"error":decision.content})}]}
  except Exception as e:
    return {"messages": [{"role": "ai", "content": json.dumps({"error":e})}]} 

def code_evaluator_and_final_script_writer(state: MessagesState):
    """
    This node receives code from different LLMs, evaluates if it's valid by calling another LLM, 
    and returns 'YES' or 'NO' for validation using structured outputs.

    Args:
    state: MessagesState object containing the messages and codes generated by different LLMs.

    Returns:
    A dictionary containing the message response indicating whether the code is valid ('YES') or not ('NO').
    """

    # vars
    # user inital query
    user_initial_query = os.getenv("USER_INITIAL_QUERY")
    # api chosen to satisfy query
    api_choice = os.getenv("API_CHOICE")
    # links for api calls for each apis existing in our choices of apis
    apis_links = apis
   
    # Extract the last few messages, assuming they are code outputs from different LLMs
    messages = state['messages']
    
    # dictionary to have code linked to each llm for easier parsing and allocation and looping through it and formatting answers
    dict_llm_codes = {}
    
    # Assume the last three messages are the code outputs
    codes = [ messages[-1].content, messages[-2].content, messages[-3].content]
    for code in codes:
      llm_name, llm_code = code.split(":")[0], code.split(":")[1]
      if llm_name.strip() == "llama_3_8b":
        dict_llm_code[llm_name.strip()] == llm_code.strip()
      elif llm_name.strip() == "llama_3_70b":
        dict_llm_code[llm_name.strip()] == llm_code.strip()
      elif llm_name.strip() == "gemma_3_7b":
        dict_llm_code[llm_name.strip()] == llm_code.strip()
      else:
        return {"messages": [{"role": "system", "content": "Error: An error occured while trying to put LLM codes in a dictionary before evaluating those."}]}

 
    # start evaluating with structured output
    for k, v in dict_llm_code.items():
      """
       make llm call with structured output to tell if code is YES valid or NO invalid.
      """
  
      # we need to fill this template to create the query for this code evaluation task
      query = prompt_creation(<PUT_HERE_PROMPT_TEMPLATE>["human"], user_initial_query=user_initial_query, api_choice=api_choice, apis_links=apis_links, llm=k, code=v)
  
      evaluation = structured_output_for_agent(<STRUCTURED_OUTPUT_CLASS>, query, <PUT_HERE_PROMPT_TEMPLATE>["system"]["template"])


# CONDITIONAL EDGES FUNCTIONS

def rewrite_documentation_or_execute(state: MessagesState):
  # this is the jodge agent message decision
  messages = state["messages"]
  last_message = messages[-1].content
  decision = json.loads(last_message)
  # decision["DECISION"] rewrite/generate ; decision["REASON"] long str ; decision["STAGE"] rewrite/internet
  decision_dict = json.loads(last_message)
  # we check retries are not null
  load_dotenv(dotenv_path=".vars.env", override=True)
  retry_doc_validation = os.getenv("RETRY_DOC_VALIDATION")
  if retry_doc_validation == 0:
    state["messages"].append({"role": "system", "content": "All retries have been consummed, failed to rewrite documentation"})
    return "error_handler"
  
  if "success" in decision:
    return ["llama_3_8b_script_creator", "llama_3_70b_script_creator", "gemma_3_7b_script_creator"] # those nodes will run in parallel
  # this one loops back up to recreate initial documentation by searching online
  elif "disagree" in decision and "internet" in decision["STAGE"]: 
    return "find_documentation_online_agent"
  # this one loops back the document writer
  elif "disagree" in decision and "rewrite" in decision["STAGE"]:
   return "documentation_writer"
  else:
    return "error_handler"



# Initialize states
workflow = StateGraph(MessagesState)

# nodes
workflow.add_node("error_handler", error_handler)
workflow.add_node("get_user_input", get_user_input)
workflow.add_node("tool_api_choose_agent", tool_api_choose_agent)
workflow.add_node("tool_agent_decide_which_api_node", tool_agent_decide_which_api_node)
workflow.add_node("find_documentation_online_agent", find_documentation_online_agent) # internet search tool with the right schema sent to the ToolNode executor
workflow.add_node("tool_search_node", tool_search_node)
workflow.add_node("documentation_writer", documentation_writer) # will get the internet search result about documentation in how to perform api call and write documentation
workflow,add_node("documentation_steps_evaluator_and_doc_judge", documentation_steps_evaluator_and_doc_judge)
workflow.add_node("llama_3_8b_script_creator", llama_3_8b_script_creator)
workflow.add_node("llama_3_70b_script_creator", llama_3_70b_script_creator)
workflow.add_node("gemma_3_7b_script_creator", gemma_3_7b_script_creator)
workflow.add_node("inter_graph_node", inter_graph_node)


# edges
workflow.set_entry_point("get_user_input")
"""
see if we add conditional adge from "get_user_input" node
"""
# tool edges
workflow.add_edge("tool_api_choose_agent", "tool_agent_decide_which_api_node")
# answer user
workflow.add_edge("error_handler", "answer_user")
workflow.add_edge("tool_agent_decide_which_api_node", "find_documentation_online_agent")
workflow.add_edge("find_documentation_online_agent", "tool_search_node")
workflow.add_edge("tool_search_node", "documentation_writer")
workflow.add_edge("documentation_writer", "documentation_steps_evaluator_and_doc_judge")
workflow.add_conditional_edge(
  "documentation_steps_evaluator_and_doc_judge",
  # conditional that will go to code execution, or loop back to find_documentation_online_agent, or loop back to  document_writer
  # we need to ask llm to read each docs or output [-1][-2][-3] and tell for eahc `OK` or `NOT OK` so that we can create conditions in function of that and go back to try again.
  # need to set a retry_max as well with env vars
  rewrite_documentation_or_execute,
)
workflow.add_edge("llama_3_8b_script_creator", "code_evaluator_and_final_script_writer")
workflow.add_edge("llama_3_70b_script_creator", "code_evaluator_and_final_script_writer")
workflow.add_edge("gemma_3_7b_script_creator", "code_evaluator_and_final_script_writer")
workflow.add_condition_edge(
  "code_evaluator_and_final_script_writer",
  """
   Fonction to be made
  """
  rewrite_or_execute
)
'''
# dont forget `retry_doc_excecution = os.getenv("RETRY_CODE_EXECUTION=")` after that
'''


# end
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
  graph_image = user_query_processing_stage.get_graph(xray=True).draw_mermaid_png()
  with open("primary_subgraph.png", "wb") as f:
    f.write(graph_image)
  
  if os.getenv("PARQUET_FILE_PATH"):
    if ".parquet" in os.getenv("PARQUET_FILE_PATH"):
      # tell to start `embeddings` graph
      return "embeddings"
    return "error"
  elif "true" in os.getenv("PRIMARY_GRAPH_NO_DOCUMENTS_NOR_URL"):
    return "done"
  else:
    return "error"

'''
if __name__ == "__main__":

  # to test it standalone
  primary_graph()
'''


