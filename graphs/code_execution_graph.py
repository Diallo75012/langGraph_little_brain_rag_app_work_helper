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
  # class for user input analysis
  AnalyseUserInput,
  # function to analyze user input
  structured_output_for_get_user_input,
  # class for document writer
  DocumentationWriter,
  # function for document writer structured ouput
  structured_output_for_documentation_writer,
  # class for script creation
  ScriptCreation,
  # function for script creation
  structured_output_for_script_creator,
  # class for document quality judgement
  CodeDocumentationEvaluation,
  # function for documentation evaluation
  structured_output_for_documentation_steps_evaluator_and_doc_judge,
  # class for script quality evaluation
  CodeScriptEvaluation,
  # function for script evaluation
  structured_output_for_code_evaluator_and_final_script_writer,
  # code comparator class
  CodeComparison,
  # function to compare and choose one script only
  structured_output_for_agent_code_comparator_choice,
  # class for requirements.txt creation
  CodeRequirements,
  # function to create the requriements.txt
  structured_output_for_create_requirements_for_code,
  # class for code analysis report
  CodeErrorAnalysis,
  # function for code execution analysis
  structured_output_for_error_analysis_node
)
# prompts
from prompts.prompts import (
  get_user_input_prompt,
  tool_api_choose_agent_prompt,
  find_documentation_online_agent_prompt,
  documentation_writer_prompt,
  rewrite_or_create_api_code_script_prompt,
  script_creator_prompt,
  code_evaluator_and_final_script_writer_prompt,
  choose_code_to_execute_node_if_many_prompt,
  create_requirements_for_code_prompt
)
from typing import Dict, List, Any, Optional, Union
# Prompts LAngchain and Custom
from langchain_core.prompts import PromptTemplate
# Node Functions from App.py 
"""
Maybe will have to move those functions OR to a file having all node functions OR to the corresponding graph directly
"""
# utils
from app_utils import prompt_creation
# Docker execution code
from docker_agent.execution_of_agent_in_docker_script import run_script_in_docker
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
  # we return last 3 messages
  last_three_messages_json = json.dumps([
    {
      "Graph Code Execution Done": "Find here the last 3 messages of code execution graph",
      "message -3": messages[-3].content,
      "message -2": messages[-2].content,
      "message -1": messages[-1].content,
    }
  ])
  
  return {"messages": [{"role": "system", "content": last_three_messages_json}]}

# api selection
def tool_api_choose_agent(state: MessagesState):
    messages = state['messages']
    last_message = messages[-1].content
    print("message state -1: ", messages[-1].content, "\nmessages state -2: ", messages[-2].content)
    # print("messages from call_model func: ", messages)
    user_initial_query = os.getenv("USER_INITIAL_QUERY")

    query = prompt_creation(tool_api_choose_agent_prompt["human"], user_initial_query=user_initial_query)
    response = llm_api_call_tool_choice.invoke(json.dumps(query))

    return {"messages": [response]}

# NODE FUNCTIONS
def get_user_input(state: MessagesState, analyse_query_class = AnalyseUserInput):
  messages = state['messages']
  last_message = messages[-1].content
  set_key(".vars.env", "USER_INITIAL_QUERY", last_message)
  load_dotenv(dotenv_path=".vars.env", override=True)

  print(f"ENV VAR SAVED QUERY: {os.getenv('USER_INITIAL_QUERY')}")
  
  # use structured output to decide if we need to generate documentation and code
  query = prompt_creation(get_user_input_prompt["human"], user_initial_query=last_message)
  try:
    query_analysis = structured_output_for_get_user_input(analyse_query_class, query, get_user_input_prompt["system"]["template"]) 
    '''
    { 
      "code": response.code,
      "onlydoc": response.onlydoc,
      "nothing": response.nothing,
      "pdforurl": response.pdforurl
    }
    '''
    # conditional edge at this node will handle the flow to return back to some nodes or to start the other graphs directly 
    for elem, status in query_analysis.items():
      # this is the route to go to another graph with only a question or with a question and a url/pdf coming with it
      if elem == "nothing" and status.lower() == "yes":
        for k, v in query_analysis.items():
          # route that will forward the key 'pdforurl' so that it will be sent to the right graph 'primary graph'
          if k == "pdforurl" and v.lower() == "yes":  
            return {"messages": [{"role": "ai", "content": str(k)}]}
        # route that will forward 'nothing' meaning that there is only a question in the query and we will send to the 'report' graph
        return {"messages": [{"role": "ai", "content": str(elem)}]}
      # otherwise check if we need doc or code to be generated
      elif (elem == "onlydoc" and status.lower() == "yes") or (elem == "code" and status.lower() == "yes"):
        # route that will forward or 'onlydoc' or 'code' in order to start this graph (we could split it and improve code to have onlydoc generation then stop or only code generation and execution then stop but we are going to keep the mvp simple yet it is advanced at this stage already)
        return {"messages": [{"role": "ai", "content": str(elem)}]}
  except Exception as e:
    return {"messages": [{"role": "ai", "content": f"An error occured while trying to analyze user input content: {e}"}]}

        
       


def error_handler(state: MessagesState):
  messages = state['messages']
  last_three_messages_json = json.dumps([
    {
      "Error": "Find here the last 3 messages of code execution graph",
      "message -3": messages[-3].content,
      "message -2": messages[-2].content,
      "message -1": messages[-1].content,
    }
  ])
  
  with open("./logs/conditional_edge_logs.log", "a", encoding="utf-8") as conditional:
      conditional.write(f"\n\nerror handler called: {last_three_messages_json}\n\n")

  return {"messages": [{"role": "ai", "content": f"An error occured, error message: {last_three_messages_json}"}]}

def find_documentation_online_agent(state: MessagesState, APIs = apis):
    messages = state['messages']
    
    ## VARS
    # should be the tool that have been picked by the agent to satisfy user request
    last_message = messages[-1].content
    # create environment variable with the API that have been chosen for next nodes to have access to it if needed
    set_key(".vars.env", "API_CHOICE", last_message)
    load_dotenv(dotenv_path=".vars.env", override=True)
    print("message state -1: ", messages[-1].content)
    user_initial_query = os.getenv("USER_INITIAL_QUERY")
    
    # create prompt
    query = prompt_creation(find_documentation_online_agent_prompt["human"], last_message=last_message, APIs=APIs, user_initial_query=user_initial_query)
    response = llm_with_internet_search_tool.invoke(json.dumps(query))
    return {"messages": [response]}

# documentation writer with structured output help
def documentation_writer(state: MessagesState, apis = apis, doc_writer_class = DocumentationWriter):
  messages = state['messages']
  
  #### VARS ####
  # this message should be the internet search result telling how to create the code to make api call to the chosen api
  last_message = messages[-1].content
  # save last_message which is the internet search result so that we can access it from other nodes in case of a loop back in a future node so not coming back here
  set_key(".vars.env", "DOCUMENTATION_FOUND_ONLINE", last_message)
  load_dotenv(dotenv_path=".vars.env", override=True)
  # online doc found
  documentation_found_online = os.getenv("DOCUMENTATION_FOUND_ONLINE")
  # agent choosen api to satisfy user api call code generation
  api_choice = os.getenv("API_CHOICE")
  # user initial query
  user_initial_query = os.getenv("USER_INITIAL_QUERY")

  # creation of the query by injecting variables in  
  query = prompt_creation(documentation_writer_prompt["human"], user_initial_query=user_initial_query, apis_links=apis, api_choice=api_choice, documentation_found_online=last_message)
  # get the answer as we want it using structured output and injecting the 'human' prompt in the system one
  try:
    written_documentation = structured_output_for_documentation_writer(doc_writer_class, query, documentation_writer_prompt["system"]["template"])
    print("Writen_documentation: ", type(written_documentation), written_documentation)
    
    # parse content, here doc is str and written_documentation a dict
    doc = written_documentation["documentation"]
    print("DOC: ", type(doc), doc)
  
    # write the documentation to a file
    with open(os.getenv("CODE_DOCUMENTATION_FILE_PATH"), "w", encoding="utf-8") as code_doc:
      code_doc.write("""'''This file contains the documentation in how to write a mardown Python script to make API to satisfy user request.Make sure that the script is having the right imports, indentations, uses Python standard libraries and do not have any error before starting writing it.'''\n\n""")
      code_doc.write(doc)
    
    # go the judge and documentation evaluator agent
    return {"messages": [{"role": "ai", "content": doc}]}
  except Exception as e:
    print("DOCUMENTATION WRITER ERROR TRIGGERED")
    return {"messages": [{"role": "ai", "content": f"error while trying to write documentation from internet search information about {api_choice} api: {e}"}]}

#### Parallel execution to get code generated by different elements
def llama_3_8b_script_creator(state: MessagesState, apis=apis, script_creation_class = ScriptCreation, llm = groq_llm_llama3_8b):
  #messages = state['messages']
  #last_message = messages[-1].content
  #response = groq_llm_llama3_8b.invoke(messages)
  #messages = [system_message, human_message]
  # human_message = HumanMessage(content=script_creator_prompt["human"]["template"])
  
  # online doc found
  documentation_found_online = json.loads(os.getenv("DOCUMENTATION_FOUND_ONLINE"))["messages"][0]
  #print("DOCUMENTATION sent to create code: ", documentation_found_online)
  # agent choosen api to satisfy user api call code generation
  api_choice = os.getenv("API_CHOICE")
  # user initial query
  user_initial_query = os.getenv("USER_INITIAL_QUERY")
  # example json for structured output
  example_json = json.dumps({'script': '<put the Python script here>',})

  # creation of the query by injecting variables in  
  query = prompt_creation(documentation_writer_prompt["human"], user_initial_query=user_initial_query, apis_links=apis, api_choice=api_choice, documentation_found_online=documentation_found_online)
  # get the answer as we want it using structured output and injecting the 'human' prompt in the system one
  try:
    script_created = structured_output_for_script_creator(script_creation_class, query, example_json, script_creator_prompt["system"]["template"], llm)
    final_script = script_created["script"]
    return {"messages": [{"role": "ai", "content": json.dumps({"llama_3_8b": final_script})}]}
  except Exception as e:
    return {"messages": [{"role": "ai", "content": json.dumps({"error":e})}]}

def llama_3_70b_script_creator(state: MessagesState, apis=apis, script_creation_class = ScriptCreation, llm = groq_llm_llama3_70b):
  # online doc found
  documentation_found_online = json.loads(os.getenv("DOCUMENTATION_FOUND_ONLINE"))["messages"][0]
  # agent choosen api to satisfy user api call code generation
  api_choice = os.getenv("API_CHOICE")
  # user initial query
  user_initial_query = os.getenv("USER_INITIAL_QUERY")
  # example json for structured output
  example_json = json.dumps({'script': '<put the Python script here>',})

  # creation of the query by injecting variables in  
  query = prompt_creation(documentation_writer_prompt["human"], user_initial_query=user_initial_query, apis_links=apis, api_choice=api_choice, documentation_found_online=documentation_found_online)
  # get the answer as we want it using structured output and injecting the 'human' prompt in the system one
  try:
    script_created = structured_output_for_script_creator(script_creation_class, query, example_json, script_creator_prompt["system"]["template"], llm)
    final_script = script_created["script"]
    return {"messages": [{"role": "ai", "content": json.dumps({"llama_3_70b": final_script})}]}
  except Exception as e:
    return {"messages": [{"role": "ai", "content": json.dumps({"error":e})}]}

def gemma_3_7b_script_creator(state: MessagesState, apis=apis, script_creation_class = ScriptCreation, llm = groq_llm_gemma_7b):
  # online doc found
  documentation_found_online = json.loads(os.getenv("DOCUMENTATION_FOUND_ONLINE"))["messages"][0]
  # agent choosen api to satisfy user api call code generation
  api_choice = os.getenv("API_CHOICE")
  # user initial query
  user_initial_query = os.getenv("USER_INITIAL_QUERY")
  # example json for structured output
  example_json = json.dumps({'script': '<put the Python script here>',})

  # creation of the query by injecting variables in  
  query = prompt_creation(documentation_writer_prompt["human"], user_initial_query=user_initial_query, apis_links=apis, api_choice=api_choice, documentation_found_online=documentation_found_online)
  # get the answer as we want it using structured output and injecting the 'human' prompt in the system one
  try:
    script_created = structured_output_for_script_creator(script_creation_class, query, example_json, script_creator_prompt["system"]["template"], llm)
    final_script = script_created["script"]
    return {"messages": [{"role": "ai", "content": json.dumps({"gemma_3_7b": final_script})}]}
  except Exception as e:
    return {"messages": [{"role": "ai", "content": json.dumps({"error":e})}]}

# judging documentation created agent
def documentation_steps_evaluator_and_doc_judge(state: MessagesState, apis = apis, code_doc_eval_class = CodeDocumentationEvaluation):
  """
    Will judge the documentation and return 'rewrite' to rewrite the documentation or 'generate' to start generating the script
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
    decision = structured_output_for_documentation_steps_evaluator_and_doc_judge(code_doc_eval_class, query, rewrite_or_create_api_code_script_prompt["system"]["template"])
    print("\n\n\nDECISION: ", decision, type(decision))
    """
      # Object returned by the structured output llm call
      { 
      "decision": response.decision,
      "reason": response.reason,
      "stage": response.stage,
      }
    """
    if decision["decision"] == "rewrite":
      return {"messages": [{"role": "ai", "content": json.dumps({"disagree":decision})}]} # use spliting technique to get what is wanted 
    elif decision["decision"] == "generate":
      return {"messages": [{"role": "ai", "content": json.dumps({"success":"generate"})}]}
    else:
      return {"messages": [{"role": "ai", "content": json.dumps({"error":decision})}]}
  except Exception as e:
    return {"messages": [{"role": "ai", "content": json.dumps({"error":e})}]} 

def code_evaluator_and_final_script_writer(state: MessagesState, apis = apis, evaluator_class = CodeScriptEvaluation):
    """
    This node receives code from different LLMs, evaluates if it's valid by calling another LLM, 
    and returns 'YES' or 'NO' for validation using structured outputs.

    Args:
    state: MessagesState object containing the messages and codes generated by different LLMs.

    Returns:
    A dictionary containing the message response indicating whether the code is valid ('YES') or not ('NO').
    """

    # vars
    # the final state messages returned by the node
    llm_code_valid_responses_list_dict = []
    llm_code_invalid_responses_list_dict = []
    # user inital query
    user_initial_query = os.getenv("USER_INITIAL_QUERY")
    # api chosen to satisfy query
    api_choice = os.getenv("API_CHOICE")
    # links for api calls for each apis existing in our choices of apis
    apis_links = apis
    # example json for structured output
    example_json = json.dumps({'validity': 'YES/NO', 'reason': 'Your explanation for the evaluation.'})
 
    # Extract the last few messages, assuming they are code outputs from different LLMs
    messages = state['messages']
    
    # dictionary to have code linked to each llm for easier parsing and allocation and looping through it and formatting answers
    dict_llm_codes = {} # name_llm: code_of_that_llm
    exceptions = []
    # Assume the last three messages are the code outputs
    codes = [ 
      json.loads(messages[-1].content),
      json.loads(messages[-2].content),
      json.loads(messages[-3].content)
    ]
    print("CODES: ", codes)
    # count how many return an exception
    count = 0
    for elem in codes:
      for llm, code in elem.items():
        if llm == "error":
          # append exception message to render it if they all fail
          exceptions.append({llm: code})
          count+=1
          if count == 3:
            print("EXCEPTIONS: ", exceptions)
            # notify next to render error
            return {"messages": [{"role": "ai", "content": json.dumps({"error": "An error occured while generating codes, all failed with exception:{exceptions}"})}]}
        if llm == "llama_3_8b":
          print("LLM NAME: ", llm)
          # we don't want code that LLM have splitted each lines with ```python ``` markdown code, we need all in one block otherwise it is not valid_code
          dict_llm_codes[llm] = code
        elif llm == "llama_3_70b":
          print("LLM NAME: ", llm)
          dict_llm_codes[llm] = code
        elif llm == "gemma_3_7b":
          print("LLM NAME: ", llm)
          dict_llm_codes[llm] = code
        else:
          return {"messages": [{"role": "system", "content": json.dumps({"error": "An error occured while trying to put LLM codes in a dictionary before evaluating those."})}]}
    
    print("codes: ", codes)
 
    # start evaluating with structured output
    for k, v in dict_llm_codes.items():
      """
       make llm call with structured output to tell if code is YES valid or NO invalid.
      """
  
      # we need to fill this template to create the query for this code evaluation task
      query = prompt_creation(code_evaluator_and_final_script_writer_prompt["human"], user_initial_query=user_initial_query, api_choice=api_choice, apis_links=apis_links, code=v)
      try:
        evaluation = structured_output_for_code_evaluator_and_final_script_writer(evaluator_class, query, example_json, code_evaluator_and_final_script_writer_prompt["system"]["template"])
        print("CODE EVALUATION: ", evaluation)
        """
        { 
          "validity": response.validity,
          "reason": response.reason,
        }
        """
        if "yes" in evaluation["validity"].lower():
          print(f"Code evaluated as YES: {k}")
          llm_code_valid_responses_list_dict.append({k: v})
        elif "no" in evaluation["validity"].lower():
          print(f"Code evaluated as NO: {k}")
          llm_code_invalid_responses_list_dict.append({k: v, "reason": evaluation["reason"]})
      except Exception as e:
        return {"messages": [{"role": "ai", "content": json.dumps({"error": "An error occured while trying to evaluate if LLM codes are valid:{e}"})}]}
    
    return {"messages": [{"role": "ai", "content": json.dumps({"valid": llm_code_valid_responses_list_dict, "invalid": llm_code_invalid_responses_list_dict})}]}

# will create requirments.txt
def choose_code_to_execute_node_if_many(state: MessagesState, comparator_choice_class = CodeComparison):
  # get all the valided codes
  messages = state["messages"]
  valid_code: List[Dict] = json.loads(messages[-1].content) 
  
  user_initial_query = os.getenv("USER_INITIAL_QUERY")
  # llm names to choose from
  name_choices = []
  
  ## ONE SCRIPT ONLY FORWARD TO CREATE REQUIREMENTS IF NEEDED
  # this mean that if there is only one code snippet int he List[Dict] of valid codes this mean that we have just one script we don't need to compare code
  if len(valid_code) < 2:
    print("Len of valid_code samller than 2, so no need to compare code, passing to next node to create requirements.txt")
    # we just pass to next node the serialized List[Dict[llm_name, script]] 
    return {"messages": [{"role": "ai", "content": json.dumps({"one_script_only": messages[-1].content})}]}

  ## MORE THAN ONE SCRIPT CHOOSE ONE ONLY FROM THE GROUP OF VALID SCRIPTS
  # will regroup all the scripts to be compared
  print("VALID CODES IN CHOICE IF MANY: ", valid_codes)
  FORMATTED_CODES: str = ""
  for elem in valid_code:
    for k, v in elem.items():
      name_choices.append(k)
      FORMATTED_CODES += f"'''Script to compare'''\n# LLM Name: {k}\n# Code of LLM name {k}:\n{v}\n\n\n"

  # use structured output to decide on the name of the file that will be chosen for code execution. returns Dict[name, reason]
  name_choices_json = json.dumps(name_choices)
  query = prompt_creation(choose_code_to_execute_node_if_many_prompt["human"], user_initial_query=user_initial_query, code=FORMATTED_CODES)
  try:
    choice = structured_output_for_agent_code_comparator_choice(comparator_choice_class, query, name_choices_json, choose_code_to_execute_node_if_many_prompt["system"]["template"])
    llm_name = choice["llm_name"]
    reason = chocie["reason"]
    print("LLM SCRIPT CHOICE AND REASON: ", llm_name, reason)
    return {"messages": [{"role": "ai", "content": json.dumps({"success": {"llm_name": llm_name, "reason": reason}})}]}
  except Exception as e:
    return {"messages": [{"role": "ai", "content": json.dumps({"error": "An error occured while trying to compare LLM codes aand choose one:{e}"})}]}

# will create requirments.txt
def create_requirements_for_code(state: MessagesState, requirements_creation_class = CodeRequirements):
  # get the valided code
  messages = state["messages"]
  llm_name = json.loads(messages[-1])["llm_name"]

  # use llm_name to get the corresponding script
  scripts_folder = os.getenv("AGENTS_SCRIPTS_FOLDER")
  scripts_list_in_folder = os.listdir(scripts_folder)
  for script in scripts_list_in_folder:
    if script.startswith(f"agent_code_execute_in_docker_{llm_name}"):
      # we get the script code that will be injected in the human prompt
      script_to_create_requirement_for = script

  # use structured output to get the requirements.txt
  query = prompt_creation(create_requirements_for_code_prompt["human"], user_initial_query=user_initial_query, code=script_to_create_requirement_for)
  try:
    requirements_response = structured_output_for_create_requirements_for_code(requirements_creation_class, query, create_requirements_for_code_prompt["system"]["template"])
    
    # responses parsing
    requirements_txt_content = requirements_response["requirements"]
    requirements_needed_or_not = requirements_response["needed"]
    print("REQUIREMENTS AND IF NEEDED OR NOT: ", requirements_txt_content, requirements_needed_or_not)
    
    # check if llm believed that code needs a requirements.txt file or not
    if "yes" in requirements_needed_or_not.lower().strip():
      # create the requirements.txt file
      sandbox_requirement_file = f"{script_folder}/{llm_name}_requirements.txt"
      with open(sandbox_requirement_file, "w", encoding="utf-8") as req:
         req.write(f"# This requirements for the code generated by {llm_name}\n")
         req.write(requirements_txt_content.strip())
    # return with name of the requirement file in the states
    return {"messages": [{"role": "ai", "content": json.dumps({"success": {"requirements": sandbox_requirement_file, "llm_name": llm_name}})}]}
  except Exception as e:
    return {"messages": [{"role": "ai", "content": json.dumps({"error": "An error occured while trying to check if code needs requirements.txt file created or not:{e}"})}]}

# will execute code in docker sandbox
def code_execution_node(state: MessagesState):
  stdout_list_dict = []
  stderr_list_dict = []
  exception_error_log_list_dict = []
  
  messages = state["messages"]
  last_message = json.loads(messages[-1].content)
  
  # get requirements file name and llm name
  requirements_file_name = last_messages["requirements"]
  llm_name = last_messages["llm_name"]
  
  # track reties
  retry_code_execution = int(os.getenv("RETRY_CODE_EXECUTION"))
  print("RETRY CODE EXECUTION: ", retry_code_execution)
  if retry_doc_execution == 0:
    state["messages"].append({"role": "system", "content": "All retries have been consummed, failed to execute code or stopped at code execution"})
    return "error_handler"

  if retry_code_execution > 0:
    # returns a tupel Tuple[stdout, stderr] , scripts present at: ./docker_agent/agents_scripts/{script_path}
    agents_scripts_dir = os.getenv("AGENTS_SCRIPTS_FOLDER")
    for agent_script_file_name in os.lisdir(agents_scripts_dir):
      if agent_script_file_name.startwith(f"agent_code_execute_in_docker_{llm_name}"):
        print("Script file name: ", agent_script_file_name)
        try:
          # returns a Tuple[stdout, stderr]
          stdout, stderr = run_script_in_docker(f"{llm_name}_dockerfile", f"{agents_scripts_dir}/{agent_script_file_name}", f"{agents_scripts_dir}/{requirements_file_name}")
          if stdout:
            stdout_list.append({
              "script_file_path": f"{agents_scripts_dir}/{agent_script_file_name}",
              "llm_script_origin": llm_name,
              "output": f"success:{stdout}",  
            })
          if stderr:
            stderr_list.append({
              "script_file_path": f"{agents_scripts_dir}/{agent_script_file_name}",
              "llm_script_origin": llm_name,
              "output": f"error:{stderr}",  
            })
        except Exception as e:
          # here make sure that the app doesn't stop but just manage the error to be stored and fowarded
          error_log_dict.append({
            "script_file_path": f"{agents_scripts_dir}/{agent_script_file_name}",
            "llm_script_origin": llm_name,
            "output": f"exception:An error occured while trying to execute code in docker sandbox:{e}",
          })

  # decrease the retry value
  retry_code_execution -= 1
  set_key(".vars.env", "RETRY_CODE_EXECUTION", str(retry_code_execution))
  load_dotenv(dotenv_path=".vars.env", override=True)

  # send the results to next node
  return {"messages": [{"role": "ai", "content": json.dumps({"stdout": stdout_list_dict, "stderr": stderr_list_dict, "exception": exception_error_log_list_dict})}]}   

# code has successfully being executed
def successful_code_execution_management(state: MessagesState):
  messages = state["messages"]
  stdout = json.loads(messages[-1].content)["stdout"]
  # create a report from here so save this successful graph code execution to an env var for business logic side to get it and forward to report_creation node
  set_key(".vars.env", "CODE_EXECUTION_GRAPH_RESULT", json.dumps({"success": stdout}))
  load_dotenv(dotenv_path=".vars.env", override=True)
  return {"messages": [{"role": "system", "content": json.dumps({"success_code_execution": stdout})}]}

# will analyse error returned by code execution
def error_analysis_node(state: MessagesState, code_error_analysis_class = CodeErrorAnalysis):
  """
    This node function will use structured output in order to identify errors in script or requirements.txt and go to execution node to try again with those new updated files
    Files names will be preserved so that execution node can easily use same pattern to re-execute code.
  """
  messages = state["messages"]
  stderr = json.loads(messages[-1].content)["stderr"]
  # get the different values
  error_message = stderr["output"]
  code = stderr["script_file_path"]
  llm_name = stderr["llm_script_origin"]
  for script in os.listdir(os.getenv("AGENTS_SCRIPT_FOLDER")):
    if script.endswith(".txt") and script.startswith("llm_name"):
      requirements_file = script
   
  # ask llm to check the error compared to the user initial query, the api chosen, the code and the requirements.txt and use structured output to provide new code, new requirements.txt and return to execute code with those new file, therefore need to create an explicative name for those new files and save those names to state to keep track
  # see if we add the dockerfile of not dockerfile=f"{llm_name}_dockerfile"
  query = prompt_creation(error_analysis_node_prompt["human"], error=error_message, code=code, requirements=requirements_file)
  '''
   don't forget that you also need to provide the dockerfile and use the same file names in formatting when rewriting those files so that the execution code node will just run the same file names but which have been overwritten
  '''
  try:
    code_analysis_response = structured_output_for_error_analysis_node(code_error_analysis_class, query, error_analysis_node_prompt["system"]["template"])

    # responses parsing
    code_requirements = code_analysis_response["requirements"]
    code_script = code_analysis_response["script"]
    requirements_needed = code_analysis_response["needed"]
    print("REQUIREMENTS AND IF NEEDED OR NOT: ", requirements_needed)
    
    # check if llm believed that code needs a requirements.txt file or not
    if "yes" in requirements_needed.lower().strip():
      # create the requirements.txt file
      sandbox_requirement_file = f"{script_folder}/{llm_name}.txt"
      with open(sandbox_requirement_file, "w", encoding="utf-8") as req:
         req.write(f"# Initial requirements code generated by {llm_name} initially but got execution error: {stderr}. Therefore, created this new one through error analysis.\n")
         req.write(requirements_txt_content.strip())
    elif code_script:
      # update script file with new code using same file
      agent_script = f"./docker_agent/agents_scripts/agent_code_execute_in_docker_{llm_name}.py"
      with open(agent_script, "w", encoding="utf-8") as agts:
         agts.write("#!/usr/bin/env python3")
         agts.write(f"'''This requirements for the code generated by {llm_name} after execution error: {stderr}\n")
         agts.write(agent_script)
    
    # return a message to notify conditional edge to go to code re-execution
    return {"messages": [{"role": "ai", "content": "re_execute_code"}]}
  except Exception as e:
    # return a message with keyword 'error' in message to go error_handler node
    return {"messages": [{"role": "system", "content": f"An error occure while trying to manage the code execution error to identify failure and correct it: {e}"}]}

 

######   CONDITIONAL EDGES FUNCTIONS
# check if code creation is needed or not to fulfil user query
def code_or_doc_needed_or_not(state: MessagesState):
  messages = state["messages"]
  # last_message will be either of those keys "code", "onlydoc", "nothing", "pdforurl"
  last_message = messages[-1].content

  # we could separate this but will just start at the beginning of the graph here for this mvp
  # but we could just do the full graph or just generate only documentation and exit
  if "code" in last_message or "onlydoc" in last_message:
    # set the envrionement variable to notified the 'Business Logic' side of the app that only code generation and documentation was needed and stop the app
    set_key(".vars.env", "DOCUMENTATION_AND_CODE_GENERATION_ONLY", "true")
    load_dotenv(dotenv_path=".vars.env", override=True)
    return "tool_api_choose_agent"
  elif "pdforurl" in last_message:
    return ""
  elif "nothing" in last_message:
    # notify the end of this graph that we need a report to be created so this will start next graph in the 'Business Logic' side of the app
    set_key(".vars.env", "REPORT_NEEDED", "true")
    load_dotenv(dotenv_path=".vars.env", override=True)
    return "inter_graph_node" # go the end of the graph and call the report graph 
  else:
    # if there is a problem we will return the error which will just render the last message or  the last 3 messages
    return "error_handler"

def evaluate_doc_or_error(state: MessagesState):
  messages = state["messages"]
  last_message = messages[-1].content
  
  if "error" in last_message:
    return "error_handler"
  else:
    return "documentation_steps_evaluator_and_doc_judge"
    

# after doc evaluation and judge
def rewrite_documentation_or_execute(state: MessagesState):
  # this is the jodge agent message decision
  messages = state["messages"]
  last_message = messages[-1].content
  # decision["decision"] rewrite/generate ; decision["reason"] long str ; decision["stage"] rewrite/internet
  decision = json.loads(last_message)
  print("REWRITE DOCUMENTATION OR EXECUTE DECISION: ", decision, type(decision))
  
  # we check retries are not null
  load_dotenv(dotenv_path=".vars.env", override=True)
  retry_doc_validation = int(os.getenv("RETRY_DOC_VALIDATION"))
  print("RETRY DOC VALIDATION: ", retry_doc_validation)
  if retry_doc_validation == 0:
    state["messages"].append({"role": "system", "content": "All retries have been consummed, failed to rewrite documentation"})
    return "error_handler"
  
  # decrease the retry value
  retry_doc_validation -= 1
  set_key(".vars.env", "RETRY_DOC_VALIDATION", str(retry_doc_validation))
  load_dotenv(dotenv_path=".vars.env", override=True)

  for k,v in decision.items():
    if k == "success":
      print("k: ", k)
      return ["llama_3_8b_script_creator", "llama_3_70b_script_creator", "gemma_3_7b_script_creator"] # those nodes will run in parallel
    # this one loops back up to recreate initial documentation by searching online
    elif k == "disagree":
      print("k: ", k)
      if v["stage"] == "internet":
        return "find_documentation_online_agent"
      # this one loops back the document writer
      elif v["stage"] == "rewrite":
        return "documentation_writer"
    elif k == "error":
      return "error_handler"

  # default return if nothing match or other issue
  return "error_handler"


# after code evaluator
def rewrite_or_create_requirements_decision(state: MessagesState):
  messages = state["messages"]
  # we receive a LIST[DICT[LLM,CODE]]
  evaluations =  json.loads(messages[-1].content)
  print("EVALUATIONS in conditional edge: ", evaluations, type(evaluations))

  load_dotenv(dotenv_path=".vars.env", override=True)
  retry_code_validation = int(os.getenv("RETRY_CODE_VALIDATION"))
  print("RETRY CODE VALIDATION: ", retry_code_validation)
  # forward try/except error to node 'error_handler'
  if "error" in evaluations or retry_code_validation == 0:
    if retry_code_validation == 0:
      state["messages"].append({"role": "system", "content": "All retries have been consummed, failed to validate documentation"})
    if "error" in evaluations:
      state["messages"].append({"role": "system", "content": f"error present in doc evaluation: {evaluations}"})
    return "error_handler"
  
  valid_code = evaluations["valid"]
  invalid_code = evaluations["invalid"]
  
  # if no valid code we will check retries and go back to the coding nodes
  if not valid_code and retry_code_validation > 0:
    
    # decrease the retry value
    retry_code_validation -= 1
    set_key(".vars.env", "RETRY_CODE_VALIDATION", str(retry_code_validation))
    load_dotenv(dotenv_path=".vars.env", override=True)
    
    return ["llama_3_8b_script_creator", "llama_3_70b_script_creator", "gemma_3_7b_script_creator"]
   
  if valid_code and retry_code_validation > 0:

    # decrease the retry value
    retry_code_validation -= 1
    set_key(".vars.env", "RETRY_CODE_VALIDATION", str(retry_code_validation))
    load_dotenv(dotenv_path=".vars.env", override=True)

    # we loop over the valid code and save those to files that will be executed in next graphm we might need to put those in a folder nested inside the docker stuff
    for elem in valid_code:
      for llm, script in elem.items():
        with open(f"./docker_agent/agents_scripts/agent_code_execute_in_docker_{llm}.py", "w", encoding="utf-8") as llm_script:
          llm_script.write("#!/usr/bin/env python3")
          llm_script.write(f"# code from: {llm} LLM Agent\n# User initial request: {os.getenv('USER_INITIAL_REQUEST')}\n'''This code have been generated by an LLM Agent'''\n\n")
          llm_script.write(script)
     
    print("All Python script files are ready to be executed!")
     
    # append the valid code to state messages LIST[DICT]
    state["messages"].append({"role": "system", "content": json.dumps(valid_code)})
    
    # go to "choose_code_to_execute_node_if_many"
    # return "create_requirements_for_code"
    return "choose_code_to_execute_node_if_many"


# after code execution
def success_execution_or_retry_or_return_error(state: MessagesState):
  messages = state["messages"]
  # unwrap the previous output results of code execution
  # LIST[DICT] : {"stdout": stdout_list_dict, "stderr": stderr_list_dict, "exception": exception_error_log_list_dict}
  code_execution_result = json.load(messages[-1].content)

  # get the results list dicts having keys: script_file_path, llm_script_orign and output, check output if it has success, error or exception in the message
  # if we have at least one good code execution we want to analyze it with next node and know which code it was, from which llm
  if code_execution_result["stdout"]:
    # we have some code that has been executed properly and now want to send to next node that will sort it out in order to decide which code will be rendered to user
    return "successful_code_execution_management"
  
  # if no code executed well we need to analyze the error or send those errors to a certain node for retry
  else:
    # first we handle the stderr we go to a certain node
    if code_execution_result["stderr"]:
      # otherwise we check if there is an exception and will return the result error_handler node that will handle the message
      state["messages"].append({"role": "system", "content": json.dumps(code_execution_result["stderr"])})
      return "error_analysis_node" # intermediary node before retry execution node to be created and will check the the log file and the stderr thouroughtly to decide if error_handler node, code rewriting again which will go to code execution, or documentation rewriting again which will go through all checks and nodes to execute again 
    else:
      # json.dumps the exception list of dictionaries and send it to error_handler as last message in the states
      state["messages"].append({"role": "system", "content": json.dumps(code_execution_result["exception"])})
      return "error_handler"

# after choice of the code snippet that will be used to create requirement.txt from and execute in docker
def create_requirements_or_error(state: MessagesState):
  messages = state["messages"]
  outcome = json.loads(messages[-1].content)
  
  ## CREATE REQUIREMENTS FOR THE BEST SCRIPT SELECTED
  if "success" in outcome: # Dict[Dict[name, reason]]
    # update state with the name of the llm only that will be passed to next node so it can find that file code and execute it in docker
    llm_name = outcome["success"]["llm_name"]
    print("LLM CHOSEN TO BE EXECUTED IN DOCKER: ", llm_name)
    state["messages"].append({"role": "system", "content": json.dumps({"llm_name": llm_name})})
    # go to code execution node
    return "create_requirements_for_code"
  
  ## CREATE REQUIREMENTS FOR THE UNIQUE SCRIPT VALID
  elif "one_script_only" in outcome: # Dict[name, script]
    # # update state with the name of the llm only that will be passed to next node so it can find that file code and execute it in docker
    llm_name = outcome["one_script_only"]
    state["messages"].append({"role": "system", "content": json.dumps({"llm_name": llm_name})})
    # go to code execution node
    return "create_requirements_for_code"
  
  ## FALLBACK TO ERROR HANDLER FORWARDING ERROR OR JUST BY DEFAULT
  else:
    if "error" in outcome: # str
      # go to error_handler
      return "error_handler"
    return "error_handler"

# execute code or return error conditional edge
def execute_code_or_error(state: MessagesState):
  messages = ["messages"]
  last_message = json.loads(messages[-1].content)
  
  if "success" in last_message:
    return "code_execution_node"
  
  elif "error" in last_message:
    return "error_handler"
  return "error_handler"

# Conditional edge that will just take the returned messages from error_analysis_node to know if we render error or if we re-execute code after having produced new code files.
def re_execute_or_error(state: MessagesState):
  messages = state['messages']
  last_message = messages[-1].content
  if "re_execute_code" in last_message and int(os.getenv("RETRY_CODE_EXECUTION")) > 0:
    return "code_execution_node"
  elif "error" in last_message:
    return "error_handler"



# Initialize states
workflow = StateGraph(MessagesState)

# nodes
workflow.add_node("error_handler", error_handler) # here we need to manage error handler to not just stop the app but loop bas to some nodes with the reasons and check retry
workflow.add_node("get_user_input", get_user_input)
workflow.add_node("tool_api_choose_agent", tool_api_choose_agent)
workflow.add_node("tool_agent_decide_which_api_node", tool_agent_decide_which_api_node)
workflow.add_node("find_documentation_online_agent", find_documentation_online_agent) # internet search tool with the right schema sent to the ToolNode executor
workflow.add_node("tool_search_node", tool_search_node)
workflow.add_node("documentation_writer", documentation_writer) # will get the internet search result about documentation in how to perform api call and write documentation
workflow.add_node("documentation_steps_evaluator_and_doc_judge", documentation_steps_evaluator_and_doc_judge)
workflow.add_node("llama_3_8b_script_creator", llama_3_8b_script_creator)
workflow.add_node("llama_3_70b_script_creator", llama_3_70b_script_creator)
workflow.add_node("gemma_3_7b_script_creator", gemma_3_7b_script_creator)
workflow.add_node("code_evaluator_and_final_script_writer", code_evaluator_and_final_script_writer)
workflow.add_node("choose_code_to_execute_node_if_many", choose_code_to_execute_node_if_many)
workflow.add_node("create_requirements_for_code", create_requirements_for_code)
workflow.add_node("code_execution_node", code_execution_node)
workflow.add_node("successful_code_execution_management", successful_code_execution_management)
workflow.add_node("error_analysis_node", error_analysis_node)
workflow.add_node("inter_graph_node", inter_graph_node)

# edges
workflow.set_entry_point("get_user_input")
workflow.add_conditional_edges(
  "get_user_input",
  code_or_doc_needed_or_not
)
"""
see if we add conditional adge from "get_user_input" node
"""
# tool edges
workflow.add_edge("tool_api_choose_agent", "tool_agent_decide_which_api_node")
# edges
workflow.add_edge("tool_agent_decide_which_api_node", "find_documentation_online_agent")
workflow.add_edge("find_documentation_online_agent", "tool_search_node")
workflow.add_edge("tool_search_node", "documentation_writer")
workflow.add_conditional_edges(
  "documentation_writer",
  evaluate_doc_or_error
)
workflow.add_conditional_edges(
  "documentation_steps_evaluator_and_doc_judge",
  rewrite_documentation_or_execute,
)
workflow.add_edge("llama_3_8b_script_creator", "code_evaluator_and_final_script_writer")
workflow.add_edge("llama_3_70b_script_creator", "code_evaluator_and_final_script_writer")
workflow.add_edge("gemma_3_7b_script_creator", "code_evaluator_and_final_script_writer")
workflow.add_conditional_edges(
  "code_evaluator_and_final_script_writer",
  rewrite_or_create_requirements_decision
)
workflow.add_conditional_edges(
  "choose_code_to_execute_node_if_many",
  create_requirements_or_error
)
workflow.add_conditional_edges(
  "create_requirements_for_code",
  execute_code_or_error
)
workflow.add_conditional_edges(
  "code_execution_node",
  success_execution_or_retry_or_return_error
)
workflow.add_edge("successful_code_execution_management", "inter_graph_node")
# after error analysis we going back to code execution as we are creating new script or requirements file... or both
workflow.add_conditional_edges(
  "error_analysis_node",
  re_execute_or_error
)
# end
workflow.add_edge("error_handler", END)
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
def code_execution_graph(user_query):
  print("Code execution Graph")
  print(f"Query: '{user_query}'")
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
  
  '''
  # subgraph drawing draw_mermaid_png()
  graph_image = user_query_processing_stage.get_graph(xray=True).draw_mermaid_code()
  with open("code_execution_subgraph.png", "wb") as f:
    f.write(graph_image)
  '''
  
  # this is set from the first node and conditional edge of the graph so that the 'Dusiness Logic' side of the app knows how to manage graph flows
  if "true" in os.getenv("REPORT_NEEDED"):
    # tell to start `primary_graph` graph
    return "primary_graph"
  elif "true" in os.getenv("DOCUMENTATION_AND_CODE_GENERATION_ONLY"):
    return "done"
  else:
    return "error"

'''
if __name__ == "__main__":

  #query = "I need to make an API call to get the age of my friend based on her name which is 'Junko'."
  # to test it standalone
  #code_execution_graph(query)
'''


