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
  structured_output_for_agent_doc_evaluator,
  # class for script quality evaluation
  CodeScriptEvaluation,
  # function for script evaluation
  structured_output_for_agent_code_evaluator,
  # code comparator class
  CodeComparison,
  # function to compare and choose one script only
  structured_output_for_agent_code_comparator_choice,
  # class for requirements.txt creation
  CodeRequirements,
  # function to create the requriements.txt
  structured_output_for_create_requirements_for_code
)
from prompts.prompts import (
  structured_outpout_report_prompt,
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
from app_utils import creation_prompt
# prompts
from prompts.prompts import rewrite_or_create_api_code_script_prompt
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
  last_message = messages[-1].content
  return {"messages": [{"role": "system", "content": last_message}]}

# api selection
def tool_api_choose_agent(state: MessagesState):
    messages = state['messages']
    last_message = messages[-1].content
    print("message state -1: ", messages[-1].content, "\nmessages state -2: ", messages[-2].content)
    # print("messages from call_model func: ", messages)
    response = llm_api_call_tool_choice.invoke(last_message)
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
  with open(os.getenv("CODE_DOCUMENTATION_FILE_PATH"), "w", encoding="utf-8") as code_doc:
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

def code_evaluator_and_final_script_writer(state: MessagesState, evaluator_class = CodeScriptEvaluation):
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
   
    # Extract the last few messages, assuming they are code outputs from different LLMs
    messages = state['messages']
    
    # dictionary to have code linked to each llm for easier parsing and allocation and looping through it and formatting answers
    dict_llm_codes = {} # name_llm: code_of_that_llm
    
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
      query = prompt_creation(code_evaluator_and_final_script_writer_prompt["human"], user_initial_query=user_initial_query, api_choice=api_choice, apis_links=apis_links, code=v)
      try:
        evaluation = structured_output_for_agent_code_evaluator(evaluator_class, query, code_evaluator_and_final_script_writer_prompt["system"]["template"])
        print("CODE EVALUATION: ", evaluation)
        """
        { 
          "VALIDITY": response.decision,
          "REASON": response.reason,
        }
        """
        if "yes" in evaluation["VALIDITY"].lower():
          llm_code_valid_responses_list_dict.append({k: v})
        elif "no" in evaluation["VALIDITY"].lower():
          llm_code_invalid_responses_list_dict.append({k: v, "reason": evaluation['REASON']})
      except Exception as e:
        return {"messages": [{"role": "ai", "content": json.dumps({"error": "An error occured while trying to evaluate if LLM codes are valid:{e}"})}]}
    
    return {"messages": [{"role": "ai", "content": json.dumps({"valid": llm_code_valid_responses_list_dict, "invalid": llm_code_invalid_responses_list_dict})}]}

# will create requirments.txt
def choose_code_to_execute_node_if_many(state: MessagesState, comparator_choice_class = CodeComparison):
  # get all the valided codes
  messages = state["messages"]
  valid_code: List[Dict] = json.loads(messages[-1].content) # name_llm: code_of_that_llm
  
  # this mean that if there is only one code snippet int he List[Dict] of valid codes this mean that we have just one script we don't need to compare code
  if len(valid_code) < 2:
    print("Len of valid_code samller than 2, so no need to compare code, passing to next node to create requirements.txt")
    # we just pass to next node the serialized List[Dict[llm_name, script]] 
    return {"messages": [{"role": "ai", "content": json.dumps({"one_script_only": messages[-1].content})}]}
  # will regroup all the scripts to be compared
  FORMATTED_CODES: str = ""
  for k, v in valid_code.items():
    FORMATED_CODES += f"'''Script to compare'''\n# Script name: {k}\n# Code of {k}:\n{v}\n\n\n"

  # use structured output to decide on the name of the file that will be chosen for code execution. returns Dict[name, reason]
  query = prompt_creation(choose_code_to_execute_node_if_many_prompt["human"], user_initial_query=user_initial_query, code=FORMATTED_CODES)
  try:
    choice = structured_output_for_agent_code_comparator_choice(comparator_choice_class, query, choose_code_to_execute_node_if_many_prompt["system"]["template"])
    llm_name = choice["LLM_NAME"]
    reason = chocie["REASON"]
    print("LLM SCRIPT CHOICE AND REASON: ", llm_name, reason )
    return {"messages": [{"role": "ai", "content": json.dumps({"success": {"llm_name": llm_name, "reason": reason}})}]}
  except Exception as e:
    return {"messages": [{"role": "ai", "content": json.dumps({"error": "An error occured while trying to compare LLM codes aand choose one:{e}"})}]}

# will create requirments.txt
def create_requirements_for_code(state: MessagesState, requirements_creation_class = CodeRequirements):
  # get the valided code
  messages = state["messages"]
  llm_name = json.loads(messages)["llm_name"]

  # use llm_name to get the corresponding script
  scripts_folder = os.getenv("AGENTS_SCRIPTS_FOLDER")
  scripts_list_in_folder = os.listdir(scripts_folder)
  for script in scripts_list_in_folder:
    if script.startswith(f"agent_code_execute_in_docker_{llm_name}")
      # we get the script code that will be injected in the human prompt
      script_to_create_requirement_for = script

  # use structured output to get the requirements.txt
  query = prompt_creation(create_requirements_for_code_prompt["human"], user_initial_query=user_initial_query, code=script_to_create_requirement_for)
  try:
    requirements_response = structured_output_for_create_requirements_for_code(requirements_creation_class, query, create_requirements_for_code_prompt["system"]["template"])
    
    # responses parsing
    requirements_txt_content = requirements_response["REQUIREMENTS"]
    requirements_needed_or_not = chocie["NEEDED"]
    print("REQUIREMENTS AND IF NEEDED OR NOT: ", requirements_txt_content, requirements_needed_or_not)
    
    # check if llm believed that code needs a requirements.txt file or not
    if "yes" in requirements_needed_or_not.lower().strip():
      # create the requirements.txt file
      sandbox_requirement_file = f"{script_folder}/{llm_name}.txt"
      with open(sandbox_requirement_file, "w", encoding="utf-8") as req:
         req.write(f"# This requirements for the code generated by {llm_name}\n")
         req.write(requirements_txt_content.strip())
    # return with name of the requirement file in the states
    return {"messages": [{"role": "ai", "content": json.dumps({"success": {"requirements": sandbox_requirement_file, "llm_name": llm_name}})}]}
  except Exception as e:
    return {"messages": [{"role": "ai", "content": json.dumps({"error": "An error occured while trying to check if code needs requirements.txt file created or not:{e}"})}]}

# will execute code in docker sandbox
def code_execution_node(state: StateMessages):
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
          stdout, stderr = run_script_in_docker(f"{llm_name}_dockerfile", f"{agents_scripts_dir}/{agent_script_file_name}", f"{agents_scripts_dir}/{requirements_name}")
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
  set_key(".var.env", "RETRY_CODE_EXECUTION", str(retry_code_execution))
  load_dotenv(dotenv_path=".vars.env", override=True)

  # send the results to next node
  return {"messages": [{"role": "ai", "content": json.dumps({"stdout": stdout_list_dict, "stderr": stderr_list_dict, "exception": exception_error_log_list_dict})}]}   

# code has successfully being executed
def successful_code_execution_management(state: MessagesState):
  messages = state["messages"]
  stdout = json.loads(messages[-1].content)["stdout"]
  # create a report from here so save this successful graph code execution to an env var for business logic side to get it and forward to report_creation node
  sek_key(".vars.env", "CODE_EXECUTION_GRAPH_RESULT", json.dumps({"success": stdout}))
  load_dotenv(dotenv_path=".vars.env", override=True)
  return {"messages": [{"role": "system", "content": json.dumps({"success": stdout})}]}

# will analyse error returned by code execution
def error_analysis_node(state: MessagesState):
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
  query = prompt_creation(error_analysis_node_prompt["human"], error=error_message, code=code, requirements=requirements_file, dockerfile=<>)
  '''
   don't forget that you also need to provide the dockerfile and use the same file names in formatting when rewriting those files so that the execution code node will just run the same file names but which have been overwritten
  '''
  try:
    code_analysis_response = structured_output_for_create_requirements_for_code(requirements_creation_class, query, error_analysis_node_prompt["system"]["template"])



 

######   CONDITIONAL EDGES FUNCTIONS

# after doc evaluation and judge
def rewrite_documentation_or_execute(state: MessagesState):
  # this is the jodge agent message decision
  messages = state["messages"]
  last_message = messages[-1].content
  decision = json.loads(last_message)
  # decision["DECISION"] rewrite/generate ; decision["REASON"] long str ; decision["STAGE"] rewrite/internet
  decision_dict = json.loads(last_message)
  # we check retries are not null
  load_dotenv(dotenv_path=".vars.env", override=True)
  retry_doc_validation = int(os.getenv("RETRY_DOC_VALIDATION"))
  print("RETRY DOC VALIDATION: ", retry_doc_validation)
  if retry_doc_validation == 0:
    state["messages"].append({"role": "system", "content": "All retries have been consummed, failed to rewrite documentation"})
    return "error_handler"
  
  # decrease the retry value
  retry_doc_validation -= 1
  set_key(".var.env", "RETRY_DOC_VALIDATION", str(retry_doc_validation))
  load_dotenv(dotenv_path=".vars.env", override=True)

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

# after code evaluator
def rewrite_or_create_requirements_decision(state: MessagesState):
  messages = state["messages"]
  # we receive a DICT[LIST[DICT]]
  evaluations =  json.loads(messages[-1].content)
  print("EVALUATIONS in conditional edge: ", evaluations, type(evalauations))

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
    set_key(".var.env", "RETRY_CODE_VALIDATION", str(retry_code_validation))
    load_dotenv(dotenv_path=".vars.env", override=True)
    
    return ["llama_3_8b_script_creator", "llama_3_70b_script_creator", "gemma_3_7b_script_creator"]
   
  if valid_code and retry_code_validation > 0:

    # decrease the retry value
    retry_code_validation -= 1
    set_key(".var.env", "RETRY_CODE_VALIDATION", str(retry_code_validation))
    load_dotenv(dotenv_path=".vars.env", override=True)

    # we loop over the valid code and save those to files that will be executed in next graphm we might need to put those in a folder nested inside the docker stuff
    for elem in valid_code:
      count = 0
      for llm, script in elem:
        count += 1
        with open(f"./docker_agent/agents_scripts/agent_code_execute_in_docker_{llm}_{count}.py", "w", encoding="utf-8") as llm_script:
          llm_script.write("#!/usr/bin/env python3")
          llm_script.write(f"# code from: {llm} LLM Agent\n# User initial request: {os.getenv('USER_INITIAL_REQUEST')}\n'''This code have been generated by an LLM Agent'''\n\n")
          llm_script.write(script)
     
    print("All Python script files are ready to be executed!")
     
    # append the valid code to state messages LIST[DICT]
    state["messages"].append({"role": "system", "content": json.dumps(valid_code)})
    
    # go to create requirement node
    return "create_requirements_for_code"

# after code execution
def success_execution_or_retry_or_return_error(state: MessagesState):
  messages = state["messages"]
  # unwrap the previous output results of code execution
  # LIST[DICT] : {"stdout": stdout_list_dict, "stderr": stderr_list_dict, "exception": exception_error_log_list_dict}
  code_execution_result = json.laod(messages[-1].content)

  # get the results list dicts having keys: script_file_path, llm_script_orign and output, check output if it has success, error or exception in the message
  # if we have at least one good code execution we want to analyze it with next node and know which code it was, from which llm
  if code_execution_result["stdout"]:
    # we have some code that has been executed properly and now want to send to next node that will sort it out in order to decide which code will be rendered to user
    return "successful_code_execution_management" # to be created
  
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
  
  if "success" in outcome: # Dict[Dict[name, reason]]
    # update state with the name of the llm only that will be passed to next node so it can find that file code and execute it in docker
    state["messages"].append({"role": "system", "content": json.dumps({"llm_name": outcome["success"]["llm_name"]})})
    # go to code execution node
    return "create_requirements_for_code"
  
  elif "one_script_only" in outcome: # Dict[name, script]
    # # update state with the name of the llm only that will be passed to next node so it can find that file code and execute it in docker
    for  k, _ in outcome.items():
      llm_name = k
    state["messages"].append({"role": "system", "content": json.dumps({"llm_name": llm_name})})
    # go to code execution node
    return "create_requirements_for_code"
  
  else 
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
workflow,add_node("documentation_steps_evaluator_and_doc_judge", documentation_steps_evaluator_and_doc_judge)
workflow.add_node("llama_3_8b_script_creator", llama_3_8b_script_creator)
workflow.add_node("llama_3_70b_script_creator", llama_3_70b_script_creator)
workflow.add_node("gemma_3_7b_script_creator", gemma_3_7b_script_creator)
workflow.add_node("code_evaluator_and_final_script_writer", code_evaluator_and_final_script_writer)
workflow.add_node("choose_code_to_execute_node_if_many", choose_code_to_execute_node_if_many)
workflow.add_node("create_requirements_for_code", create_requirements_for_code)
workflow.add_node("code_execution_node", code_execution_node)
workflow.add_node("successful_code_execution_management", successful_code_execution_management) # need to be created
workflow.add_node("error_analysis_node", error_analysis_node) # need to be created
workflow.add_node("report_creation_node", report_creation_node) # need to be created
workflow.add_node("inter_graph_node", inter_graph_node)
'''
  - next graph need an agent that will create the requirement.txt file for the code created here 
  - and another that will execute the code in docker 
  - and a report agent to make a report on code execution for end user response to request
'''

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
  rewrite_documentation_or_execute,
)
workflow.add_edge("llama_3_8b_script_creator", "code_evaluator_and_final_script_writer")
workflow.add_edge("llama_3_70b_script_creator", "code_evaluator_and_final_script_writer")
workflow.add_edge("gemma_3_7b_script_creator", "code_evaluator_and_final_script_writer")
workflow.add_condition_edge(
  "code_evaluator_and_final_script_writer",
  rewrite_or_create_requirements_decision
)
workflow.add_conditional_edge(
  "choose_code_to_execute_node_if_many",
  create_requirements_or_error
)
workflow.add_conditional_edge(
  "create_requirements_for_code",
  execute_code_or_error
)
workflow.add_condition_edge(
  "code_execution_node",
  success_execution_or_retry_or_return_error
)
workflow.add_edge("successful_code_execution_management", "inter_graph_node")
workflow.add_condition_edge(
  "error_analysis_node", # to be created
  rewrite_documentation_or_rewrite_code_or_return_error # to be created
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


