import os
# for typing func parameters and outputs and states
from typing import Dict, List, Tuple, Any, Optional
# one is @tool decorator and the other Tool class
from langchain_core.tools import tool, Tool
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langchain_community.tools import (
  # Run vs Results: Results have more information
  DuckDuckGoSearchRun,
  DuckDuckGoSearchResults
)
from llms.llms import (
  groq_llm_mixtral_7b,
  groq_llm_llama3_8b,
  groq_llm_llama3_8b_tool_use,
  groq_llm_llama3_70b,
  groq_llm_llama3_70b_tool_use,
  groq_llm_gemma_7b,
)
# DOCKER REMOTE CODE EXECUTION
# eg.: print(run_script_in_docker("test_dockerfile", "./app.py"))
# from docker_agent.execution_of_agent_in_docker_script import run_script_in_docker # returns `Tuple[str, str]` stdout,stderr
from dotenv import load_dotenv


# load env vars
load_dotenv(dotenv_path='.env', override=False)
load_dotenv(dotenv_path=".vars", override=True)


# TOOLS

## Internet Search Tool
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
    return {"messages": [search_results]}
# INTERNET TOOL NODES
tool_search_node = ToolNode([search])
# LLMs WITH BINDED TOOLS
llm_with_internet_search_tool = groq_llm_mixtral_7b.bind_tools([search]) 

## API call tool
@tool
def jokes(state: MessagesState = MessagesState()) -> List[str]:
  """
  APIs that provides random jokes.
    
  <choices of tools>
    <tool>
        <name>joke</name>
        <description>Provides random jokes.</description>>
    </tool>
  </choices of tools>
  """
  return {"messages": [{"role": "ai", "content": ["joke"]}]}
@tool
def agify(state: MessagesState = MessagesState()) -> List[str]:
  """
  APIs that predicts the age based on a given name.
    
  <choices of tools>
    <tool>
        <name>agify</name>
        <description>This API predicts the age based on a given name.</description>>
    </tool>
  </choices of tools>
  """
  return {"messages": [{"role": "ai", "content": ["agify"]}]}
@tool
def dogimages(state: MessagesState = MessagesState()) -> List[str]:
  """
  APIs that returns random images of dogs
    
  <choices of tools>
    <tool>
        <name>dogimage</name>
        <description>Returns random images of dogs</description>>
    </tool>
  </choices of tools>
  """
  return {"messages": [{"role": "ai", "content": ["dogimages"]}]}

# API TOOL CHOICE NODE
tool_agent_decide_which_api_node = ToolNode([jokes, agify, dogimages])
# LLM BINDED WITH TOOLS
llm_api_call_tool_choice = groq_llm_mixtral_7b.bind_tools([jokes, agify, dogimages])







