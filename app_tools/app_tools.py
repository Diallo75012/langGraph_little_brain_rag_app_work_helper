import os
# for llm call with func or tool and prompts formatting
from langchain_groq import ChatGroq
# one is @tool decorator and the other Tool class
from langchain_core.tools import tool, Tool
from langchain.tools import StructuredTool
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langchain_community.tools import (
  # Run vs Results: Results have more information
  DuckDuckGoSearchRun,
  DuckDuckGoSearchResults
) 
from langchain.pydantic_v1 import BaseModel, Field
from llms import (
  groq_llm_mixtral_7b,
  groq_llm_llama3_8b,
  groq_llm_llama3_8b_tool_use,
  groq_llm_llama3_70b,
  groq_llm_llama3_70b_tool_use,
  groq_llm_gemma_7b,
)
from dotenv import load_dotenv


# load env vars
load_dotenv(dotenv_path='.env', override=False)

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
    return {"messages": [search_results]}

tool_search_node = ToolNode([search])

llm_with_internet_search_tool = groq_llm_mixtral_7b.bind_tools([search]) 



