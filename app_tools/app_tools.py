import os
# for llm call with func or tool and prompts formatting
from langchain_groq import ChatGroq
# one is @tool decorator and the other Tool class
from langchain_core.tools import tool, Tool
from langchain.tools import StructuredTool
from langchain_community.tools import (
  # Run vs Results: Results have more information
  DuckDuckGoSearchRun,
  DuckDuckGoSearchResults
) 
from langchain.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv


internet_search_tool = DuckDuckGoSearchRun()
tool_internet = Tool(
    name="duckduckgo_search",
    description="Search DuckDuckGO for recent results.",
    func=internet_search_tool.run,
)

# load env vars
load_dotenv(dotenv_path='.env', override=False)

groq_llm_mixtral_7b = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_MIXTRAL_7B"),
max_tokens=int(os.getenv("GROQ_MAX_TOKEN")),)
groq_llm_llama3_8b = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_LLAMA3_8B"), max_tokens=int(os.getenv("GROQ_MAX_TOKEN")),)
groq_llm_llama3_8b_tool_use = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_LLAMA3_8B_TOOL_USE"), max_tokens=int(os.getenv("GROQ_MAX_TOKEN")),)
groq_llm_llama3_70b = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_LLAMA3_70B"), max_tokens=int(os.getenv("GROQ_MAX_TOKEN")),)
groq_llm_llama3_70b_tool_use = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_LLAMA3_70B_TOOL_USE"), max_tokens=int(os.getenv("GROQ_MAX_TOKEN")),)
groq_llm_gemma_7b = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_GEMMA_7B"), max_tokens=int(os.getenv("GROQ_MAX_TOKEN")),)


# using `StructuredTool`
class UseQuery(BaseModel):
  query: str = Field(default="{query}", description="use query that need to be search through the internet to get more information about it and answer to user")

def search_tool(query: str) -> str:
   """A search tool used to query DuckDuckGo for search results when trying to find information from the internet."""
   # Tool logic here
   search = DuckDuckGoSearchRun()
   return search.run(query)

internet_search_query = StructuredTool.from_function(
  func=search_tool,
  name="internet search query tool",
  description=  """
    This tool will perform an internet search about user query.
  """,
  args_schema=UseQuery,
  # return_direct=True, # returns tool output only if no TollException raised
  # coroutine= ... <- you can specify an async method if desired as well
  # callback==callback_function # will run after task is completed
)


# will be used as `llm_with_internet_search_tool(query)`
llm_with_internet_search_tool = groq_llm_mixtral_7b.bind_tools([internet_search_tool, tool_internet, internet_search_query])




