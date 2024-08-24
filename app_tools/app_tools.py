import os
# for llm call with func or tool and prompts formatting
from langchain_groq import ChatGroq
# one is @tool decorator and the other Tool class
from langchain_core.tools import tool, Tool
from langchain_community.tools import (
  # Run vs Results: Results have more information
  DuckDuckGoSearchRun,
  DuckDuckGoSearchResults
) 

internet_search_tool = DuckDuckGoSearchRun()
tool_internet = Tool(
    name="duckduckgo_search",
    description="Search DuckDuckGO for recent results.",
    func=internet_search_tool.run,
)
from dotenv import load_dotenv


# load env vars
load_dotenv(dotenv_path='.env', override=False)

groq_llm_mixtral_7b = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_MIXTRAL_7B"),
max_tokens=int(os.getenv("GROQ_MAX_TOKEN")),)
groq_llm_llama3_8b = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_LLAMA3_8B"), max_tokens=int(os.getenv("GROQ_MAX_TOKEN")),)
groq_llm_llama3_8b_tool_use = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_LLAMA3_8B_TOOL_USE"), max_tokens=int(os.getenv("GROQ_MAX_TOKEN")),)
groq_llm_llama3_70b = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_LLAMA3_70B"), max_tokens=int(os.getenv("GROQ_MAX_TOKEN")),)
groq_llm_llama3_70b_tool_use = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_LLAMA3_70B_TOOL_USE"), max_tokens=int(os.getenv("GROQ_MAX_TOKEN")),)
groq_llm_gemma_7b = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_GEMMA_7B"), max_tokens=int(os.getenv("GROQ_MAX_TOKEN")),)


# can put more tools in the list and agent will see which one to use reading doctrings
internet = [tool_internet]
# will be used as `llm_with_internet_search_tool(query)`
llm_with_internet_search_tool = groq_llm_mixtral_7b.bind_tools(internet)
