import os
import json
# LLM chat AI, Human, System
from langchain_core.messages import (
  AIMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage
)
from typing import Dict, List, Any, Optional, Union
# Prompts LAngchain and Custom
from langchain_core.prompts import PromptTemplate
from prompts.prompts import answer_user_with_report_from_retrieved_data_prompt
# Tools
from app_tools.app_tools import (
  # internet node & internet llm binded tool
  tool_search_node,
  llm_with_internet_search_tool
)
# Node Functions from App.py 
"""
Maybe will have to move those functions OR to a file having all node functions OR to the corresponding graph directly
"""
from app import (
  process_query,
  store_dataframe_to_db,
  custom_chunk_and_embed_to_vectordb
)
# LLMs
from llms import (
  groq_llm_mixtral_7b,
  groq_llm_llama3_8b,
  groq_llm_llama3_8b_tool_use,
  groq_llm_llama3_70b,
  groq_llm_llama3_70b_tool_use,
  groq_llm_gemma_7b,
)
# Custom States
from app_states.app_graph_states import GraphStatePersistFlow
# for graph creation and management
from langgraph.checkpoint import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
# display drawing of graph
from IPython.display import Image, display
# env vars
from dotenv import load_dotenv


# load env vars
load_dotenv(dotenv_path='.env', override=False)
laod_dotenv(dotenv_path=",vars", override=True)


"""

Here we will make the primary graph components and create the app that will be compile and called in the `app.py`

"""
import os
import json
# LLM chat AI, Human, System
from langchain_core.messages import (
  AIMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage
)
from typing import Dict, List, Any, Optional, Union
# Prompts LAngchain and Custom
from langchain_core.prompts import PromptTemplate
from prompts.prompts import answer_user_with_report_from_retrieved_data_prompt
# Tools
from app_tools.app_tools import (
  # internet node & internet llm binded tool
  tool_search_node,
  llm_with_internet_search_tool
)
# Node Functions from App.py 
"""
Maybe will have to move those functions OR to a file having all node functions OR to the corresponding graph directly
"""
from app import (
  process_query,
  store_dataframe_to_db,
  custom_chunk_and_embed_to_vectordb
)
# LLMs
from llms import (
  groq_llm_mixtral_7b,
  groq_llm_llama3_8b,
  groq_llm_llama3_8b_tool_use,
  groq_llm_llama3_70b,
  groq_llm_llama3_70b_tool_use,
  groq_llm_gemma_7b,
)
# Custom States
from app_states.app_graph_states import GraphStatePersistFlow
# for graph creation and management
from langgraph.checkpoint import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
# display drawing of graph
from IPython.display import Image, display
# env vars
from dotenv import load_dotenv


# load env vars
load_dotenv(dotenv_path='.env', override=False)
laod_dotenv(dotenv_path=",vars", override=True)

"""
Here we will create teh components of the primary graph and compile it.
It will be called from the `app.py`
"""

def primary_graph():
  print("Primary_Graph")
  count = 0
  for step in app.stream(
    {"messages": [SystemMessage(content="Primary Graph")]},
    config={"configurable": {"thread_id": int(os.getenv("THREAD_ID"))}}):
    count += 1
    if "messages" in step:
      output = beautify_output(step['messages'][-1].content)
      print(f"Step {count}: {output}")
      # update custom state graph messages list
      GraphStatePersistFlow.subgraph_messages.append(str(output))
    else:
      output = beautify_output(step)
      print(f"Step {count}: {output}")
      # update custom state graph messages list
      GraphStatePersistFlow.subgraph_messages.append(str(output))

  # subgraph drawing
  graph_image = app.get_graph().draw_png()
  with open("retrieval_subgraph.png", "wb") as f:
    f.write(graph_image)

  return "All Jobs And Graphs Done"

if __name__ == "__main__":

  # to test it standalone
  primary_graph()
