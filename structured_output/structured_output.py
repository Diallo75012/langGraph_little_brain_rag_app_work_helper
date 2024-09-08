import os
# Structured Output First Way
from langchain.output_parsers import PydanticOutputParser #
from langchain_core.prompts import PromptTemplate #
from langchain_core.pydantic_v1 import BaseModel, Field, validator #
# LLMs
from llms import (
  groq_llm_mixtral_7b,
  groq_llm_llama3_8b,
  groq_llm_llama3_8b_tool_use,
  groq_llm_llama3_70b,
  groq_llm_llama3_70b_tool_use,
  groq_llm_gemma_7b,
)
from typing import Literal, TypedDict, Dict, List, Tuple, Any, Optional, Union
from dotenv import load_dotenv


# load env vars
load_dotenv(dotenv_path='.env', override=False)

'''
# Structured Output Classes
class <class_name>(BaseModel):
    """<docstring>"""
    <parameter>: <type> = Field(default="", description="")


# First Way Used Here As per test more consistent and reliable
# Set up a parser + inject instructions into the prompt template.
structured_output_<name_of_function>(<class_name>: class_name) -> Dict[str:Any]:

  parser = PydanticOutputParser(pydantic_object=<class_name>)

  prompt = PromptTemplate(
    template="Answer user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
  )

  # And a query intended to prompt a language model to populate the data structure.
  prompt_and_model = prompt | groq_llm_mixtral_7b | parser
  response = prompt_and_model.invoke({"query": "<user_query>"})

  return {"response": reponse}

'''
