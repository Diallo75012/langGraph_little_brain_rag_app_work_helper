"""
Here we will put all function that will be used to analyze, decompose, rephrase or other type of operation on user initial query
"""
import os
import re
from typing import Tuple, List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from prompts.prompts import llm_call_prompt
from typing import Dict, Any, List, Optional


load_dotenv()

# Utility function to determine if the query contains a PDF or URL
def detect_content_type(llm: ChatGroq, query: str, prompt: Dict) -> str:
  """
  Utility function to determine if the query contains a PDF or URL and extract URL, PDF, Text, Question in s tructured way (JSON like).

  Args:
  llm ChatGroq: The LLM client model chosen
  query str: The user query string.
  prompt dict: the prompt chat template system/human and AI optionally 

  Returns:
  str: A str containing the type of content ('pdf', 'url', or 'text') like a JSON and the detected and format content for easy fetching
  """
  system_message_tuple = ("system",) + (prompt["system"]["template"],)
  human_message_tuple = ("human",) + (query.strip(),)
  messages = [system_message_tuple, human_message_tuple]
  # print("Messages: ", messages)
  llm_called = llm.invoke(messages)

  #print("LLM call answer: ", llm_called.content)

  llm_called_answer = llm_called.content.split("```")[1].strip("markdown").strip()
  return llm_called_answer

