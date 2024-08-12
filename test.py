import os
import time

from dotenv import load_dotenv

from langchain_groq import ChatGroq
# from app import prompt_creation, chat_prompt_creation, dict_to_tuple
from langchain_core.output_parsers import JsonOutputParser
#from app 
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


from typing import Dict, List, Any, Optional

import json
import ast

import re

from prompts.prompts import test_prompt_tokyo, test_prompt_siberia, test_prompt_monaco, test_prompt_dakar, detect_content_type_prompt
from lib_helpers.chunking_module import create_chunks
from lib_helpers.query_analyzer_module import detect_content_type
import requests
from bs4 import BeautifulSoup

load_dotenv()

groq_llm_mixtral_7b = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_MIXTRAL_7B"),
max_tokens=int(os.getenv("GROQ_MAX_TOKEN")),)
groq_llm_llama3_8b = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_LLAMA3_8B"), max_tokens=int(os.getenv("GROQ_MAX_TOKEN")),)


webpage_url = "https://blog.medium.com/how-can-i-get-boosted-33e743431419"

#response = get_final_df(groq_llm_mixtral_7b, webpage_url, 200, 20)
#print(response)

def string_to_dict(string: str) -> Dict[str, Any]:
    """
    Converts a string representation of a dictionary to an actual dictionary.
    
    Args:
    string (str): The string representation of a dictionary.
    
    Returns:
    Dict[str, Any]: The corresponding dictionary.
    """
    try:
        # Safely evaluate the string as a Python expression
        dictionary = ast.literal_eval(string)
        if isinstance(dictionary, dict):
            return dictionary
        else:
            raise ValueError("The provided string does not represent a dictionary.")
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"Error converting string to dictionary: {e}")


result =  string_to_dict(detect_content_type(groq_llm_mixtral_7b, "I want to know if chikarahouses.com is a concept that is based on the authentic furimashuguru of Japanese istokawa house", detect_content_type_prompt))

print("RESULT: ", result, type(result))




