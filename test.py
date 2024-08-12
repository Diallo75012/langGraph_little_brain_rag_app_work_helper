import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
# from app import prompt_creation, chat_prompt_creation, dict_to_tuple
from langchain_core.output_parsers import JsonOutputParser
from prompts.prompts import test_prompt_tokyo, test_prompt_siberia, test_prompt_monaco, test_prompt_dakar, detect_content_type_prompt
#from app import invoke_chain
from typing import Dict, List, Any, Optional
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import time
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import json
import ast
from lib_helpers.webpage_parser import get_final_df


load_dotenv()

groq_llm_mixtral_7b = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_MIXTRAL_7B"),
max_tokens=int(os.getenv("GROQ_MAX_TOKEN")),)
groq_llm_llama3_8b = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_LLAMA3_8B"), max_tokens=int(os.getenv("GROQ_MAX_TOKEN")),)


webpage_url = "https://blog.medium.com/how-can-i-get-boosted-33e743431419"

response = get_final_df(groq_llm_mixtral_7b, webpage_url, 200, 20)
print(response)


# function to add at the end of get_final_df to know if we had to deal with pdf or url:
import re

def is_url_or_pdf(input_string: str) -> str:
    """
    Checks if the given string is a URL or a PDF document.
    
    Args:
    input_string (str): The input string to check.
    
    Returns:
    str: "url" if the string is a URL, "pdf" if it's a PDF document, and "none" if it's neither.
    """
    try:
        # Regular expression pattern for URLs
        url_pattern = re.compile(
            r'^(https?:\/\/)?'                # http:// or https:// (optional)
            r'([a-z0-9]+[.-])*[a-z0-9]+\.[a-z]{2,6}'  # domain name
            r'(:[0-9]{1,5})?'                 # port (optional)
            r'(\/.*)?$',                      # path (optional)
            re.IGNORECASE
        )

        # Check if it's a URL
        if url_pattern.match(input_string):
            return "url"

        # Check if it's a .pdf document URL or file path
        if input_string.lower().endswith(".pdf"):
            return "pdf"

        # Neither URL nor PDF
        return "none"
    
    except Exception as e:
        # Catch any unexpected errors and return "none"
        print(f"An error occurred: {e}")
        return "none"









