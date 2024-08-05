import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from prompts import prompts
from app import prompt_creation, chat_prompt_creation
from langchain_core.output_parsers import JsonOutputParser
from prompts.prompts import test_prompt, test_system_prompt


load_dotenv()

groq_llm_mixtral_7b = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_MIXTRAL_7B"),
max_tokens=int(os.getenv("GROQ_MAX_TOKEN")),)

import re
from typing import Tuple
# Utility function to determine if the query contains a PDF or URL
def detect_content_type(query: str) -> Tuple:
    """
    Utility function to determine if the query contains a PDF or URL.

    Args:
    query (str): The user query string.

    Returns:
    tuple: A tuple containing the type of content ('pdf', 'url', or 'text') and the detected content (URL or query).
    """
    pdf_pattern = r"https?://[^\s]+\.pdf"
    url_pattern = r"https?://[^\s]+"

    if re.search(pdf_pattern, query):
        return 'pdf', re.search(pdf_pattern, query).group(0)
    elif re.search(url_pattern, query):
        return 'url', re.search(url_pattern, query).group(0)
    else:
        return 'text', query

def llm_call(query: str) -> str:
  message=[
          (
            "system",
             """
              - Identify if the user query have a .pdf file in it, or an url, or just text, also if there is any clear question to rephrase it in a easy way for an llm to be able to retrieve it easily in a vector db. 
              - If non of those found or any of those just put an empty string as value in the corresponding key in the response schema.
              - User might not clearly write the url if any. It could be identified when having this patterm: `<docmain_name>.<TLD: top level domain>`. Make sure to analyze query thouroughly to not miss any url.
              - Put your answer in this schema:
                {
                  'url': <url in user query make sure that it always has https. and get url from the query even if user omits the http>,
                  'pdf': <pdf full name or path in user query>,
                  'text': <if text only without any webpage url or pdf file .pdf path or file just put here the user query>,
                  'question': <if quesiton identified in user query rephrase it in an easy way to retrieve data from vector database search without the filename and in a good english grammatical way to ask question>
                }
              Answer only with the schema in markdown between ```python ```.
             """
           ),
          (
            "human", "I have been in the ecommerce site chikarahouses.com and want to know if they have nice product for my Japanese garden",
          )
         ]

  llm_called = groq_llm_mixtral_7b.invoke(message)

  print("LLM call answer: ", llm_called.content)

  llm_called_answer = llm_called.content.split("```")[1].strip("python").strip()
  return llm_called_answer


def test(query: str) -> str:
  system_prompt = {"template": """
              - Identify if the user query have a .pdf file in it, or an url, or just text, also if there is any clear question to rephrase it in a easy way for an llm to be able t>
              - If non of those found or any of those just put an empty string as value in the corresponding key in the response schema.
              - User might not clearly write the url if any. It could be identified when having this patterm: `<docmain_name>.<TLD: top level domain>`. Make sure to analyze query t>
              - Put your answer in this schema:
                  {
                    'url': <url in user query make sure that it always has https. and get url from the query even if user omits the http>,
                    'pdf': <pdf full name or path in user query>,
                    'text': <if text only without any webpage url or pdf file .pdf path or file just put here the user query>,
                    'question': <if quesiton identified in user query rephrase it in an easy way to retrieve data from vector database search without the filename and in a good engli>
                  }
              - Answer only with the schema in markdown between ```python ```.
             """,
             "input_variables": [],
  }
  print("System prompt: ", system_prompt)
  human_prompt = {"template": """
                  {query}
                """,
                "input_variables": ["query"],
  }
  print("Human prompt: ", human_prompt)
  chat_prompt = chat_prompt_creation(system_prompt, human_prompt, query=query.strip())
  print("Chat Prompt: ", chat_prompt)

  response = invoke_chain(chat_prompt, groq_llm_mixtral_7b, JsonOutputParser())
  # llm_called = groq_llm_mixtral_7b.invoke(chat_prompt)

  print("LLM call answer: ", response.content)

  llm_called_answer = response.content.split("```")[1].strip("python").strip()
  return llm_called_answer



# user_input = input("Enter You Query: ")
user_input = "I have been in the ecommerce site chikarahouses.com and want to know if they have nice product for my Japanese garden"
# print(detect_content_type(user_input.strip()))
#print(test(user_input.strip()))


a = prompt_creation(test_prompt, test_var="Junko")
print("A: ", a)
b = prompt_creation(test_system_prompt, test_var="Junko")
print("B: ", b)
c = chat_prompt_creation(test_system_prompt, test_prompt, test_var="Junko")
print("C: ", c)
