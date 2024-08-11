"""
Here we will put all function that will be used to analyze, decompose, rephrase or other type of operation on user initial query
"""
import os
import re
from typing import Tuple, List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from prompts.prompts import llm_call_prompt


load_dotenv()

groq_llm_mixtral_7b = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_MIXTRAL_7B"),
max_tokens=int(os.getenv("GROQ_MAX_TOKEN")),)

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

def llm_call(query: str, prompt_llm_call: List[Tuple[str,str]]) -> str:
  messages=[
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
            "human", query.strip(),
          )
         ].format({"query":query.strip()})

  llm_called = groq_llm_mixtral_7b.invoke(messages)

  print("LLM call answer: ", llm_called.content)

  llm_called_answer = llm_called.content.split("```")[1].strip("python").strip()
  return llm_called_answer

