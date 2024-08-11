import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from prompts import prompts
# from app import prompt_creation, chat_prompt_creation, dict_to_tuple
from langchain_core.output_parsers import JsonOutputParser
from prompts.prompts import test_prompt_tokyo, test_prompt_siberia, test_prompt_monaco, test_prompt_dakar
#from app import invoke_chain
from typing import Dict, List, Any, Optional
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import time
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


load_dotenv()

groq_llm_mixtral_7b = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_MIXTRAL_7B"),
max_tokens=int(os.getenv("GROQ_MAX_TOKEN")),)
groq_llm_llama3_8b = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_LLAMA3_8B"), max_tokens=int(os.getenv("GROQ_MAX_TOKEN")),)

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
  #print("System prompt: ", system_prompt, "\n", dict_to_tuple(system_prompt))
  human_prompt = {"template": """
                  {query}
                """,
                "input_variables": ["query"],
  }
  print("Human prompt: ", human_prompt)
  # chat_prompt = chat_prompt_creation(system_prompt, human_prompt, query=query.strip())
  print("Chat Prompt: ", chat_prompt)

  #response = invoke_chain(chat_prompt, groq_llm_mixtral_7b, JsonOutputParser(), {"query":query.strip()})
  # llm_called = groq_llm_mixtral_7b.invoke(chat_prompt)

  print("LLM call answer: ", response.content)

  llm_called_answer = response.content.split("```")[1].strip("python").strip()
  return llm_called_answer



# user_input = input("Enter You Query: ")
user_input = "I have been in the ecommerce site chikarahouses.com and want to know if they have nice product for my Japanese garden"
# print(detect_content_type(user_input.strip()))
#print(test(user_input.strip()))

def call_chain(model: ChatGroq, prompt: PromptTemplate, prompt_input_vars: Optional[Dict], prompt_chat_input_vars: Optional[Dict]):
  # only to concatenate all input vars for the system/human/ai chat
  chat_input_vars_dict = {}
  for k, v in prompt_chat_input_vars.items():
    for key, val in v.items():
      chat_input_vars_dict[key] = val
  print("Chat input vars: ",  chat_input_vars_dict)

  # default question/answer
  if prompt and prompt_input_vars:
    print("Input variables found: ", prompt_input_vars)
    chain = ( prompt | model )
    response = chain.invoke(prompt_input_vars)
    print("Response: ", response, type(response))
    return response.content.split("```")[1].strip("python").strip()
    
  # special for chat system/human/ai
  elif prompt and prompt_chat_input_vars:
    print("Chat input variables found: ", prompt_chat_input_vars)
    chain = ( prompt | model )
    response = chain.invoke(chat_input_vars_dict)
    print("Response: ", response, type(response))
    return response.content.split("```")[1].strip("python").strip()

  print("Chat input variables NOT found or missing prompt!")
  chain = ( prompt | model )
  response = chain.invoke(input={})
  print("Response: ", response, type(response))
  return response.content.split("```")[1].strip("python").strip()

def make_normal_or_chat_prompt_chain_call(llm_client, prompt_input_variables_part: Dict, prompt_template_part: Optional[Dict], chat_prompt_template: Optional[Dict]):
  
  # default prompt question/answer
  if prompt_template_part:
    prompt = (
      PromptTemplate.from_template(prompt_template_part)
    )
    response = call_chain(llm_client, prompt, prompt_input_variables_part, {})
    return response
  
  # chat prompts question/answer system/human/ai
  elif chat_prompt_template:
    prompt = (
      SystemMessage(content=chat_prompt_template["system"]["template"]) + HumanMessage(content=chat_prompt_template["human"]["template"]) + AIMessage(content=chat_prompt_template["ai"]["template"])
    )
    response = call_chain(llm_client, prompt, {}, {"system": chat_prompt_template["system"]["input_variables"], "human": chat_prompt_template["human"]["input_variables"], "ai": chat_prompt_template["ai"]["input_variables"]})
    return response
  return {'error': "An error occured while trying to create prompts. You must provide either: `prompt_template_part` with `prompt_input_variables_part` or `chat_prompt_template` - in combinaison with llm_client which is always needed." }


print("SIBERIA: \n", make_normal_or_chat_prompt_chain_call(groq_llm_llama3_8b, test_prompt_siberia["input_variables"], test_prompt_siberia["template"], {}))
time.sleep(0.5)
print("TOKYO: \n", make_normal_or_chat_prompt_chain_call(groq_llm_llama3_8b, test_prompt_tokyo["input_variables"], test_prompt_tokyo["template"], {}))
time.sleep(0.5)
print("DAKAR: \n", make_normal_or_chat_prompt_chain_call(groq_llm_llama3_8b, {}, {}, test_prompt_dakar))
time.sleep(0.5)
print("MONACO: \n", make_normal_or_chat_prompt_chain_call(groq_llm_llama3_8b, {}, {}, test_prompt_monaco))









