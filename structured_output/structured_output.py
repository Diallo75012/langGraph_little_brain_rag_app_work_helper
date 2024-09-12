import os
# Structured Output First Way
from langchain.output_parsers import PydanticOutputParser #
from langchain_core.prompts import PromptTemplate #
from langchain_core.pydantic_v1 import BaseModel, Field, validator #
# LLMs
'''
from llms.llms import (
  groq_llm_mixtral_7b,
  groq_llm_llama3_8b,
  groq_llm_llama3_8b_tool_use,
  groq_llm_llama3_70b,
  groq_llm_llama3_70b_tool_use,
  groq_llm_gemma_7b,
)
'''
from typing import Literal, TypedDict, Dict, List, Tuple, Any, Optional, Union
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# load env vars
load_dotenv(dotenv_path='.env', override=False)

groq_llm_mixtral_7b = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_MIXTRAL_7B"),
max_tokens=int(os.getenv("GROQ_MAX_TOKEN")),)
groq_llm_llama3_8b = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_LLAMA3_8B"), max_tokens=int(os.getenv("GROQ_MAX_TOKEN")),)
groq_llm_llama3_8b_tool_use = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_LLAMA3_8B_TOOL_USE"), max_tokens=int(os.getenv("GROQ_MAX_TOKEN")),)
groq_llm_llama3_70b = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_LLAMA3_70B"), max_tokens=int(os.getenv("GROQ_MAX_TOKEN")),)
groq_llm_llama3_70b_tool_use = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_LLAMA3_70B_TOOL_USE"), max_tokens=int(os.getenv("GROQ_MAX_TOKEN")),)
groq_llm_gemma_7b = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_GEMMA_7B"), max_tokens=int(os.getenv("GROQ_MAX_TOKEN")),)

# structured classes
class ReportAnswerCreation(BaseModel):
    """Create a detailed professional report using markdown"""
    title: str = Field(default="", description="From the query make a very engaging title int he form of a question using markdown format")
    advice: str = Field(default="", description="Your advice about the subject of the query in a markdown format with title and answer")
    answer: str = Field(default="", description="The detailed answer to user query in a mardowm format")
    bullet_points: str = Field(default="", description="Bullet points with some examples in a markdown format")


#
def structured_output_for_agent(structured_class: ReportAnswerCreation, query: str):
  # Set up a parser + inject instructions into the prompt template.
  parser = PydanticOutputParser(pydantic_object=structured_class)

  prompt = PromptTemplate(
    template="Answer the query in the form of a detailed report.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
  )

  # And a query intended to prompt a language model to populate the data structure.
  prompt_and_model = prompt | groq_llm_mixtral_7b | parser
  response = prompt_and_model.invoke({"query": query})
  return response


if __name__ == "__main__":

  query = '''question:'What are the different types of thermoceptors?'. Retrieved answer form our documentation: {"first_graph_message": "What are the different types of thermoceptors?", "second_graph_message": "What are the different types of thermoceptors?", "last_graph_message": "success_hash: [[{\'UUID\': \'81f3bd1a-2258-42e7-aecf-545c6d5ed4f2\', \'score\': 0.6005029563884516, \'content\': \'Sure, let\\\'s discuss the sensation of a \"burning\" feeling. This type of heat is often associated with inflammation or injury to the body. It can be caused by various factors such as exposure to chemicals, extreme temperatures, or medical conditions. The burning sensation is the body\\\'s way of signaling damage or irritation to the affected area. It can range from mild to severe and may be accompanied by other symptoms such as redness, swelling, or pain. In some cases, the burning feeling may be a sign of a more serious condition, so it is essential to seek medical attention if it persists or is accompanied by other symptoms.\', \'row_data\': {\'id\': \'81f3bd1a-2258-42e7-aecf-545c6d5ed4f2\', \'doc_name\': \'docs/feel_temperature.pdf\', \'title\': \'In the previous discussion, we explored the concept of how ion channels help us feel one type of hot sensation. However, there is another kind of heat that brings a burning feeling, which is a completely different experience. In this topic, we will dive deeper into the world of ion channels and discover how they contribute to our perception of burning heat.\', \'content\': \'Sure, let\\\'s discuss the sensation of a \"burning\" feeling. This type of heat is often associated with inflammation or injury to the body. It can be caused by various factors such as exposure to chemicals, extreme temperatures, or medical conditions. The burning sensation is the body\\\'s way of signaling damage or irritation to the affected area. It can range from mild to severe and may be accompanied by other symptoms such as redness, swelling, or pain. In some cases, the burning feeling may be a sign of a more serious condition, so it is essential to seek medical attention if it persists or is accompanied by other symptoms.\'}}, {\'UUID\': \'9633545f-9d54-4f6d-b454-28968ac42150\', \'score\': 0.6005029563884516, \'content\': \'The challenge is converting thermal and kinetic energy into electrical signals for the brain. Receptors are involved in signal transmission, and in temperature perception, this is no different.\', \'row_data\': {\'id\': \'9633545f-9d54-4f6d-b454-28968ac42150\', \'doc_name\': \'docs/feel_temperature.pdf\', \'title\': \'\"Thermoreceptors: Unraveling the Conversion of Thermal Energy to Electrical Signals\"\', \'content\': \'The challenge is converting thermal and kinetic energy into electrical signals for the brain. Receptors are involved in signal transmission, and in temperature perception, this is no different.\'}}]]"}'''
  answer = structured_output_for_agent(ReportAnswerCreation, query)
  print(type(answer))
  print("FULL Answer", answer)
  print("\nTITLE: ", answer.title)
  print("\nADVICE: ", answer.advice)
  print("\nANSWER: ", answer.answer)
  print("\nBULLET POINTS: ", answer.bullet_points)

