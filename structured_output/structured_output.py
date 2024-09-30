import os
import json
# Structured Output First Way
from langchain.output_parsers import PydanticOutputParser #
from langchain_core.prompts import PromptTemplate #
from langchain_core.pydantic_v1 import BaseModel, Field, validator #
# LLMs
from langchain_groq import ChatGroq
from llms.llms import (
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

################################################ REPORT GRAPH STRUCTURES OUTPUT ###################################################
## REPORT 
# structured classe for report generation
class ReportAnswerCreationClass(BaseModel):
    """Create a detailed professional report using markdown"""
    title: str = Field(default="", description="From the query make a very engaging title int he form of a question using markdown format")
    advice: str = Field(default="", description="Your advice about the subject of the query in a markdown format with title and answer")
    answer: str = Field(default="", description="The detailed answer to user query in a mardowm format")
    bullet_points: str = Field(default="", description="Bullet points with some examples in a markdown format")


# function for report generation structured output 
def structured_output_for_agent(structured_class: ReportAnswerCreationClass, query: str, prompt_template_part: str) -> Dict:
  # Set up a parser + inject instructions into the prompt template.
  parser = PydanticOutputParser(pydantic_object=structured_class)

  prompt = PromptTemplate(
    template=prompt_template_part,
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
  )
  print("Prompt before call structured output: ", prompt)

  # And a query intended to prompt a language model to populate the data structure.
  prompt_and_model = prompt | groq_llm_llama3_70b | parser
  response = prompt_and_model.invoke({"query": query})
  response_dict = { 
    "TITLE": response.title,
    "ADVICE": response.advice,
    "ANSWER": response.answer,
    "BULLET_POINTS": response.bullet_points
  }
  return response_dict



##################################### CODE EXECUTION STRUCTURED OUTPUT CLASSES AND FUNCTIONS ##############################################
### USER INITIAL INPUT NEEDS OR NOT CODE CREATION AND EXECUTION OR JUST DOCUMENTATION
# structured output class for get_user_input management
class AnalyseUserInput(BaseModel):
    """Analyse user query in order to identify if only documentation is needed to be created or if Python script code creation is needed."""
    code: str = Field(default="NO", description="Say YES if user query needs Python script creation. Otherwise say NO. Answer using markdown")
    doc: str = Field(default="NO", description="Say YES if user query needs only documentation to be created and NOT Python code script. Otherwise say NO. Answer using markdown.")
    nothing: str = Field(default="NO", description="Analyze the query, and if no url or pdf have been identified in the query and no Python script code nor documentation is needed to be created say 'YES'. Meaning that it is just a simple question that doesn't require code or documentation creation.")
    pdforurl: str = Field(default="NO", description="Analyze the query and identify if there is any pdf or url in it. Answer 'YES' to tell that the query have a .pdf file mentioned in it or an url like address.")
    


# function for report generation structured output 
def structured_output_for_get_user_input(structured_class: AnalyseUserInput, query: str, prompt_template_part: str) -> Dict:
  # Set up a parser + inject instructions into the prompt template.
  parser = PydanticOutputParser(pydantic_object=structured_class)

  prompt = PromptTemplate(
    template=prompt_template_part,
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
  )
  print("Prompt before call structured output: ", prompt)

  # And a query intended to prompt a language model to populate the data structure.
  prompt_and_model = prompt | groq_llm_llama3_70b | parser
  response = prompt_and_model.invoke({"query": query})
  response_dict = { 
    "code": response.code,
    "doc": response.doc,
    "nothing": response.nothing,
    "pdforurl": response.pdforurl
  }
  print("'structured_output_for_get_user_input' structured output response:", response_dict)
  return response_dict

### DOCUMENT WRITER
# structured output class for documentation writer
class DocumentationWriter(BaseModel):
    """Writing Python documentation about how to make API call. Only the instructions is created for another LLM to be able to follow create the wanted scriptand have example."""
    documentation: str = Field(default="", description="Documentation and guidance for Python script creation in markdown format about what the user needs. Just ouput the documentation with all steps for Python developer to understand how to write the script.")


# function for report generation structured output 
def structured_output_for_documentation_writer(structured_class: DocumentationWriter, query: str, prompt_template_part: str) -> Dict:
  # Set up a parser + inject instructions into the prompt template.
  parser = PydanticOutputParser(pydantic_object=structured_class)

  prompt = PromptTemplate(
    template=prompt_template_part,
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
  )
  print("Prompt before call structured output: ", prompt)

  # And a query intended to prompt a language model to populate the data structure.
  prompt_and_model = prompt | groq_llm_llama3_70b | parser
  response = prompt_and_model.invoke({"query": query})
  print("REEEEEEESSSSPONSEEEEEEEEEEEEEEE:: ", response)
  try:
    response_dict = { 
      "documentation": response.documentation,
    }
    print("'structured_output_for_documentation_writer' structured output response:", response_dict)
    return response_dict
  
  # if it is not a valid JSON response we need render the full response and parse it using python splitting techniques
  except json.JSONDecodeError as e:
    with open("./logs/structured_output_logs.log", "a", encoding="utf-8") as conditional:
      conditional.write(f"\n\nInvalid JSON response: {response}. Error: {e}\n\n")
    raise ValueError(f"Invalid JSON: {e}")




### DOCUMENTATION EVALUATION
# structured output class for documentation steps evaluator and doc judge
class CodeDocumentationEvaluation(BaseModel):
    """Evaluate quality of documentation for code generation created for other LLM agents to be able to generate Python scripts following the those instructions."""
    decision: str = Field(default="", description="Analyse the documentation created instruction and evaluate if it needs to be written again  or if it is validated as good documentation. Answer 'rewrite' to request documentation to be witten again or 'generate' to validate as good documentation for LLM Agent to understand it and generate code easily following those instructions.")
    reason: str = Field(default="", description="Reasons motivating decision.")
    stage: str =  Field(default="", description="if decision is 'rewrite' indicate here which stage need to be done again: 'internet' for internet search to get more information as poorly informed or 'rewrite' for just rewriting the documentation in a better way.")

# function for report generation structured output 
def structured_output_for_documentation_steps_evaluator_and_doc_judge(structured_class: CodeDocumentationEvaluation, query: str, prompt_template_part: str) -> Dict:
  # Set up a parser + inject instructions into the prompt template.
  parser = PydanticOutputParser(pydantic_object=structured_class)

  prompt = PromptTemplate(
    template=prompt_template_part,
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
  )
  print("Prompt before call structured output: ", prompt)

  # And a query intended to prompt a language model to populate the data structure.
  prompt_and_model = prompt | groq_llm_llama3_70b | parser
  response = prompt_and_model.invoke({"query": query})
  response_dict = { 
    "decision": response.decision,
    "reason": response.reason,
    "stage": response.stage,
  }
  print("'structured_output_for_documentation_steps_evaluator_and_doc_judge' structured output response:", response_dict)
  return response_dict


### CREATE SCRIPTS 
# structured output class to create requirements.txt
class ScriptCreation(BaseModel):
    """Analyze requirements to produce a Python script that uses Python standard libraries"""
    script: str = Field(default="", description="The content of the Python script file with right synthaxe, indentation and logic. It should be executable as it is. Do not use any markdown code block delimiters (i.e., ``` and ```python) replace those ''. This value MUST be JSON serializable and deserializable, therefore, make sure it is well formatted.")



# function for report generation structured output 
def structured_output_for_script_creator(structured_class: ScriptCreation, query: str, example_json: str, prompt_template_part: str, llm: ChatGroq) -> Dict:
  # Set up a parser + inject instructions into the prompt template.
  parser = PydanticOutputParser(pydantic_object=structured_class)

  prompt = PromptTemplate(
    template=prompt_template_part,
    input_variables=["query", "example_json"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
  )
  print("Prompt before call structured output: ", prompt)

  # And a query intended to prompt a language model to populate the data structure. groq_llm_llama3_70b as many code sent so long context
  prompt_and_model = prompt | llm | parser
  response = prompt_and_model.invoke({"query": query, "example_json": example_json})

  # Preprocess the response to remove markdown code block indicators
  processed_response = response.script.replace('```python', '').replace('```', '').strip()

  response_dict = { 
    "script": processed_response,
  }
  print("'structured_output_for_script_creator' structured output response:", response_dict)
  return response_dict



### CODE EVALUATION
# structured output class for code evaluator and final script writer
class CodeScriptEvaluation(BaseModel):
    """Evaluate quality of Python script code created to make API call."""
    validity: str = Field(default="", description="Say 'yes' if Python script code is evaluated as well written, otherwise 'no'.")
    reason: str = Field(default="", description="Tell reason why the code is evaluated as valid or not.")


# function for report generation structured output 
'''
 here we need an example of the required output as the LLM is not escaping characters correctly: example_json passed as json.dumps(example_json)
'''
def structured_output_for_code_evaluator_and_final_script_writer(structured_class: CodeScriptEvaluation, query: str, example_json: str, prompt_template_part: str) -> Dict:
  # Set up a parser + inject instructions into the prompt template.
  parser = PydanticOutputParser(pydantic_object=structured_class)

  prompt = PromptTemplate(
    template=prompt_template_part,
    input_variables=["query", "example_json"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
  )
  print("Prompt before call structured output: ", prompt)

  # And a query intended to prompt a language model to populate the data structure.
  try:
    prompt_and_model = prompt | groq_llm_llama3_70b | parser
  except Exception as e:
    print("prompt_and_model = prompt | groq_llm_llama3_70b | parser ERROR: ", e)
    
  try:
    response = prompt_and_model.invoke({"query": query, "example_json": example_json})
    print("tructured_output_for_code_evaluator_and_final_script_writer RESPONSE: ", response)
  except Exception as e:
    print("Structured Output Response ERROR: ", e)
 
  response_dict = { 
    "validity": response.validity,
    "reason": response.reason,
  }
  print("'structured_output_for_code_evaluator_and_final_script_writer' structured output response:", response_dict)
  return response_dict

### CHOOSE CODE BEST CODE AMONG SEVERAL CODE SNIPPETS
# structured output class to choose best code
class CodeComparison(BaseModel):
    """Compares Python scripts to decide which one is the best if we had to choose only one of those."""
    name: str = Field(default="", description="Codes are labelled with names. Answer the name of the code that you have selected ad being the best out the choices.")
    reason: str = Field(default="", description="Tell reason why you chose that llm code among the different code snippets analyzed.")


# function for report generation structured output 
def structured_output_for_agent_code_comparator_choice(structured_class: CodeComparison, query: str, prompt_template_part: str) -> Dict:
  # Set up a parser + inject instructions into the prompt template.
  parser = PydanticOutputParser(pydantic_object=structured_class)

  prompt = PromptTemplate(
    template=prompt_template_part,
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
  )
  print("Prompt before call structured output: ", prompt)

  # And a query intended to prompt a language model to populate the data structure. groq_llm_llama3_70b as many code sent so long context
  prompt_and_model = prompt | groq_llm_llama3_70b | parser
  response = prompt_and_model.invoke({"query": query, "name_choices": name_choices_json})
  response_dict = { 
    "llm_name": response.name,
    "reason": response.reason,
  }
  print("'structured_output_for_agent_code_comparator_choice' structured output response:", response_dict)
  return response_dict

### CREATE REQUIREMENTS.TXT
# structured output class to create requirements.txt
class CodeRequirements(BaseModel):
    """Analyze Python script to determine what should be in a corresponding requirements.txt file."""
    requirements: str = Field(default="", description="A markdown representation of what should be content of the code corresponding requirements.txt file content, with right versions and format of a requirements.txt file content.")
    needed: str = Field(default="", description="Answer 'YES' or 'NO' depending on if the code requires a requirements.txt file.")



# function for report generation structured output 
def structured_output_for_create_requirements_for_code(structured_class: CodeRequirements, query: str, prompt_template_part: str) -> Dict:
  # Set up a parser + inject instructions into the prompt template.
  parser = PydanticOutputParser(pydantic_object=structured_class)

  prompt = PromptTemplate(
    template=prompt_template_part,
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
  )
  print("Prompt before call structured output: ", prompt)

  # And a query intended to prompt a language model to populate the data structure. groq_llm_llama3_70b as many code sent so long context
  prompt_and_model = prompt | groq_llm_llama3_70b | parser
  response = prompt_and_model.invoke({"query": query})
  response_dict = { 
    "requirements": response.requirements,
    "needed": response.needed,
  }
  print("'structured_output_for_create_requirements_for_code' structured output response:", response_dict)
  return response_dict


### ANALYZE CODE EXECUTION STDERR ERRORS
# structured output class to analyze error after code execution in docker
class CodeErrorAnalysis(BaseModel):
    """Analyze Python script, user query, requirements.txt file if any and error message from code execution to come up with new markdown Python script with corresponding requirements.txt only if needed."""
    requirements: str = Field(default="", description="A markdown requirements.txt content corresponding to new Python script only if needed. Or a correction of the previous requirements.txt if error comes from it.")
    script: str = Field(default="", description="New Python script that addresses the error in markdown format or the previous script if the error wasn't coming from the code but from the requirements.txt content.")
    needed: str = Field(default="", description="Answer 'YES' or 'NO' depending on if the code requires a requirements.txt file.")


# function for report generation structured output 
def structured_output_for_error_analysis_node(structured_class: CodeErrorAnalysis, query: str, prompt_template_part: str) -> Dict:
  # Set up a parser + inject instructions into the prompt template.
  parser = PydanticOutputParser(pydantic_object=structured_class)

  prompt = PromptTemplate(
    template=prompt_template_part,
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
  )
  print("Prompt before call structured output: ", prompt)

  # And a query intended to prompt a language model to populate the data structure. groq_llm_llama3_70b as many code sent so long context
  prompt_and_model = prompt | groq_llm_llama3_70b | parser
  response = prompt_and_model.invoke({"query": query})
  response_dict = { 
    "requirements": response.requirements,
    "script": response.script,
    "needed": response.needed,
  }
  print("'structured_output_for_error_analysis_node' structured output response:", response_dict)
  return response_dict

'''
if __name__ == "__main__":

  query = 'question:'What are the different types of thermoceptors?'. Retrieved answer form our documentation: {"first_graph_message": "What are the different types of thermoceptors?", "second_graph_message": "What are the different types of thermoceptors?", "last_graph_message": "success_hash: [[{\'UUID\': \'81f3bd1a-2258-42e7-aecf-545c6d5ed4f2\', \'score\': 0.6005029563884516, \'content\': \'Sure, let\\\'s discuss the sensation of a \"burning\" feeling. This type of heat is often associated with inflammation or injury to the body. It can be caused by various factors such as exposure to chemicals, extreme temperatures, or medical conditions. The burning sensation is the body\\\'s way of signaling damage or irritation to the affected area. It can range from mild to severe and may be accompanied by other symptoms such as redness, swelling, or pain. In some cases, the burning feeling may be a sign of a more serious condition, so it is essential to seek medical attention if it persists or is accompanied by other symptoms.\', \'row_data\': {\'id\': \'81f3bd1a-2258-42e7-aecf-545c6d5ed4f2\', \'doc_name\': \'docs/feel_temperature.pdf\', \'title\': \'In the previous discussion, we explored the concept of how ion channels help us feel one type of hot sensation. However, there is another kind of heat that brings a burning feeling, which is a completely different experience. In this topic, we will dive deeper into the world of ion channels and discover how they contribute to our perception of burning heat.\', \'content\': \'Sure, let\\\'s discuss the sensation of a \"burning\" feeling. This type of heat is often associated with inflammation or injury to the body. It can be caused by various factors such as exposure to chemicals, extreme temperatures, or medical conditions. The burning sensation is the body\\\'s way of signaling damage or irritation to the affected area. It can range from mild to severe and may be accompanied by other symptoms such as redness, swelling, or pain. In some cases, the burning feeling may be a sign of a more serious condition, so it is essential to seek medical attention if it persists or is accompanied by other symptoms.\'}}, {\'UUID\': \'9633545f-9d54-4f6d-b454-28968ac42150\', \'score\': 0.6005029563884516, \'content\': \'The challenge is converting thermal and kinetic energy into electrical signals for the brain. Receptors are involved in signal transmission, and in temperature perception, this is no different.\', \'row_data\': {\'id\': \'9633545f-9d54-4f6d-b454-28968ac42150\', \'doc_name\': \'docs/feel_temperature.pdf\', \'title\': \'\"Thermoreceptors: Unraveling the Conversion of Thermal Energy to Electrical Signals\"\', \'content\': \'The challenge is converting thermal and kinetic energy into electrical signals for the brain. Receptors are involved in signal transmission, and in temperature perception, this is no different.\'}}]]"}'
  answer = structured_output_for_agent(ReportAnswerCreationClass, query)
  print(type(answer))
  print("FULL Answer", answer)
  print("\nTITLE: ", answer.title)
  print("\nADVICE: ", answer.advice)
  print("\nANSWER: ", answer.answer)
  print("\nBULLET POINTS: ", answer.bullet_points)
'''
