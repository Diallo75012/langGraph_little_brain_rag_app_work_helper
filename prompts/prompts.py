## PROMPT TEMPLATES

# Prompt Direct question/answer LLM Calls
test_prompt_siberia = {
  "template": "What is the weather like in {test_var} usually.\nAnswer only with the schema in markdown between ```markdown ```.",
  "input_variables": {"test_var": "Siberia"}
}
test_prompt_tokyo = {
  "template": "We is the weather like in Japan in summer usually.\nAnswer only with the schema in markdown between ```markdown ```.",
  "input_variables": {}
}

# Prompt Chat System/Human/Ai LLM Calls
test_prompt_dakar = {
  "system": {
    "template": "You are an expert in climat change and will help by answering questions only about climat change. If any other subject is mentioned, you must answer that you don't know as you are only expert in climat change related questions.\nAnswer only with the schema in markdown between ```markdown ```.", 
    "input_variables": {}
  },
  "human": {
    "template": "I need two know if Dakar is going to struggle in the future beacause of the rise of ocean water level due to global warming? {format_of_answer}", 
    "input_variables": {"format_of_answer": "Format your answer please using only one sentence and three extra bullet points."}
  },
  "ai": {
    "template": "", 
    "input_variables": {}
  },
}
test_prompt_monaco = {
  "system": {
    "template": "You are an expert in climat change and will help by answering questions only about climat change. If any other subject is mentioned, you must answer that you don't know as you are only expert in climat change related questions.\nAnswer only with the schema in markdown between ```markdown ```.", 
    "input_variables": {}
  },
  "human": {
    "template": "I am following F1 and want to know when is the next Monaco GP and will {car_house} win again?", 
    "input_variables": {"car_house": "Ferrari"}
  },
  "ai": {
    "template": "I believe that Renault will win next {city} GP", 
    "input_variables": {"city": "Singapour"}
  },
}

# Query Analysis prompt. Needs to be put in a list on the function call `messages` and passed in hte function that transform it to a Tuple[str,str]
llm_call_prompt = [
  {"system": """Identify if the user query have a .pdf file in it, or an url, or just text, also if there is any clear question to rephrase it in a easy way for an llm to be able to retrieve it easily in a vector db.\n- If non of those found or any of those just put an empty string as value in the corresponding key in the response schema.\n- User might not clearly write the url if any. It could be identified when having this patterm: `<docmain_name>.<TLD: top level domain>`. Make sure to analyze query thouroughly to not miss any url.\n- Put your answer in this schema:\n{\n'url': <url in user query make sure that it always has https. and get url from the query even if user omits the http>,\n'pdf': <pdf full name or path in user query>,\n'text': <if text only without any webpage url or pdf file .pdf path or file just put here the user query>,\n'question': <if quesiton identified in user query rephrase it in an easy way to retrieve data from vector database search without the filename and in a good english grammatical way to ask question>\n}\nAnswer only with the schema in markdown between ```markdown ```."""},
  {"human": "{query}"}
]

# here we just extract the system template and will use tuple concatenation to make the message
detect_content_type_prompt = {
  "system": {
    "template": """
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
      Answer only with the schema in markdown between ```markdown ```.
      """, 
    "input_variables": {}
  },
  "human": {
    "template": "", 
    "input_variables": {}
  },
  "ai": {
    "template": "", 
    "input_variables": {}
  },
}

# summary of content and to create title
summarize_text_prompt = {
  "system": {
    "template": "Answer putting text in markdown tags ```markdown ``` using no more than {maximum} characters to summarize this: {row['text']}.", 
    "input_variables": {}
  },
  "human": {
    "template": "", 
    "input_variables": {}
  },
  "ai": {
    "template": "", 
    "input_variables": {}
  },
}

generate_title_prompt =   {
  "system": {
    "template": "Please create a title in no more than {maximum} characters for: {row['section']} - {row['text']}. Make sure  your answering using in markdown format. Make sure that your answer is contained between markdown tags like in this example: ```markdown'title of the text here' ```.", 
    "input_variables": {}
  },
  "human": {
    "template": "", 
    "input_variables": {}
  },
  "ai": {
    "template": "", 
    "input_variables": {}
  },
}

# for tools calls can be filled in the function or here create different ones
generate_from_empty_prompt = {
  "system": {
    "template": "", 
    "input_variables": {}
  },
  "human": {
    "template": "", 
    "input_variables": {}
  },
  "ai": {
    "template": "", 
    "input_variables": {}
  },
}


# prompt for node that creates report from retrieved data.
answer_user_with_report_from_retrieved_data_prompt = {
  "system": {
    "template": """You are an expert in writing markdown reports from information provided to you. You format it well and provide, title, paragraphs and bullet points. You also add your advice. Answer only with in markdown between ```markdown ```.""", 
    "input_variables": {}
  },
  "human": {
    "template": """Here is my data, please make a professional report from those informations using the markdown format.: 
      - Topic: {question}
      - Information to be extracted for more context for the topic: {info_data_retrieved} 
      - Internet extended actual information about topic: {internet_search_result}
    """, 
    "input_variables": {}
  },
  "ai": {
    "template": "", 
    "input_variables": {}
  },
}


# prompt for structured output OR here have a list of prompts for each structured output or use this empty one and fill it with messages and requirements in the node/tool function
structured_outpout_report_prompt = {
  "system": {
    "template": "Answer the query in the form of a detailed report.\n{format_instructions}\n{query}\n", 
    "input_variables": {}
  },
  "human": {
    "template": "", 
    "input_variables": {}
  },
  "ai": {
    "template": "", 
    "input_variables": {}
  },
}

# prompt for user input analysis
# prompt for documentation_writer that will be using structured output to have a mardown formatted documentation
get_user_input_prompt = {
  "system": {
    "template": "You are an expert in user query analysis. You identify if there any need from the user to create some Python script code or to only create some coding documentation or nothing like code generation or documentation is needed. You use markdown to answer.\n{format_instructions}\n{query}\n", 
    "input_variables": {}
  },
  "human": {
    "template": "User query is: <INITIAL USER QUERY>{user_initial_query}</INITIAL USER QUERY>. Analyze the query to see if we need to generate Python script code for it or not.  Or if we need ONLY documentation created or not. Or if we don't need any code nor documentation as it is just a general question not needing any code generation or code documentation.", 
    "input_variables": {"user_initial_query": ""}
  },
  "ai": {
    "template": "", 
    "input_variables": {}
  },
}

# prompt tool_api_choose_agent
tool_api_choose_agent_prompt = {
  "system": {
    "template": "", 
    "input_variables": {}
  },
  "human": {
    "template": "Analyze user query: {user_initial_query}. And choose the appropriate API tool to address user query.", 
    "input_variables": {"user_initial_query": ""}
  },
  "ai": {
    "template": "", 
    "input_variables": {}
  },
}

# prompt find_documentation_online_agent
find_documentation_online_agent_prompt = {
  "system": {
    "template": "", 
    "input_variables": {}
  },
  "human": {
    "template": "We have to find online how to make a Python script to make a simple API call and get the response in mardown to this: {last_message}. Here is the different APIs urls that we have: {APIs}. Select just the one corresponding accordingly to user intent: {user_initial_query}. And search how to make a Python script to call that URL and return the response using Python in markdown format.", 
    "input_variables": {"last_message": "", "APIs": "", "user_initial_query": ""}
  },
  "ai": {
    "template": "", 
    "input_variables": {}
  },
}

# prompt for documentation_writer that will be using structured output to have a mardown formatted documentation
documentation_writer_prompt = {
  "system": {
    "template": "You are an expert in Python code documentation writing and creation. You will analyse information for the required documentation and will understand what user wants. You write instruction like documentation so that LLM agent called will understand and provide the corresponding code. So just ouput the documentation with all steps for Python developer to understand how to write the script. Therefore, DO NOT write the script, just the documentation and guidance in how to do it in markdown format but return the output strictly as a valid JSON object. \n{format_instructions}\n{query}\n", 
    "input_variables": {}
  },
  "human": {
    "template": "I wanted a Python script in markdown to call an API: <INITIAL USER QUERY>{user_initial_query}</INITIAL USER QUERY>. Our agent chosen this api to satisfy user request: <API CHOICE>{api_choice}</API CHOICE>; and found some documentation online: <DOCUMENTATION FOUND ONLINE>{documentation_found_online}</DOCUMENTATION FOUND ONLINE>. Can you write in markdown format detailed documentation in how to write the script that will call the API chosen by user which you can get the reference from: <API LINKS>{apis_links}</API LINKS>.", 
    "input_variables": {"user_initial_query": "", "apis_links": "", "api_choice": "", "documentation_found_online": ""}
  },
  "ai": {
    "template": "", 
    "input_variables": {}
  },
}

# prompt for parallel llm script creators
script_creator_prompt = {
  "system": {
    "template": "", 
    "input_variables": {}
  },
  "human": {
    "template": "Create a Python script to call the API following the instructions. Make sure that it is in markdown format and have the right indentations and imports.", 
    "input_variables": {}
  },
  "ai": {
    "template": "", 
    "input_variables": {}
  },
}

# prompt for judge agent that will evaluate if documentation written by agent need to be rewritten of if it allow for agents to start writting code script based on that document
rewrite_or_create_api_code_script_prompt = {
  "system": {
    "template": "You are an expert in Python code documentation review. You decide if the documentation needs to be rewritten as it contains errors or is not explained properly for an LLM to generate a Python script based on those instructions OR you validate the documentation as it is satisfactory to user needs and LLM agents will be able to comprehend/understand the documentation and easily create code following those instructions.\n{format_instructions}\n{query}\n", 
    "input_variables": {}
  },
  "human": {
    "template": "We need to instruct LLMs to create a Python script to make API calls based on inital request which is: {user_initial_query}. Out of the APIs choices which were those ones: {apis_links}, LLM agents have chosen that one {api_choice}. Based on that our LLM Agent have created some Python document to instruct LLM agents how to create the script. Please see and analyse the following documentation generated then tell if it can be used to instruct LLM agents to make a Python script satisfying inital request using markdown to answer: <INSTRUCTION DOCUEMNTATION>{documentation}</INSTRUCTION DOCUMENTATION>.", 
    "input_variables": {"user_inital_query": "", "apis_links": "", "api_choice": "", "documentation": ""}
  },
  "ai": {
    "template": "", 
    "input_variables": {}
  },
}

# prompt for code script evaluation when receiving code from different nodes. We are injecting the human side of the prompt by formating a query that is injected tot he system prompt
code_evaluator_and_final_script_writer_prompt = {
  "system": {
    "template": "You are an expert in Python script code review. You decide if the code is valid or not by checking if it has th eright imports, does the job required, have good indentation and check anything else that is required to check to make sure it is a valid working and executable code as it is. Answer using markdown but return the output strictly as a valid JSON object as follows: {'validity': 'YES/NO', 'reason': 'Your explanation for the evaluation.'}.\n{format_instructions}\n{query}\n", 
    "input_variables": {}
  },
  "human": {
    "template": "We want to know if the script fulfills our initial intent which is: {user_initial_query}. We have chosen one apis to be called from this: {apis_links}, and, have chosen this one: {api_choice}. Please can you tell if the code is executable, have no errors and will be a valid to execute an API call using markdown to answer. Here is the code: <CODE>{code}</CODE>.", 
    "input_variables": {"user_inital_query": "", "apis_links": "", "api_choice": "", "code": ""}
  },
  "ai": {
    "template": "", 
    "input_variables": {}
  },
}

# prompt for code comparator that will choose only one script before notifying requirements.txt creator node.  We are injecting the human side of the prompt by formating a query that is injected tot he system prompt
choose_code_to_execute_node_if_many_prompt = {
  "system": {
    "template": "You are an expert in Python script code review. You will be presented different Python script names and their corresponding codes. You will analyze those thouroughly and decide which ONE, and ONLY ONE, is the best for what the user want to do.\n{format_instructions}\n{query}\n", 
    "input_variables": {}
  },
  "human": {
    "template": "We want to know which script fulfills the best initial intent which is: {user_initial_query}. Using markdown to answer analyze those codes having their name and corresponding codes:<CODES TO ANALYZE>{code}</CODES TO ANALYZE>.", 
    "input_variables": {"user_inital_query": "", "code": ""}
  },
  "ai": {
    "template": "", 
    "input_variables": {}
  },
}

# prompt to create requirements.txt file content
create_requirements_for_code_prompt = {
  "system": {
    "template": "You are an expert in Python script code review and requirements.txt file. You will be presented different Python script and will decide if it needs a requirements.txt file. If it need one you will provide the content of the corresponding requirements.txt in markdown format.\n{format_instructions}\n{query}\n", 
    "input_variables": {}
  },
  "human": {
    "template": "Can you please check if that code requires any requirements.txt, if yes, provide the content of that file using markdown:<CODES TO ANALYZE>{code}</CODES TO ANALYZE>.", 
    "input_variables": {"code": ""}
  },
  "ai": {
    "template": "", 
    "input_variables": {}
  },
}

# prompt to analyze code execution stderr
error_analysis_node_prompt = {
  "system": {
    "template": "You are an expert in Python script code execution error analysis. You will be presented an error message from Python script execution, with the corresponding script and the requirements.txt file content. You will check if the error come from the script or the requirements or both. Maybe it doesn't need any requirements.txt content as code can be natively executed by python 3.9 slim docker container. If it needs new script or requirements or both, you will provide the content of the corresponding in markdown format.\n{format_instructions}\n{query}\n", 
    "input_variables": {}
  },
  "human": {
    "template": "Can you please check this error message coming from Python script execution: {error}.\nHere is the code:<CODES TO ANALYZE>{code}</CODES TO ANALYZE>.\nAnd here is the requirements.txt file content: <REQUIREMENTS TO ANALYZE>{requirementss}</REQUIREMENTS TO ANALYZE>\n", 
    "input_variables": {"error": "", "code": "", "requirements": ""}
  },
  "ai": {
    "template": "", 
    "input_variables": {}
  },
}
