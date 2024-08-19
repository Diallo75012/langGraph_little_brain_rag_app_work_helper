## PROMPT TEMPLATES

# Prompt Direct LLM Calls question/answer
test_prompt_siberia = {
  "template": "What is the weather like in {test_var} usually.\nAnswer only with the schema in markdown between ```markdown ```.",
  "input_variables": {"test_var": "Siberia"}
}
test_prompt_tokyo = {
  "template": "We is the weather like in Japan in summer usually.\nAnswer only with the schema in markdown between ```markdown ```.",
  "input_variables": {}
}

# Prompt System/Human/Ai
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





     
