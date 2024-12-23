import os
import json
# tools
from app_tools.app_tools import (
  tool_search_node,
  llm_with_internet_search_tool
)
from llms.llms import (
  groq_llm_mixtral_7b,
  groq_llm_llama3_8b,
  groq_llm_llama3_8b_tool_use,
  groq_llm_llama3_70b,
  groq_llm_llama3_70b_tool_use,
  groq_llm_gemma_7b,
)
# structured output
from structured_output.structured_output import (
  # class created to structure the output
  ReportAnswerCreationClass,
  # function taking in class output structured and the query, returns dict
  structured_output_for_agent
)
# LLM chat call AI, System, Human
from langchain_core.messages import (
  AIMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage
)
# for graph creation and management
from langgraph.checkpoint import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
# helpers modules
from lib_helpers.query_matching import handle_query_by_calling_cache_then_vectordb_if_fail
from prompts.prompts import structured_outpout_report_prompt
# display drawing of graph
from IPython.display import Image, display
# env vars
from dotenv import load_dotenv, set_key


# load env vars
load_dotenv(dotenv_path='.env', override=False)
load_dotenv(dotenv_path=".vars.env", override=True)


# HELPER FUNCTIONS
def message_to_dict(message):
  if isinstance(message, (AIMessage, HumanMessage, SystemMessage, ToolMessage)):
    return {
      "content": message.content,
      "additional_kwargs": message.additional_kwargs,
      "response_metadata": message.response_metadata if hasattr(message, 'response_metadata') else None,
      "tool_calls": message.tool_calls if hasattr(message, 'tool_calls') else None,
      "usage_metadata": message.usage_metadata if hasattr(message, 'usage_metadata') else None,
      "id": message.id,
      "role": getattr(message, 'role', None),
    }
  return message

def convert_to_serializable(data):
  if isinstance(data, list):
    return [convert_to_serializable(item) for item in data]
  elif isinstance(data, dict):
    return {k: convert_to_serializable(v) for k, v in data.items()}
  elif isinstance(data, (AIMessage, HumanMessage, SystemMessage, ToolMessage)):
    return message_to_dict(data)
  return data

def beautify_output(data):
  serializable_data = convert_to_serializable(data)
  return json.dumps(serializable_data, indent=4)


# NODE FUNCTIONS

# For Internet Search Node
def internet_search_agent(state: MessagesState):
    messages = state['messages']
    print("message state -1: ", messages[-1].content, "\nmessages state -2: ", messages[-2].content)
    # print("messages from call_model func: ", messages)
    response = llm_with_internet_search_tool.invoke(messages[-1].content)
    if ("success_hash" or "success_semantic" or "success_vector_retrieved_and_cached") in messages[-1].content:
      print(f"\nAnswer retrieved, create schema for tool choice of llm, last message: {messages[-1].content}")
      response = llm_with_internet_search_tool.invoke(f"to the query {messages[-2].content} we found response in organization internal documents with content and source id: {messages[-1].content}. Analyze thouroughly the answer retrieved. Correlate the question to the answer retrieved. Find extra information by making an internet search about the content retrieved to answer the question the best.")
    # print("response from should_continue func: ", response)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

# for error handling Node
def error_handler(state: MessagesState):
  messages = state['messages']
  last_message = messages[-1].content
  
  with open("./logs/conditional_edge_logs.log", "a", encoding="utf-8") as conditional:
      conditional.write(f"\n\nerror handler called (from subgraph): {last_message}\n\n")

  return {"messages": [{"role": "ai", "content": f"An error occured, error message: {last_message}"}]}

# to fetch query reformulated froms state
def fetch_query_reformulated(state: MessagesState):
  messages = state['messages']
  last_message = messages[-1].content
  return {"messages": [{"role": "ai", "content": last_message}]}

'''
def answer_user_with_report(state: MessagesState):
  messages = state['messages']
  print("Answer User With Report State Messages: \n", "Message[-3]: ", messages[-3], "Message[-2]: ", messages[-2], "Message[-1]: ", messages[-1],)

  # check if tool have been called or not. if it haven't been called the -2 will be the final answer else we keep it the same -1 is the final answer
  if "tool_calls" in messages[-2].additional_kwargs and messages[-2].additional_kwargs["tool_calls"] == []:
    try:
      last_message = messages[-1].content
      response = groq_llm_mixtral_7b.invoke(f"I need a detailed report about {os.getenv('QUERY_REFORMULATED')} in markdown with title, pertinent answers and well formatted with bullet points. Put your report answer between markdown tags ```markdown ```: {last_message}")
      formatted_answer = response.content.split("```")[1].strip("markdown").strip()
    except IndexError as e:
      formatted_answer = response.content
      print(f"We found an error. answer returned by llm withotu markdown tags: {e}")
    return {"messages": [{"role": "ai", "content": f"success:{str(formatted_answer)}"}]}
  # otherwise we return -1 message as it is the tool answer
  try:
    last_message = messages[-2].content
    response = groq_llm_mixtral_7b.invoke(f"I need a detailed report about {os.getenv('QUERY_REFORMULATED')} in markdown with title, pertinent answers and well formatted with bullet points. Put your report answer between markdown tags ```markdown ```: {last_message}")
    formatted_answer = response.content.split("```")[1].strip("markdown").strip()
  except IndexError as e:
    formatted_answer = response.content
    print(f"We found an error. answer returned by llm without markdown tags: {e}")
  #formatted_answer_structured_output = response
  return {"messages": [{"role": "ai", "content": f"success:{str(formatted_answer)}"}]}
'''
def answer_user_with_report(state: MessagesState):

  messages = state['messages']
    
  try:
    if "tool_calls" in messages[-2].additional_kwargs and messages[-2].additional_kwargs["tool_calls"] != []:
      print("TOOL HAS BEEN CALLED: adding the internet search result in the query for report.")
      internet_search_result = messages[-1].content
      query = f"question: {os.getenv('QUERY_REFORMULATED')}. Extra information from internet: {internet_search_result}"    
    else:
      print("TOOL HASN'T BEEN CALLED: No internet search in the query.")
      query = f"question: {os.getenv('QUERY_REFORMULATED')}."
    
    # get the answer as we want it using structured output
    answer = structured_output_for_agent(ReportAnswerCreationClass, query, structured_outpout_report_prompt["system"]["template"])
    print("\n\n\nANSWER: ", answer, type(answer))
    """
      # Object returned by the structured output llm call
      { 
        "TITLE": response.title)
        "ADVICE": response.advice)
        "ANSWER": response.answer)
        "BULLET POINTS": response.bullet_points)
      }
    """
    
    # create the report
    report_path = os.getenv("REPORT_PATH")
    with open(report_path, "w", encoding="utf-8") as report:
      report.write(f"# Creditizens Answer\n\n")
      report.write(f"{answer['TITLE']}\n") 
      report.write(f"{answer['ADVICE']}\n")
      report.write(f"{answer['ANSWER']}\n")
      report.write(f"{answer['BULLET_POINTS']}")

    # save reuslt to .var env file for app business logic to know that next graph don't need to be started as report has been created
    set_key(".vars.env", "RETRIEVAL_GRAPH_RESULT", f"report:{report_path}")
    load_dotenv(dotenv_path=".vars.env", override=True)
    
    return {"messages": [{"role": "ai", "content": json.dumps(answer)}]}

  except Exception as e:
    print(f"We found an error with the structured output llm call: {e}")
    # save reuslt to .var env file
    set_key(".vars.env", "REPORT_GRAPH_RESULT", f"error:{e}")
    load_dotenv(dotenv_path=".vars.env", override=True)
    return {"messages": [{"role": "ai", "content": f"error:{e}"}]}

# answer_user Node function to be  adapted to this graph
def answer_user(state: MessagesState):
  messages = state['messages']  
  # we need to have this ai content message be a string otherwise we will get a pydantic validation error expecting a str after we could deserialize this using `json.laods`
  last_message = json.dumps({"first_graph_message": messages[0].content, "last_graph_message": messages[-1].content})
  print("LAST MESSAGE SET OF RETRIEVAL GRAPH (indexes[first graph messages 0, second graph message 1,last message -1]): ", last_message)
  # save reuslt to .var env file
  set_key(".vars.env", "RETRIEVAL_GRAPH_RESULT", f"success:{last_message}")
  load_dotenv(dotenv_path=".vars.env", override=True)

  return {"messages": [{"role": "ai", "content": last_message}]}

# CONDITIONAL EDGES FUNCTIONS
# `handle_query_by_calling_cache_then_vectordb_if_fail` conditional edge function
def handle_query_by_calling_cache_then_vectordb_if_fail_conditional_edge_decision(state: MessagesState):
  messages = state['messages']
  last_message = messages[-1].content
  with open("./logs/conditional_edge_logs.log", "a", encoding="utf-8") as conditional:
    conditional.write("\nhandle_query_by_calling_cache_then_vectordb_if_fail_conditional_edge_decision:\n")
    conditional.write(f"- last message: {last_message}")

  if ("error_hash" or "error_semantic" or "error_vector_retrieved_and_cached" or "error_vector" or "error_cache_and_vector_retrieval") in last_message:
    with open("./logs/conditional_edge_logs.log", "a", encoding="utf-8") as conditional:
      conditional.write(f"- error: {last_message}\n\n")
    return "error_handler"

  elif "success_hash" in last_message:
    with open("./logs/conditional_edge_logs.log", "a", encoding="utf-8") as conditional:
      conditional.write(f"- success_hash: {last_message}\n\n")
    #return "chunk_and_embed_from_db_data"
    #return "internet_search_agent"
    return "answer_user"

  elif "success_semantic" in last_message:
    with open("./logs/conditional_edge_logs.log", "a", encoding="utf-8") as conditional:
      conditional.write(f"- success_semantic: {last_message}\n\n\n\n")
    #return "internet_search_agent"
    return "answer_user"

  elif "success_vector_retrieved_and_cached" in last_message:
    with open("./logs/conditional_edge_logs.log", "a", encoding="utf-8") as conditional:
      conditional.write(f"- success_vector_retrieved_and_cached: {last_message}\n\n")
    #return "internet_search_agent"
    return "answer_user"

  elif "nothing_in_cache_nor_vectordb" in last_message:
    with open("./logs/conditional_edge_logs.log", "a", encoding="utf-8") as conditional:
      conditional.write(f"- nothing_in_cache_nor_vectordb: {last_message}\n\n\n\n")
    # will process user query and start the dataframe creation flow and embedding storage of that df
    return "internet_search_agent"

  else:
    with open("./logs/conditional_edge_logs.log", "a", encoding="utf-8") as conditional:
      conditional.write(f"- error last: {last_message}\n\n")
    return "error_handler"
'''
# reference to be used for last report message
def answer_user_with_report_from_retrieved_data(state: MessagesState):
  messages = state['messages']
  #print("Message state: ", messages)
  # # -4 user input, -3 data retrieved, -2 schema internet tool, -1 internet search result
  internet_search_result = messages[-1].content
  info_data_retrieved = messages[-3].content
  question = messages[-4].content
  prompt = answer_user_with_report_from_retrieved_data_prompt["human"]["template"]
  prompt_human_with_input_vars_filled = eval(f'f"""{prompt}"""')
  print(f"\nprompt_human_with_input_vars_filled: {prompt_human_with_input_vars_filled}\n")
  system_message = SystemMessage(content=prompt_human_with_input_vars_filled)
  human_message = HumanMessage(content=prompt_human_with_input_vars_filled)
  messages = [system_message, human_message]
  response = groq_llm_mixtral_7b.invoke(messages)
  #formatted_answer = response.content.split("```")[1].strip("markdown").strip()

  # couldn't get llm to answer in markdown tags???? so just getting content saved to states
  return {"messages": [{"role": "ai", "content": response.content}]}
'''


# Initialize states
workflow = StateGraph(MessagesState)

# retrieval main functions
workflow.add_node("fetch_query_reformulated", fetch_query_reformulated)
workflow.add_node("handle_query_by_calling_cache_then_vectordb_if_fail", handle_query_by_calling_cache_then_vectordb_if_fail)
# tools
workflow.add_node("internet_search_agent", internet_search_agent)
workflow.add_node("tool_search_node", tool_search_node)
# answer
workflow.add_node("answer_with_report", answer_user_with_report)
workflow.add_node("answer_user", answer_user)
# errors
workflow.add_node("error_handler", error_handler)

# edges
workflow.set_entry_point("fetch_query_reformulated")
workflow.add_edge("fetch_query_reformulated", "handle_query_by_calling_cache_then_vectordb_if_fail")
# `handle_query_by_calling_cache_then_vectordb_if_fail` edge
workflow.add_conditional_edges(
    "handle_query_by_calling_cache_then_vectordb_if_fail",
    handle_query_by_calling_cache_then_vectordb_if_fail_conditional_edge_decision,
    #"dataframe_from_query"
)
# tool nodes with report answer
workflow.add_edge("internet_search_agent", "tool_search_node")
workflow.add_edge("tool_search_node", "answer_with_report")
workflow.add_edge("answer_with_report", END)
# answer user to pass it to next graph which will make a report
workflow.add_edge("error_handler", "answer_user")
workflow.add_edge("answer_user", END)


# compile with memory
checkpointer = MemorySaver()
graph_retrieval = workflow.compile(checkpointer=checkpointer)


'''
# using: INVOKE
final_state = app.invoke(
  #{ "query": UserInput.user_initial_input },
  {"messages": [HumanMessage(content="initialize messages")]},
  config={"configurable": {"thread_id": 11}}
)

# Get the final message
final_message = final_state["messages"][-1].content
print("Final Message:", final_message)
'''

# using STREAM
# we can maybe get the uder input first and then inject it as first message of the state: `{"messages": [HumanMessage(content=user_input)]}`

# test_query for pdf: What is responsible for detecting temperature changes in our skin?
# answer expected" Specialized nerve endings called thermoreceptors are responsible for detecting temperature changes in our skin?

def retrieval_subgraph(user_query):
  print("Retrieval Graph")
  count = 0
  for step in graph_retrieval.stream(
    {"messages": [SystemMessage(content=user_query)]},
    config={"configurable": {"thread_id": int(os.getenv("THREAD_ID"))}}):
    count += 1
    if "messages" in step:
      output = beautify_output(step['messages'][-1].content)
      print(f"Step {count}: {output}")
    else:
      output = beautify_output(step)
      print(f"Step {count}: {output}")

  # subgraph drawing
  graph_image = graph_retrieval.get_graph().draw_png()
  with open("retrieval_subgraph.png", "wb") as f:
    f.write(graph_image)

  if "success" in os.getenv("RETRIEVAL_GRAPH_RESULT"):
    # tell to start `retrieval` graph
    return "report"
  elif "report" in os.getenv("RETRIEVAL_GRAPH_RESULT"):
    # tell to start `retrieval` graph
    return "report"
  else:
    return "error"

'''
# can use this to test this graph as a standalone
if __name__ == "__main__":

  retrieval_subgraph()
'''



