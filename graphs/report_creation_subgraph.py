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
from prompts.prompts import answer_user_with_report_from_retrieved_data_prompt
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
    rephrased_query = messages[-1].content
    retrieved_answers = json.loads(messages[-2].content)["retrieved_answers"]
    print("RETRIEVED ANSWERS: ", retrieved_answers)
    response = llm_with_internet_search_tool.invoke(f"to the query {rephrased_query} We found answers that are in this list: {retrieved_answers}. Analyze the answers retrieved thouroughly. Correlate the question to the answer retrieved if any correlation exist and that we can say that some answers are relevant. Find extra information by making an internet search about the relevant content retrieved or just to answer to the question the best.")
    return {"messages": [response]}

# to fetch query reformulated froms state
def fetch_answer_and_query_reformulated(state: MessagesState):
  # get retrieval result from previous graph `retrieval_subgraph`
  load_dotenv(dotenv_path=".vars.env", override=True)
  retrieved_answers_list = os.getenv("RETRIEVAL_GRAPH_RESULT").split("success:")[1]
  reformulated_initial_question = os.getenv("QUERY_REFORMULATED")
  return {"messages": [{"role": "ai", "content": json.dumps({"retrieved_answers": retrieved_answers_list})}, {"role": "ai", "content": reformulated_initial_question}]}


def answer_user_with_report(state: MessagesState):
  """
    See if we can use here our structured output in order to have the report the way we want it to be or we can use the previous llm agent nodes to have this structured output
  """
  messages = state['messages']
  # this is just to check the history of messages -1, -2 and -3
  print("Answer User With Report State Messages: ", "\nMessage[-3]: ", messages[-3], "\nMessage[-2]: ", messages[-2], "\nMessage[-1]: ", messages[-1],)

  # check if tool have been called or not. if it haven't been called the -2 will be the final answer else we keep it the same -1 is the final answer
  if "tool_calls" in messages[-2].additional_kwargs and messages[-2].additional_kwargs["tool_calls"] == []:
    print("Tool not called: (messages[-2].additional_kwargs['tool_calls'])   ", messages[-2].additional_kwargs["tool_calls"])
    # we try this when no tool has been used using prompting technique to request a report from llm
    try:
      last_message = messages[-1].content
      print("LLM CALLED LAST MESSAGE: ", last_message)
      response = groq_llm_mixtral_7b.invoke("I need a detailed report about, {last_message}. Format the report in markdown using title, parapgraphs, subtitles, bullet points and advice section to get a fully complete professional report. Put your report answer between markdown tags ```markdown ```.")
      formatted_answer = response.content.split("```")[1].strip("markdown").strip()
      # save reuslt to .var env file
      set_key(".vars.env", "REPORT_GRAPH_RESULT", f"success:{formatted_answer}")
      load_dotenv(dotenv_path=".vars.env", override=True)
      # create the report
      report_path = os.getenv("REPORT_PATH")
      with open(report_path, "w", encoding="utf-8") as report:
        report.write(f"# Question: `{os.getenv('QUERY_REFORMULATED')}`\n\n")
        report.write(last_message)
      return {"messages": [{"role": "ai", "content": str(formatted_answer)}]}
    except IndexError as e:
      formatted_answer = f" {e}: {response.content}"
      print(f"We found an error. answer returned by llm without markdown tags: {e}")
      # save reuslt to .var env file
      set_key(".vars.env", "REPORT_GRAPH_RESULT", f"error:{e}")
      load_dotenv(dotenv_path=".vars.env", override=True)
      return {"messages": [{"role": "ai", "content": f"error:{e}"}]}


  # otherwise we return -1 message as it is the tool answer and tool has been used
  try:
    print("Tool called!: (messages[-2].additional_kwargs['tool_calls'])   ", messages[-2].additional_kwargs["tool_calls"])
    last_message = messages[-2].content
    response = groq_llm_mixtral_7b.invoke("I need a detailed report about, {last_message}. Format the report in markdown using title, parapgraphs, subtitles, bullet points and advice section to get a fully complete professional report. Put your report answer between markdown tags ```markdown ```.")
    formatted_answer = response.content.split("```")[1].strip("markdown").strip()
    # save reuslt to .var env file
    set_key(".vars.env", "REPORT_GRAPH_RESULT", f"success:{formatted_answer}")
    load_dotenv(dotenv_path=".vars.env", override=True)
    # create the report
    report_path = os.getenv("REPORT_PATH")
    with open(report_path, "w", encoding="utf-8") as report:
      report.write(f"# Question: `{os.getenv('QUERY_REFORMULATED')}`\n\n")
      report.write(last_message)
    return {"messages": [{"role": "ai", "content": str(formatted_answer)}]}
  except IndexError as e:
    formatted_answer = response.content
    print(f"We found an error. answer returned by llm without markdown tags: {e}")
    # save reuslt to .var env file
    set_key(".vars.env", "REPORT_GRAPH_RESULT", f"error:{e}")
    load_dotenv(dotenv_path=".vars.env", override=True)
    return {"messages": [{"role": "ai", "content": f"error:{e}"}]}


# CONDITIONAL EDGES FUNCTIONS

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

# report data collection main functions
workflow.add_node("fetch_answer_and_query_reformulated", fetch_answer_and_query_reformulated)
# tools
workflow.add_node("internet_search_agent", internet_search_agent)
workflow.add_node("tool_search_node", tool_search_node)
# answer
workflow.add_node("answer_with_report", answer_user_with_report)

# edges
workflow.set_entry_point("fetch_answer_and_query_reformulated")
workflow.add_edge("fetch_answer_and_query_reformulated", "internet_search_agent")
# tool nodes
workflow.add_edge("internet_search_agent", "tool_search_node")
workflow.add_edge("tool_search_node", "answer_with_report")
# answer user
workflow.add_edge("answer_with_report", END)

# compile with memory
checkpointer = MemorySaver()
graph_report_creation = workflow.compile(checkpointer=checkpointer)


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

def report_creation_subgraph(retrieval_result):
  print("Retrieval Graph")
  count = 0
  for step in graph_report_creation.stream(
    {"messages": [SystemMessage(content=retrieval_result)]},
    config={"configurable": {"thread_id": int(os.getenv("THREAD_ID"))}}):
    count += 1
    if "messages" in step:
      output = beautify_output(step['messages'][-1].content)
      print(f"Step {count}: {output}")
    else:
      output = beautify_output(step)
      print(f"Step {count}: {output}")

  # subgraph drawing
  graph_image = graph_report_creation.get_graph().draw_png()
  with open("report_creation_subgraph.png", "wb") as f:
    f.write(graph_image)

  if "success" in os.getenv("REPORT_GRAPH_RESULT"):
    # tell to start `retrieval` graph
    return "success"
  return "error"

'''
# can use this to test this graph as a standalone
if __name__ == "__main__":

  report_creation_subgraph()
'''


