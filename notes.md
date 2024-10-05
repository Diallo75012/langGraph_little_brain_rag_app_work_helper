# LangGraph

## LangGraph boilerplate output objects at every stage

#### here is tool creation so a funciton that is having the decorator `@tool` (`from langchain_core.tools import tool`)
```python 
tools = [search]
# Outputs
Tools:  [StructuredTool(name='search', description='Call to surf the web.', args_schema=<class 'pydantic.v1.main.searchSchema'>, func=<function search at 0x7a18160244c0>)]
```
#### creation of a node type tool
```python
tool_node = ToolNode(tools)
# outputs
tool_node:  tools(recurse=True, tools_by_name={'search': StructuredTool(name='search', description='Call to surf the web.', args_schema=<class 'pydantic.v1.main.searchSchema'>, func=<function search at 0x7a18160244c0>)}, handle_tool_errors=True)
```
#### creatiion of model adding the `bind_tool` and passing the tool in it in order to have the Openai tool mimmicing done and not get the error `missing bind` or something like that 
```python
model = groq_llm_llama3_70b.bind_tools(tools)
# outputs
Model with bind_tools:  bound=ChatGroq(client=<groq.resources.chat.completions.Completions object at 0x7a1816042680>, async_client=<groq.resources.chat.completions.AsyncCompletions object at 0x7a18160419c0>, model_name='llama3-70b-8192', temperature=0.1, groq_api_key=SecretStr('**********'), max_tokens=1024) kwargs={'tools': [{'type': 'function', 'function': {'name': 'search', 'description': 'Call to surf the web.', 'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}}, 'required': ['query']}}}]}
```
#### defining a new graph using state to track messages
```python
workflow = StateGraph(MessagesState)
# outputs
workflow:  <langgraph.graph.state.StateGraph object at 0x7a1816ebd150>
```
#### adding different node to the graph
```python
workflow.add_node("agent", call_model)
# outputs
Workflow add node 'agent':  <langgraph.graph.state.StateGraph object at 0x7a1816ebd150>

workflow.add_node("tools", tool_node)
# outputs
Workflow add node 'tools':  <langgraph.graph.state.StateGraph object at 0x7a1816ebd150>
```
#### here we set where the graph will start `set_entrypoint`
```python
workflow.set_entry_point("agent")
# outputs
Workflow set entry point 'agent':  <langgraph.graph.state.StateGraph object at 0x7a1816ebd150>
```
#### here we define the lines of the workflow with `add_conditional_edges`, the function being the one determining the next step in function of its output
```python
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)
# outputs
Workflow add conditional edge 'agent' -> should_continue func:  <langgraph.graph.state.StateGraph object at 0x7a1816ebd150>
```
#### here we define the lines of the workflow with a 'normal edge' this time from `tools` to `agent`
```python
workflow.add_edge("tools", 'agent')
# outputs
Workflow add edge 'tools' -> 'agent':  <langgraph.graph.state.StateGraph object at 0x7a1816ebd150>
```
#### here we set up a checkpointer that persistent state between runs like a memory `from langgraph.checkpoint import MemorySaver`. (we might study this function more thouroughly to see if we can implement custom functions for other purposes using this method of checkpointing)
```python
checkpointer = MemorySaver()
# outputs
Memory checkpointer:  <langgraph.checkpoint.memory.MemorySaver object at 0x7a18160323b0>
```
#### here we compile the graph to become a LangChain runnable therefore we will be able to use .invoke(), .stream(), .batch(), and the memory checkpointer here is optional
```python
app = workflow.compile(checkpointer=checkpointer)
# outputs
App compiled with checkpointer:  nodes={'__start__': PregelNode(config={'tags': ['langsmith:hidden'], 'metadata': {}, 'configurable': {}}, channels=['__start__'], triggers=['__start__'], writers=[ChannelWrite<messages>(recurse=True, writes=[ChannelWriteEntry(channel='messages', value=<object object at 0x7a181858b370>, skip_none=False, mapper=_get_state_key(recurse=False))], require_at_least_one_of=['messages']), ChannelWrite<start:agent>(recurse=True, writes=[ChannelWriteEntry(channel='start:agent', value='__start__', skip_none=False, mapper=None)], require_at_least_one_of=None)]), 'agent': PregelNode(config={'tags': [], 'metadata': {}, 'configurable': {}}, channels={'messages': 'messages'}, triggers=['tools', 'start:agent'], mapper=functools.partial(<function _coerce_state at 0x7a181657a830>, <class 'langgraph.graph.message.MessagesState'>), writers=[ChannelWrite<agent,messages>(recurse=True, writes=[ChannelWriteEntry(channel='agent', value='agent', skip_none=False, mapper=None), ChannelWriteEntry(channel='messages', value=<object object at 0x7a181858b370>, skip_none=False, mapper=_get_state_key(recurse=False))], require_at_least_one_of=['messages']), _route(recurse=True, _is_channel_writer=True)]), 'tools': PregelNode(config={'tags': [], 'metadata': {}, 'configurable': {}}, channels={'messages': 'messages'}, triggers=['branch:agent:should_continue:tools'], mapper=functools.partial(<function _coerce_state at 0x7a181657a830>, <class 'langgraph.graph.message.MessagesState'>), writers=[ChannelWrite<tools,messages>(recurse=True, writes=[ChannelWriteEntry(channel='tools', value='tools', skip_none=False, mapper=None), ChannelWriteEntry(channel='messages', value=<object object at 0x7a181858b370>, skip_none=False, mapper=_get_state_key(recurse=False))], require_at_least_one_of=['messages'])])} channels={'messages': <langgraph.channels.binop.BinaryOperatorAggregate object at 0x7a1816032410>, '__start__': <langgraph.channels.ephemeral_value.EphemeralValue object at 0x7a1816031510>, 'agent': <langgraph.channels.ephemeral_value.EphemeralValue object at 0x7a18160312a0>, 'tools': <langgraph.channels.ephemeral_value.EphemeralValue object at 0x7a1816031120>, 'start:agent': <langgraph.channels.ephemeral_value.EphemeralValue object at 0x7a18160313c0>, 'branch:agent:should_continue:tools': <langgraph.channels.ephemeral_value.EphemeralValue object at 0x7a18160311e0>} auto_validate=False stream_mode='updates' output_channels=['messages'] stream_channels=['messages'] input_channels='__start__' checkpointer=<langgraph.checkpoint.memory.MemorySaver object at 0x7a18160323b0> builder=<langgraph.graph.state.StateGraph object at 0x7a1816ebd150>
```
#### here we use the Langchain Graph runnable created and use .invoke(), the `{"thread_id": 42}` can be used again to use the memory in order to get same kind of response but adapted to the new user query. so if same ID is used we will get same kind of response formatting.
```python
final_state = app.invoke(
    {"messages": [HumanMessage(content="what is the weather in sf")]},
    config={"configurable": {"thread_id": 42}}
)
# outputs
messages from call_model func:  [HumanMessage(content='what is the weather in sf', id='75ade55f-d939-46d5-926e-fa293af0cf14')]
```
#### here we see the function called logs. Here for 
```python
def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    print("messages from should_continue func: ", messages)
    last_message = messages[-1]
    print("last message from should_continue func: ", last_message)
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        print("Tool called!")
        return "tools"
    # Otherwise, we stop (reply to the user)
    print("Tool not called returning answer to user.")
    return END


# Define the function that calls the model
def call_model(state: MessagesState):
    messages = state['messages']
    print("messages from call_model func: ", messages)
    response = model.invoke(messages)
    print("response from should_continue func: ", response)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


## flow of outputs when 'final_state' invokes the Graph runnable 'app'
## agent > conditional edge > tool call > tools edge answer to agent edge > answer to user 
##                          > tool not called > answer to user
# Here teh conditional edge decides if tool needs to be called and decides to call the tool to get an answer
response from should_continue func:  content='' additional_kwargs={'tool_calls': [{'id': 'call_4gr6', 'function': {'arguments': '{"query":"weather in san francisco"}', 'name': 'search'}, 'type': 'function'}]} response_metadata={'token_usage': {'completion_tokens': 46, 'prompt_tokens': 903, 'total_tokens': 949, 'completion_time': 0.14465532, 'prompt_time': 0.065623244, 'queue_time': None, 'total_time': 0.210278564}, 'model_name': 'llama3-70b-8192', 'system_fingerprint': 'fp_c1a4bcec29', 'finish_reason': 'tool_calls', 'logprobs': None} id='run-09d0c91f-a6c9-476d-bc6e-cb332d4757b2-0' tool_calls=[{'name': 'search', 'args': {'query': 'weather in san francisco'}, 'id': 'call_4gr6', 'type': 'tool_call'}] usage_metadata={'input_tokens': 903, 'output_tokens': 46, 'total_tokens': 949}
messages from should_continue func:  [HumanMessage(content='what is the weather in sf', id='75ade55f-d939-46d5-926e-fa293af0cf14'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_4gr6', 'function': {'arguments': '{"query":"weather in san francisco"}', 'name': 'search'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 46, 'prompt_tokens': 903, 'total_tokens': 949, 'completion_time': 0.14465532, 'prompt_time': 0.065623244, 'queue_time': None, 'total_time': 0.210278564}, 'model_name': 'llama3-70b-8192', 'system_fingerprint': 'fp_c1a4bcec29', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-09d0c91f-a6c9-476d-bc6e-cb332d4757b2-0', tool_calls=[{'name': 'search', 'args': {'query': 'weather in san francisco'}, 'id': 'call_4gr6', 'type': 'tool_call'}], usage_metadata={'input_tokens': 903, 'output_tokens': 46, 'total_tokens': 949})]
last message from should_continue func:  content='' additional_kwargs={'tool_calls': [{'id': 'call_4gr6', 'function': {'arguments': '{"query":"weather in san francisco"}', 'name': 'search'}, 'type': 'function'}]} response_metadata={'token_usage': {'completion_tokens': 46, 'prompt_tokens': 903, 'total_tokens': 949, 'completion_time': 0.14465532, 'prompt_time': 0.065623244, 'queue_time': None, 'total_time': 0.210278564}, 'model_name': 'llama3-70b-8192', 'system_fingerprint': 'fp_c1a4bcec29', 'finish_reason': 'tool_calls', 'logprobs': None} id='run-09d0c91f-a6c9-476d-bc6e-cb332d4757b2-0' tool_calls=[{'name': 'search', 'args': {'query': 'weather in san francisco'}, 'id': 'call_4gr6', 'type': 'tool_call'}] usage_metadata={'input_tokens': 903, 'output_tokens': 46, 'total_tokens': 949}
Tool called!
# here the model is called to get an answer
messages from call_model func:  [HumanMessage(content='what is the weather in sf', id='75ade55f-d939-46d5-926e-fa293af0cf14'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_4gr6', 'function': {'arguments': '{"query":"weather in san francisco"}', 'name': 'search'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 46, 'prompt_tokens': 903, 'total_tokens': 949, 'completion_time': 0.14465532, 'prompt_time': 0.065623244, 'queue_time': None, 'total_time': 0.210278564}, 'model_name': 'llama3-70b-8192', 'system_fingerprint': 'fp_c1a4bcec29', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-09d0c91f-a6c9-476d-bc6e-cb332d4757b2-0', tool_calls=[{'name': 'search', 'args': {'query': 'weather in san francisco'}, 'id': 'call_4gr6', 'type': 'tool_call'}], usage_metadata={'input_tokens': 903, 'output_tokens': 46, 'total_tokens': 949}), ToolMessage(content="It's 60 degrees and foggy.", name='search', id='f60bbd34-9c08-48ad-b6bb-74c96b741a31', tool_call_id='call_4gr6')]
# here tool is not called and returning answer to user. This because we called it twice with same user query and the second time here it uses the memory to form its answer.
response from should_continue func:  content='The weather in San Francisco is 60 degrees and foggy.' response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 979, 'total_tokens': 993, 'completion_time': 0.042933596, 'prompt_time': 0.074989348, 'queue_time': None, 'total_time': 0.117922944}, 'model_name': 'llama3-70b-8192', 'system_fingerprint': 'fp_c1a4bcec29', 'finish_reason': 'stop', 'logprobs': None} id='run-576d3c0f-6915-4fd3-858d-2532c76ed5c0-0' usage_metadata={'input_tokens': 979, 'output_tokens': 14, 'total_tokens': 993}
messages from should_continue func:  [HumanMessage(content='what is the weather in sf', id='75ade55f-d939-46d5-926e-fa293af0cf14'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_4gr6', 'function': {'arguments': '{"query":"weather in san francisco"}', 'name': 'search'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 46, 'prompt_tokens': 903, 'total_tokens': 949, 'completion_time': 0.14465532, 'prompt_time': 0.065623244, 'queue_time': None, 'total_time': 0.210278564}, 'model_name': 'llama3-70b-8192', 'system_fingerprint': 'fp_c1a4bcec29', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-09d0c91f-a6c9-476d-bc6e-cb332d4757b2-0', tool_calls=[{'name': 'search', 'args': {'query': 'weather in san francisco'}, 'id': 'call_4gr6', 'type': 'tool_call'}], usage_metadata={'input_tokens': 903, 'output_tokens': 46, 'total_tokens': 949}), ToolMessage(content="It's 60 degrees and foggy.", name='search', id='f60bbd34-9c08-48ad-b6bb-74c96b741a31', tool_call_id='call_4gr6'), AIMessage(content='The weather in San Francisco is 60 degrees and foggy.', response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 979, 'total_tokens': 993, 'completion_time': 0.042933596, 'prompt_time': 0.074989348, 'queue_time': None, 'total_time': 0.117922944}, 'model_name': 'llama3-70b-8192', 'system_fingerprint': 'fp_c1a4bcec29', 'finish_reason': 'stop', 'logprobs': None}, id='run-576d3c0f-6915-4fd3-858d-2532c76ed5c0-0', usage_metadata={'input_tokens': 979, 'output_tokens': 14, 'total_tokens': 993})]
last message from should_continue func:  content='The weather in San Francisco is 60 degrees and foggy.' response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 979, 'total_tokens': 993, 'completion_time': 0.042933596, 'prompt_time': 0.074989348, 'queue_time': None, 'total_time': 0.117922944}, 'model_name': 'llama3-70b-8192', 'system_fingerprint': 'fp_c1a4bcec29', 'finish_reason': 'stop', 'logprobs': None} id='run-576d3c0f-6915-4fd3-858d-2532c76ed5c0-0' usage_metadata={'input_tokens': 979, 'output_tokens': 14, 'total_tokens': 993}
Tool not called returning answer to user.

# if we want the full object with all answers
# first call for answer with tool used and stored to memory
final_state
# outputs 
Final State = answer:  {'messages': [HumanMessage(content='what is the weather in sf', id='75ade55f-d939-46d5-926e-fa293af0cf14'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_4gr6', 'function': {'arguments': '{"query":"weather in san francisco"}', 'name': 'search'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 46, 'prompt_tokens': 903, 'total_tokens': 949, 'completion_time': 0.14465532, 'prompt_time': 0.065623244, 'queue_time': None, 'total_time': 0.210278564}, 'model_name': 'llama3-70b-8192', 'system_fingerprint': 'fp_c1a4bcec29', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-09d0c91f-a6c9-476d-bc6e-cb332d4757b2-0', tool_calls=[{'name': 'search', 'args': {'query': 'weather in san francisco'}, 'id': 'call_4gr6', 'type': 'tool_call'}], usage_metadata={'input_tokens': 903, 'output_tokens': 46, 'total_tokens': 949}), ToolMessage(content="It's 60 degrees and foggy.", name='search', id='f60bbd34-9c08-48ad-b6bb-74c96b741a31', tool_call_id='call_4gr6'), AIMessage(content='The weather in San Francisco is 60 degrees and foggy.', response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 979, 'total_tokens': 993, 'completion_time': 0.042933596, 'prompt_time': 0.074989348, 'queue_time': None, 'total_time': 0.117922944}, 'model_name': 'llama3-70b-8192', 'system_fingerprint': 'fp_c1a4bcec29', 'finish_reason': 'stop', 'logprobs': None}, id='run-576d3c0f-6915-4fd3-858d-2532c76ed5c0-0', usage_metadata={'input_tokens': 979, 'output_tokens': 14, 'total_tokens': 993})]}

# if we want the last message only from the stte which is the answer:
# second call for answer no need tool call, using memory to form it's answer to same query
final_state["messages"][-1].content
# outputs
Final state last message content:  The weather in San Francisco is 60 degrees and foggy.
```

## LangGraph Object needed or interesting information to get from objects

##### LLM Response last message content (add [-1] to target last message if needed only that one)
reponse["messages"][-1].content

##### AI LLM Response token usage data
response["response_metadata"]["token_usage"]
response["response_metadata"]["completion_tokens"]
response["response_metadata"]["prompt_tokens"]
response["response_metadata"]["total_tokens"]
response["response_metadata"]["completion_time"]
response["response_metadata"]["prompt_time"]
response["response_metadata"]["total_time"]
response["model_name"]
response["usage_metadata"]["input_tokens"]
response["usage_metadata"]["output_tokens"]
response["usage_metadata"]["total_tokens"]

if tool called we bet this additional part inside of it:
response["aDDitional-kwargs"]["tool_calls"][0]["function"]["arguments"]["query"] # argument passed in the function
response["aDDitional-kwargs"]["tool_calls"][0]["function"]["name"] # name of function
response["aDDitional-kwargs"]["tool_calls"][0]["type"] # type function for example
                                         additional_kwargs={
                                           'tool_calls': [
                                             {
                                               'id': 'call_p5pe',
                                               'function': {
                                                 'arguments': '{
                                                   "query":"current weather in san francisco"
                                                 }',
                                                 'name': 'search'
                                               },
                                               'type': 'function'
                                             }
                                           ]
                                         }

##### Tool Message Object (its content and function name)
response.content
response.name 

##### Human Message object (its content only)
response.content

##### function call LLM list structure
[Human message, AI message, Tool message]


# LANGCHAIN & LANGGRAPH PRECISIONS ON WORKFLOWS

## PROMPTING
- PROMPT = prompt_template, input_vars=[...,...,...]
- type_of_action = PROMPT  |  LLM  |  PARSER(if any)

- eg.:
input_var1 = user_query
input_var2 = func(....) output

response = type_of_action.invoke(
  {
    "input_var1": input_var1 ,
    "input_var2": input_var2
  }
)


## GRAPH LOGIC
- All possible nodes:
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae
workflow.add_node("web_search", web_search)  # web search

- All nodes used to make workflow, start route flow:
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")

- Like a 'IF/ELSE' which will route depending on output of function 
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "search": "web_search", # if output == "search" use web_search node
        "generate": "generate",  # if output == "generate" use generate node
    },
)

- After 'IF/ELSE' possible routes and the next logic
workflow.add_edge("web_search", "generate") # OR web_search, then Generate and then END
workflow.add_edge("generate", END) # OR generate, then END

## Biriculouskous Diagram of the worlfow
--- START --- Retrieve --- Grade_documents (uses decide_to_generate func output to decide where to go after) --- output = search --- Web_search --- Generate --- END
                                                                                                             --- output = generate --- Generate --- END
#### Diagram
- First version: End[Diagram Link:](https://excalidraw.com/#json=Id5LJIJ2GnPsu3Fk6E1Nr,39wVy9-u6aX7PJY-YRPirw)
- Second version: [Diagram Link:](https://excalidraw.com/#json=W5qMMT6topHNXtdcVMrcp,B8MDvIeeI5XxW-8JjFZeNg)

[---------------------------------------------------------]

# LangGraph States
- Use of Pydantic to have multi-states enabled in LangGraph
### define states
from pydantic import BaseModel
from typing import List

class MessageState(BaseModel):
    messages: List[str]

class UserState(BaseModel):
    user_id: int
    preferences: dict

### initialize
message_state = MessageState(messages=[])
user_state = UserState(user_id=1, preferences={})

### Node access to states
def update_message_state(state: MessageState, new_message: str):
    state.messages.append(new_message)
    return state

def update_user_state(state: UserState, new_preferences: dict):
    state.preferences.update(new_preferences)
    return state

### Example of use
from langgraph import StateGraph

graph = StateGraph()

graph.add_node("update_message_state", update_message_state)
graph.add_node("update_user_state", update_user_state)

graph.add_edge("start", "update_message_state")
graph.add_edge("update_message_state", "update_user_state")

- OR Use Composite state class
class CompositeState(BaseModel):
    message_state: MessageState
    user_state: UserState

composite_state = CompositeState(
    message_state=MessageState(messages=[]),
    user_state=UserState(user_id=1, preferences={})
)

### Nodes
def update_message_in_composite_state(state: CompositeState, new_message: str):
    state.message_state.messages.append(new_message)
    return state

def update_user_in_composite_state(state: CompositeState, new_preferences: dict):
    state.user_state.preferences.update(new_preferences)
    return state

### Graph
graph = StateGraph()

graph.add_node("update_message", update_message_in_composite_state)
graph.add_node("update_user", update_user_in_composite_state)

graph.add_edge("start", "update_message")
graph.add_edge("update_message", "update_user")



# LangGraph States & Langfuse Tracing

See also if we can use langfuse prompts in order to have prompts injected in the code:
eg.:
### Build and create the prompt
langfuse.create_prompt(
    name="movie-critic",
    type="chat", # can be "text" for chat prompt
    prompt="Do you like {{movie}}?",
    labels=["production"],  # directly promote to production
    config={
        "model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "supported_languages": ["en", "fr"],
    },  # optionally, add configs (e.g. model parameters or model tools) or tags
)

from langfuse import Langfuse
###### Initialize Langfuse client
langfuse = Langfuse()
###### Get current production version of a text prompt
prompt = langfuse.get_prompt("movie-critic")
###### Insert variables into prompt template
compiled_prompt = prompt.compile(movie="Dune 2")
###### "Do you like Dune 2?"

###### Get current production version of a chat prompt
chat_prompt = langfuse.get_prompt("movie-critic-chat", type="chat") # type arg infers the prompt type (default is 'text') 
###### Insert variables into chat prompt template
compiled_chat_prompt = chat_prompt.compile(movie="Dune 2")
###### [
     {
       "role": "system",
       "content": "You are an expert on Dune 2"
     }
   ]
- some commands for the prompts
###### Get specific version
prompt = langfuse.get_prompt("movie-critic", version=1)
###### Get specific label
prompt = langfuse.get_prompt("movie-critic", label="staging")
###### Get latest prompt version. The 'latest' label is automatically maintained by Langfuse.
prompt = langfuse.get_prompt("movie-critic", label="latest")
###### Extend cache TTL from default 60 to 300 seconds
prompt = langfuse.get_prompt("movie-critic", cache_ttl_seconds=300)

- access RAW prompt and configs
###### Raw prompt including {{variables}}. For chat prompts, this is a list of chat messages.
prompt.prompt
###### Config object
prompt.config

- Langfuse tracing in function:
@observe(as_type="generation")
def nested_generation():
    prompt = langfuse.get_prompt("movie-critic")
 
    langfuse_context.update_current_observation(
        prompt=prompt,
    )
    
#### The need of reducer function change the behavior of LangGraph States
LangGraph state will update at every node getting rid of the previous value, therefore, if you want to update but keep track of changes you need a reducer function for that.
That is when `Annotated` from `tying` is coming to permit to have a reducer added to your state, then you would just use that function to for example add values to your state.
```python
from typing import TypedDict, Annotated
from operator import add
class State(TypedDict):
    foo: int
    bar: Annotated[list[str], add]

# OR here
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict
class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

#### If you don't want to use reducer, you can use prebuilt method that adds up messages to your state like ``
from langgraph.graph import MessagesState
class State(MessagesState):
    documents: list[str]
[------------------------------------------------------------------------------------------------------]


# LangGraph Edges
A node can have MULTIPLE outgoing edges. If a node has multiple out-going edges, all of those destination nodes will be executed in parallel as a part of the next superstep.
node --- conditional edge ---  run next --------------------------------------
                          ---  condition not met don't run next x
                          ---  run next --------------------------------------   ALL --- Will Run in Parallele (Interesting!) so be carefull with state updates to not mess up
                          ---  run next --------------------------------------
                          ---  condition not met don't run next x
                          ---  run next --------------------------------------

Because if this, it can be confusing with the state passed to different nodes and that is when we might really need to manage different states for different routes.
LangGraph documentation suggests to use the mapReduce design and permit us to have a specific state sending its messages to that state for example. The method used is `send` which allow us to pass the state to the next node (a specific state as here we have running parallele actions as the conditional edge condition is met for several next node to be activated):
eg:.
```python
# we just need to pass that state to the function
def continue_to_jokes(state: OverallState):
    return [Send("generate_joke", {"subject": s}) for s in state['subjects']]
# conditional edge
graph.add_conditional_edges("node_a", continue_to_jokes)
```

# LangGraph Checkpointers
Checkpointers are used for human interraction in the loop or if there a chat to save the messges as a `memory` so that the context is preserved.
Checkpointer are also a good way to see the state at a certain important moment if needed.
Checkpointer need thread to be passed in to keep an id that would be used to manage for example concurrent use by different user of the app. Which will maiantain different states for different users.
```python
# here we show as well that a specific thread can be called with it's ID when invoking LangChain runnables
config = {"configurable": {"thread_id": "a"}}
graph.invoke(inputs, config=config)
```
##### Checkpointer values to be passed in
values: This is the value of the state at this point in time.
next: This is a tuple of the nodes to execute next in the graph.

#### Example of a checkpointer coming from the library
```python
from langgraph.checkpoint import MemorySaver
checkpointer = MemorySaver()
# Note that we're (optionally) passing the memory when compiling the graph but better for sophisticated workflows like ones including breakpoints
app = workflow.compile(checkpointer=checkpointer)
```

#### Now you can get state at a checkpoint by calling `get_state`
``` python
graph.get_state(config)
```
config: the config should contain thread_id specifying which thread to update when needed to update it.

#### use `update_state` to update the state
eg. with reducer to understand it all:
```python
from typing import TypedDict, Annotated
from operator import add
class State(TypedDict):
    foo: int
    bar: Annotated[list[str], add]

# Let's now assume the current state of the graph is
{"foo": 1, "bar": ["a"]}

# If you update the state as below:
graph.update_state(config, {"foo": 2, "bar": ["b"]})

# Then the new state of the graph will be:
{"foo": 2, "bar": ["a", "b"]}
```

The reducer keeps the previous states therefore we can track the changes while the normal state is overrided with a new value. That is why we use reducers.
After that two different behaviors:
- you call `as_node` to precise which node is considered to be one changing the state 
- OR you don't call `as_node` and then the default behavior is to consider that changes are coming from the previous node

# LangGraph Configurations

You can set the configuration at the beginning in order to be able to access to those different configs at any node point. This allow customization and use of different options like different models for example and see what is the outcome for different models and compare those like that.
```python
class ConfigSchema(TypedDict):
    llm: str

graph = StateGraph(State, config_schema=ConfigSchema)

# You can then pass this configuration into the graph using the configurable config field.
config = {"configurable": {"llm": "anthropic"}}
graph.invoke(inputs, config=config)

# You can then access and use this configuration inside a node:
def node_a(state, config):
    # here we access the config to get the llm config we want
    llm_type = config.get("configurable", {}).get("llm", "openai")
    # here we use that llm which has been saved to a variable
    llm = get_llm(llm_type)
    ...
```

# LangGraph Breakpoints

Set breakpoints either before a node executes (using interrupt_before) or after a node executes (using interrupt_after.)
This can be useful if needed to have a human interaction which would stop the graph and resume it after.
Note:
 - checkpointer (pass it at the compilation moment like memory) need to be used here in order for the graph to know the state where it should restart from
 - use `None` to restart where you stopped at
```python
# Initial run of graph
graph.invoke(inputs, config=config)

# Resume by passing in None
graph.invoke(None, config=config)
```

This is really interesting as we could stop the graph to do some other things in the application code or use other systems and have some variables updated that could later on be used by next nodes in order to have sophisticated workflow which can stop and restart while waiting for other operations to be done.

##### Example
```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display

class State(TypedDict):
    input: str

def step_1(state):
    print("---Step 1---")
    pass

def step_2(state):
    print("---Step 2---")
    pass

def step_3(state):
    print("---Step 3---")
    pass

builder = StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)
builder.add_node("step_3", step_3)
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", END)

# Set up memory
memory = MemorySaver()

# Add and put the `interrupt_before` specifiying at which step
graph = builder.compile(checkpointer=memory, interrupt_before=["step_3"])

# View
display(Image(graph.get_graph().draw_mermaid_png()))
```

```python
# now we need a thread ID for the checkpointer before interruption at step 3
# Input
initial_input = {"input": "hello world"}
# Thread
thread = {"configurable": {"thread_id": "1"}}
# Run the graph until the first interruption
for event in graph.stream(initial_input, thread, stream_mode="values"):
    print(event)
# Here use is asked if we continue after the interruption at step 3
user_approval = input("Do you want to go to Step 3? (yes/no): ")
if user_approval.lower() == "yes":
    # If approved, continue the graph execution as here we use the `None` keyword as first argument, the thread ID is provided so the checkpint know where to restart from
    for event in graph.stream(None, thread, stream_mode="values"):
        print(event)else:
    print("Operation cancelled by user.")

Outputs:
{'input': 'hello world'}
---Step 1---
---Step 2---
---Step 3---
Do you want to go to Step 3? (yes/no): 
```
##### precision about `stream_mode` different values that it takes
- `"values"` which output full state
- `"updates"` which ouput only updates
- `"debug"` which output as much as possible to debug

#### `astream_events`
This is to stream event like llm calls and token tracking while those are being called inside nodes.
In the documentation examples this is used for async functions


# LangGraph ReAct Agent

Providing multiple tools to an llm and letting the llm use those tools as much as it wants and decide on the input of those tools until it decides that it doesn't need to use any tool anymore and exists the while loop. Define in the docs as: `In this architecture, an LLM is called repeatedly in a while-loop. At each step the agent decides which tools to call, and what the inputs to those tools should be. Those tools are then executed, and the outputs are fed back into the LLM as observations. The while-loop terminates when the agent decides it is not worth calling any more tools.`

#### Example of ReAct Agent
```python
from langgraph.prebuilt import create_react_agent
```
check documentation as example is too long: [Example of ReAct Agent Docs](https://langchain-ai.github.io/langgraph/reference/prebuilt/#create_react_agent)
Short example:
- create a ReAct Type of Agent Tool
```python
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.graph import MessageGraph
from langgraph.prebuilt import ToolNode, tools_condition

@tool
def divide(a: float, b: float) -> int:
    """Return a / b."""
    return a / b

llm = ChatAnthropic(model="claude-3-haiku-20240307")
tools = [divide]
graph_builder = MessageGraph()
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_node("chatbot", llm.bind_tools(tools))
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_conditional_edges(
...     "chatbot", tools_condition
... )
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile()
graph.invoke([("user", "What's 329993 divided by 13662?")])
```
- Then create a validation node
```python
from typing import Literal
from langchain_anthropic import ChatAnthropic
from langchain_core.pydantic_v1 import BaseModel, validator
from langgraph.graph import END, START, MessageGraph
from langgraph.prebuilt import ValidationNode

class SelectNumber(BaseModel):
    a: int

    @validator("a")
    def a_must_be_meaningful(cls, v):
        if v != 37:
            raise ValueError("Only 37 is allowed")
        return v

builder = MessageGraph()
llm = ChatAnthropic(model="claude-3-haiku-20240307").bind_tools([SelectNumber])
builder.add_node("model", llm)
builder.add_node("validation", ValidationNode([SelectNumber]))
builder.add_edge(START, "model")

def should_validate(state: list) -> Literal["validation", "__end__"]:
    if state[-1].tool_calls:
        return "validation"
    return END

builder.add_conditional_edges("model", should_validate)

def should_reprompt(state: list) -> Literal["model", "__end__"]:
    for msg in state[::-1]:
        # None of the tool calls were errors
        if msg.type == "ai":
            return END
        if msg.additional_kwargs.get("is_error"):
            return "model"
    return END

builder.add_conditional_edges("validation", should_reprompt)
graph = builder.compile()
res = graph.invoke(("user", "Select a number, any number"))
# Show the retry logic
for msg in res:
   msg.pretty_print()

Outputs:
================================ Human Message =================================
Select a number, any number
================================== Ai Message ==================================
[{'id': 'toolu_01JSjT9Pq8hGmTgmMPc6KnvM', 'input': {'a': 42}, 'name': 'SelectNumber', 'type': 'tool_use'}]
Tool Calls:
SelectNumber (toolu_01JSjT9Pq8hGmTgmMPc6KnvM)
Call ID: toolu_01JSjT9Pq8hGmTgmMPc6KnvM
Args:
    a: 42
================================= Tool Message =================================
Name: SelectNumber
ValidationError(model='SelectNumber', errors=[{'loc': ('a',), 'msg': 'Only 37 is allowed', 'type': 'value_error'}])
Respond after fixing all validation errors.
================================== Ai Message ==================================
[{'id': 'toolu_01PkxSVxNxc5wqwCPW1FiSmV', 'input': {'a': 37}, 'name': 'SelectNumber', 'type': 'tool_use'}]
Tool Calls:
SelectNumber (toolu_01PkxSVxNxc5wqwCPW1FiSmV)
Call ID: toolu_01PkxSVxNxc5wqwCPW1FiSmV
Args:
    a: 37
================================= Tool Message =================================
Name: SelectNumber
{"a": 37}
```

#### workaround found by a user if with_structured_output() can't be used
As here we can't use `with_structured_output()` to create our tools we can use `parser_chain` using the `PydanticOutputParser` to get the structured output:
```python
graph = create_react_agent(model=llm,tools=self.tools,messages_modifier=prompt) | RunnableLambda(lambda data: {"text": str(data["messages"][-1].content)}) | parser_chain
```

# What if we want the LLM to run the script that it has created and not arm the server running the app?

##### Running in a Separate Process with Limited Permissions
Subprocess Module: Running the code in a subprocess with restricted permissions can isolate the execution from the main process.

```python
import subprocess

try:
    result = subprocess.run(['python3', 'script.py'], capture_output=True, text=True, timeout=10)
    print(result.stdout)
except subprocess.TimeoutExpired:
    print("The script took too long to complete.")
```

##### Using Docker for Sandbox (my preferred one)
```bash
# nano Dockerfile
# Dockerfile
FROM python:3.9-slim
# Copy the requirements.txt file into the container
COPY requirements.txt .
# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt
# script that we want to run inside docker
COPY app.py /app/app.py
# copy .dynamic.env file special for agent if needed (optional)
# COPY .dynamic.env /app/.dynamic.env
WORKDIR /app
CMD ["python3", "app.py"]
# Build and run (this step will be replaced by a script capturing output)
docker build -t sandbox-python .
docker run --rm sandbox-python



# BUT, we will use the following script so that the Agent can capture the output of the script ran in docker using `subprocess`
import subprocess

def run_script_in_docker(dockerfile_name_to_run_script: str, agent_script_file_name: str) -> Tuple[str, str]:

    # Write the script content to a file
    with open("agent_created_script_to_execute_in_docker.py", "w", encoding="utf-8") as going_to_docker_script_file, open(agent_script_file_name, "r", encoding="utf-8") as script_content:
        going_to_docker_script_file.write(script_content.read())

    # Create the dockerfile for the agent to run the script inside docker
    with open(dockerfile_name_to_run_script, "w", encoding="utf-8") as docker_file:
        docker_file.write("FROM python:3.9-slim\n")
        docker_file.write("COPY requirements.txt .\n")
        docker_file.write("RUN pip install --no-cache-dir -r requirements.txt\n")
        docker_file.write("COPY agent_created_script_to_execute_in_docker.py /app/agent_created_script_to_execute_in_docker.py\n")
        docker_file.write("COPY .dynamic.env /app/.dynamic.env\n")
        docker_file.write("WORKDIR /app\n")
        docker_file.write('CMD ["python", "agent_created_script_to_execute_in_docker.py"]')

    try:
        # Build the Docker image
        build_command = ['docker', 'build', '-t', 'sandbox-python', '-f', f'{dockerfile_name_to_run_script}', '.']
        subprocess.run(build_command, check=True)

        # Run the Docker container and capture the output
        run_command = ['docker', 'run', '--rm', 'sandbox-python']
        result = subprocess.run(run_command, capture_output=True, text=True)

        stdout, stderr = result.stdout, result.stderr
        with open("agent_docker_script_execution_result.md", "w", encoding="utf-8") as script_execution_result:
          script_execution_result.write("# Python script executed in docker, this is the result of captured stdout and stderr")
          script_execution_result.write("""
            This is the result after the execution of the code
            Returns:
            stdout str: the standard output of the script execution which runs in docker, therefore, we capture the stdout to know if the script output is as expected. You need to analyze it and see why it is empty if emppty, why it is not as expected to suggest code fix, and if the script executes correctly, get this stdout value and answer using markdown ```python ``` saying just one word: OK'
            stderr str: the standard error of the script execution. If this value is not empty answer witht the content of the value with a suggestion in how to fix it. Answer using mardown and put a JSON of the error message with key ERROR between this ```python ```. 
          """)
          script_execution_result.write(f"\n\nstdout: {stdout}\nstderr: {stderr}")

    except subprocess.CalledProcessError as e:
        stdout, stderr = '', str(e)
    finally:
        # Remove the Docker image
        cleanup_command = ['docker', 'rmi', '-f', 'sandbox-python']
        subprocess.run(cleanup_command, check=False)

    # Return the captured output
    return stdout, stderr

# Example script content created by the LLM
script_content = """
print('Hello, World!')
"""

stdout, stderr = run_script_in_docker(script_content)

print("Output:")
print(stdout)
print("Errors:")
print(stderr)


# PGVector and Postgreql linking

#### How to link PGVector collection to the corresponding data from Postrgesql database table.
##### Solution is to use a UUID which will be also embedded:
- The Posgreql database have a column with unique identifier UUID
- The column containing the cleaned data that we want to embed will have a UUID
- Embed the data with the UUID
- Retrieve on user query
- Then parse the UUID to fetch the database row and get the column that we want


# Matching Query Semantic vs Exact
When a user sends its query we need a way to check if we can just use the cache to answer or if we need to perform a vector search.
For that if we decide to check on the cache before making any vector search, we need to find a similar query:
- Exact match: we find the smae query and fetch the content stored
- Semantic mecth: we use embedding of the query to search for queries int he cache with are similar (I prefer this one)
 
But we will implement a function that does both, if it founds same query, it fetches from cache but will also check if there is semantic similiraty relevance. If not,the last chance will be to go to the vector db search.

When all fails, even the vector db search, LLM can use other tools to search online.

# beware of `langchain_community.vectorstores.pgvector` metadata storage as JSONB instead of JSON
```bash
# warning message
/home/creditizens/langgraph/langgraph_venv/lib/python3.10/site-packages/langchain_community/vectorstores/pgvector.py:328: LangChainPendingDeprecationWarning: Please use JSONB instead of JSON for metadata. This change will allow for more efficient querying that involves filtering based on metadata.Please note that filtering operators have been changed when using JSOB metadata to be prefixed with a $ sign to avoid name collisions with columns. If you're using an existing database, you will need to create adb migration for your metadata column to be JSONB and update your queries to use the new operators. 
  warn_deprecated(
```

# Store and query embeddings metadata: `full example`
Interesting when wanting to filter the content and narrow down the search to specific parts of the documents and to organize those like that.
It could be a table column having those metadata and the collection having the embedded target document with same metadata. So two different agents or nodes could be used, one that search the metada using user query and the other taking that result to perform a vector search. The last would send it to another node for analysis and answer formating or next step actions.

To store vectors as JSONB in a PostgreSQL table, especially when dealing with high-dimensional vectors like 4096 dimensions, you would create a table with a JSONB column and an additional column to store the vectors. Here's how you can do this using the xample of a table storage for everything to make it easily understandable:


```sql
# table creation with JSONB fields and 4096(ollama mistral:7b dimensions) vector dimension field
CREATE TABLE documents (
    id UUID PRIMARY KEY,
    doc_name TEXT,
    title TEXT,
    content JSONB,
    embedding VECTOR(4096),
    metadata JSONB
);

# insert data using `::JSONB`
INSERT INTO documents (id, doc_name, title, content, embedding, metadata)
VALUES (
    gen_random_uuid(),
    'example_doc.pdf',
    'An Example Document',
    '{"paragraph1": "This is an example paragraph.", "paragraph2": "This is another example."}',
    '[0.1, 0.2, 0.3, ..., 0.4096]'::VECTOR(4096),
    '{"author": "John Doe", "tags": ["example", "demo"]}'::JSONB
);

# query metada using `@>`
SELECT * 
FROM documents 
WHERE metadata @> '{"author": "John Doe"}';

# perform vector similarity using `<=>` and `::VECTOR(4096)`
SELECT id, title, embedding
FROM documents
ORDER BY embedding <=> '[0.1, 0.2, 0.3, ..., 0.4096]'::VECTOR(4096)
LIMIT 10;
```
- The <=> operator is used for vector similarity search.
- The JSONB @> operator is used to check if the JSONB column contains the specified key-value pair.

**For the moment**
- Can search using metadata `key` using: db.similarity_search_with_score(query, filter={"metadata.key": "value"}) for the moment as we are using PGVector collection which is different way to store embeddings and not a Postgresql normal table column but a separate entity where embeddings are stored
**BUT if using above example**
- if using postgresql table column insstead of collection(pgvector) to store metadata as JSONB: use `$` in front of metadata key `db.similarity_search_with_score(query, filter={"metadata.$key": "value"})`

**Note about table stored vector**
- PostgreSQL does not generate embeddings: You must use an embedding model to create the vector representation of your data.
- PostgreSQL VECTOR(4096) column: This is simply a storage type for your pre-computed vectors; it does not perform any transformations on the data.
- Use here ollama mistral:7b to change data into a vector before storing it in the db for example.


____________________________________________________

# retrieve_relevant_vectors object returned:
List[Dict[str,any]]: [{'UUID': info['UUID'], 'content': info['content'], 'score': score}]

# answer_retriever: Object embedded fields plus db row corresponding to that object. So here we could use the 'row_data' field to get other info needed for answer like `title` and `doc_name`
List[Dict[str,any]]: [{ 'UUID': doc['UUID'], 'score': doc['score'], 'content': doc['content'], 'row_data': document},]
 
# similarity_search_with_score
returns a list ordered from the highest score to the lowest score.
The higher the score, the more relevant is the answer.
 
# LangGraph custom state
You just need to create your custom states and use those in the different tools or nodes functions.
Therefore, you can create, update, delete those vars as your app is processing different actions
```python
from typing import TypedDict

class CustomState(TypedDict):
    input: str
    steps_completed: int
    data: Dict[str, str]


def create_initial_state(input_value: str) -> CustomState:
    return CustomState(input=input_value, steps_completed=0, data={})

# example of function that uses the state in it:
def step_1(state: CustomState):
    (...)
    Some logic
    (...)
    Some logic
    (...)
    print("---Step 1---")
    state['steps_completed'] += 1
    state['data']['step_1'] = "Processed Step 1"
    return state

(...)
Node creation
(...)
Tools and conditional edges etc...
(...)
compilation of graph
graph=...

from langgraph import GraphRunner

# Create the initial state
initial_state = create_initial_state("Initial Input")

# Create the graph runner to run the langgraph compile object. Here `graph` is the compiled langgraph graph
runner = GraphRunner(graph)

# Run the graph
final_state = runner.run(initial_state)

print("Final State:", final_state)
```

# Langchain prompts
- PromptTemplate.from_template(template=a_template_str_with_LCEL_{variable}) = use this with LCEL {variable} and when you invoke pass in the variable, eg.: ...invoke({"variable": "value"})
- PromptTemplate(template=a_template_str_with_LCEL_{variable}, input_variables=["variable",]) Or use here `.format({"variable": "value"})` or when using invoke pass in the variable. eg.: ...invoke({"variable": "value"}) 

- advanced prompts
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
) 

 # use of invoke, run and predict
- Use invoke() when you have a full chain involving multiple components (prompt template, LLM, parser) and need to execute the entire chain. so here for chains for example
- Use run() when you are dealing with a StateGraph or similar workflow structure and need to execute the defined sequence of operations. so here for full graph run for example
- Use predict() when you need to generate a prediction from a model based on specific input. so here just for input/output from model
 
 
# ISSUES
 
## Database storage `''` vs  `""`

- Here the issue is that the values are stored to db using double quotes `""` while those should be encapsulated in single quote `''`
```bash
Error inserting row 0: column "https://chikarahouses.com" does not exist
LINE 3:                 WHERE doc_name = "https://chikarahouses.com"...
                                         ^
```

- Here we get single quotes `''` for valeu stored in db and for `table_name` we can use sql.Identifier which will use `""`
```sql
cursor.execute(sql.SQL("""
        INSERT INTO {} (id, doc_name, title, content, retrieved)
        VALUES (%s, %s, %s, %s, %s)
    """).format(sql.Identifier(table_name)), 
    [row["id"], row["doc_name"], row["title"], row["content"], row["retrieved"]])
    conn.commit()
```

## Data serialization saved to state which has limit of storage
We can have large documents and we want to store those in db by creating a dataframe beforehands.
This dataframe stored to state to pass it to the next node which is going to store it to the database.
- But the dtaframe can be too large and we used JSON serialization but it is not optimal for size.
- We are using for the moment parquet stored to memory which takes less space (pyarrow)
- But instead of having different nodes to do those jobs we could just make that node do those two tasks parsing document and storing to db to prevent large state variables

```python
import pyarrow as pa
import pyarrow.parquet as pq

# Serialize the dataframe to a Parquet format stored in memory
def serialize_dataframe(df):
    table = pa.Table.from_pandas(df)
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return sink.getvalue().to_pybytes()

# Deserialize the dataframe from the Parquet format
def deserialize_dataframe(data):
    buffer = pa.BufferReader(data)
    table = pq.read_table(buffer)
    return table.to_pandas()
```

Attempting to serialize binary or non-UTF-8 encoded data into a field that expects text content, leading to encoding errors
therefore, we need to find another solution to make it easy. I though using a file to store the dataframe and that is what we are going to do.
- store df to file
- store path location of that file in the lanngraph states message of the node.
- this way we can have separate nodes for separate tasks

```python
import uuid

def save_dataframe(df, directory="dataframes"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_id = str(uuid.uuid4())
    file_path = os.path.join(directory, f"{file_id}.parquet")
    df.to_parquet(file_path)
    return file_path

def load_dataframe(file_path):
    return pd.read_parquet(file_path)
```

## PromptTemplate object
One import used did not return a `PromptTemplate` but a `str` which is annoying
- from langchain_core.prompts.prompt import PromptTemplate : type = `str`
- from langchain.prompts import PromptTemplate : type = `PromptTemplate`
 
## Incorrect Type Checking:
- The original type checks were checking if the formatted prompt was of type str instead of PromptTemplate.
```python
if not isinstance(formatted_system_prompt, str):
    raise ValueError("formatted_system_prompt is not an instance of PromptTemplate")
if not isinstance(formatted_human_prompt, str):
    raise ValueError("formatted_human_prompt is not an instance of PromptTemplate")
```
This check should verify if the prompt is an instance of PromptTemplate, not str.

- Correct Type Checking:
The corrected code properly checks if the formatted prompts are instances of PromptTemplate.
```python
if not isinstance(formatted_system_prompt, PromptTemplate):
    raise ValueError("formatted_system_prompt is not an instance of PromptTemplate")
if not isinstance(formatted_human_prompt, PromptTemplate):
    raise ValueError("formatted_human_prompt is not an instance of PromptTemplate")
```

## Returning String Instead of PromptTemplate:

- The prompt_creation function returned a str when formatting the template, but it should always return a PromptTemplate.
```python
if input_variables:
    return PromptTemplate(
        template=target_prompt["template"],
        input_variables=input_variables
    ).format(**kwargs)
else:
    return PromptTemplate(
        template=target_prompt["template"],
        input_variables=input_variables
    )
```
This code incorrectly handles the formatting and type of the returned object.

- Returning PromptTemplate Instead of str:
The prompt_creation function ensures that it always returns a PromptTemplate, even after formatting.
```python
def prompt_creation(target_prompt: Dict[str, Any], **kwargs: Any) -> PromptTemplate:
    input_variables = target_prompt.get("input_variables", [])
    
    prompt = PromptTemplate(
        template=target_prompt["template"],
        input_variables=input_variables
    )
    
    formatted_template = prompt.format(**kwargs) if input_variables else target_prompt["template"]
    
    return PromptTemplate(
        template=formatted_template,
        input_variables=[]
    )
```

## Improper Handling of Empty Input Variables:

- The original code does not correctly handle cases when input_variables is empty, leading to potential errors during formatting or type mismatches.

- Proper Handling of Empty Input Variables:
The corrected code handles cases where input_variables might be empty by defaulting to an empty list.
```python
input_variables = target_prompt.get("input_variables", [])
```

## Incorrect Prompt Initialization:

- When initializing SystemMessagePromptTemplate and HumanMessagePromptTemplate, the prompt should be a PromptTemplate, not a dictionary or string.
```python
system_message_prompt = SystemMessagePromptTemplate(prompt={"template": formatted_system_prompt.template})
```

## Correct Prompt Initialization:

- The corrected code initializes SystemMessagePromptTemplate and HumanMessagePromptTemplate with the PromptTemplate directly.
```python
system_message_prompt = SystemMessagePromptTemplate(prompt=formatted_system_prompt)
human_message_prompt = HumanMessagePromptTemplate(prompt=formatted_human_prompt)
```

## Corrected full functions and imports"
```python
from typing import Dict, Any
from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate

def prompt_creation(target_prompt: Dict[str, Any], **kwargs: Any) -> PromptTemplate:
    input_variables = target_prompt.get("input_variables", [])
    
    prompt = PromptTemplate(
        template=target_prompt["template"],
        input_variables=input_variables
    )
    
    formatted_template = prompt.format(**kwargs) if input_variables else target_prompt["template"]
    
    return PromptTemplate(
        template=formatted_template,
        input_variables=[]
    )

def chat_prompt_creation(system_prompt: Dict[str, Any], human_prompt: Dict[str, Any], *args: Dict[str, Any], **kwargs: Any) -> ChatPromptTemplate:
    ai_message_prompt_list = []
  
    # Define system prompt and user prompt
    formatted_system_prompt = prompt_creation(system_prompt, **kwargs)
    formatted_human_prompt = prompt_creation(human_prompt, **kwargs)

    # Ensure the formatted prompts are instances of PromptTemplate
    if not isinstance(formatted_system_prompt, PromptTemplate):
        raise ValueError("formatted_system_prompt is not an instance of PromptTemplate")
    if not isinstance(formatted_human_prompt, PromptTemplate):
        raise ValueError("formatted_human_prompt is not an instance of PromptTemplate")

    system_message_prompt = SystemMessagePromptTemplate(prompt=formatted_system_prompt)
    human_message_prompt = HumanMessagePromptTemplate(prompt=formatted_human_prompt)
    print("system message prompt: ", system_message_prompt)
    print("human message prompt: ", human_message_prompt)
  
    # Can be used as an example of an answer saying "This is an example of answer..."
    for arg in args:
        formatted_ai_message_prompt = prompt_creation(arg, **kwargs)
        if not isinstance(formatted_ai_message_prompt, PromptTemplate):
            raise ValueError("formatted_ai_message_prompt is not an instance of PromptTemplate")
        ai_message_prompt = AIMessagePromptTemplate(prompt=formatted_ai_message_prompt)
        ai_message_prompt_list.append(ai_message_prompt)
  
    # Instantiate the prompt for the chat
    if ai_message_prompt_list:
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt, *ai_message_prompt_list])
    else:
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
  
    return chat_prompt
```

## Final prompt structure will be:
- simplification of prompt structure and uniformization
```code
# to simplify all prompts will have this structure and we will pull or fill what is needed for easy llm call
<function_name>_prompt = {
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
```

# use of invoke using the prompt
```python
# where prompt_text is targetting what is needed eg: `prompt_text = <function_name>_prompt["system"]["template"]`
response = llm.invoke(prompt_text)
```

### Lot of confusion around prompt and errors: The following is from the doc, just follow that!!!

##### NORMAL VERSION
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# get the template part of the prompt
prompt = (
    PromptTemplate.from_template("Tell me a joke about {topic}")
    + ", make it funny"
    + "\n\nand in {language}"
)
# get input variable if any for the prompt template part if using .format
prompt.format(topic="sports", language="spanish")

# for example for the model
model = ChatGroq(...)

# or if not using .format just put prompt in LLMChain and put input vars in run()
chain = LLMChain(llm=model, prompt=prompt)
chain.run(topic="sports", language="spanish")

# make the function for that: prompt_template is going to be imported from the prompt module
def make_normal_prompt(prompt_template: Dict, **kwargs):
  model = ChatGroq(...)
  prompt = (
    PromptTemplate.from_template(prompt_template["template"])
  )
  
  if prompt_template["input_variables"]:
    chain = LLMChain(llm=model, prompt=prompt)
    response = chain.run(**kwargs)
    return response.content.split("```")[1].strip("python").strip()

  chain = LLMChain(llm=model, prompt=prompt)
  response = chain.run()
  return response.content.split("```")[1].strip("python").strip()

## Groq internal server error
We need to handle this error if it happens
```python
groq.InternalServerError: Error code: 503 - {'error': {'message': 'Service Unavailable', 'type': 'internal_server_error'}}
```

## Custom State Not Persisting Between Graphs When Using A Chain Of Specialized Graphs
- Good to know: Langgraph clear th ememory so the there wont be any persistence in the values of the custom state

- `Pydantic` is a stateless state class because it is made to validate the fields initially

- We can use a memory system by saving to database the state, file or env vars.
In this project we are using a dynamic env file that is saving the output of each graph for the next graph to be able to use it as first message

- OR: save to file when one graph is done and load the state when the next graph is starting
Eg.
```python
from pydantic import BaseModel
import json

# Pydantic state class
class GraphState(BaseModel):
    field1: str = ""
    field2: int = 0
    field3: list = []
```

# save state at the end of the graph
```python
def save_state_to_file(state: GraphState, filename="state.json"):
    with open(filename, 'w') as f:
        json.dump(state.dict(), f)

# Example usage:
graph_state = GraphState(field1="value1", field2=123, field3=["item1", "item2"])
save_state_to_file(graph_state)
```

# load state at the beginning of the next graph
```python
def load_state_from_file(filename="state.json") -> GraphState:
    with open(filename, 'r') as f:
        state_dict = json.load(f)
    return GraphState(**state_dict)

# Example usage:
graph_state = load_state_from_file()
print(graph_state)
```

## Node error with pydantic validation error when returning a AI message or any other human/ system
- Pydantic is expecting us to return a str in the `content` part of the AI message
- We therefore, use serialization using json.dumps and if needed to fetch content we could deserialize it using json.loads

- eg. Error:
```python

# Error case
def answer_user(state: MessagesState):
  messages = state['messages']
  #print("Message state: ", messages)
  
  # here last message is <type dict>`
  last_message = {"first_graph_message": messages[0].content, "second_graph_message": messages[1].content, "last_graph_message": messages[-1].content}
  return {"messages": [{"role": "ai", "content": last_message}]}

Outputs:
Exception: An error occured while running 'retrieval_flow': 2 validation errors for AIMessage
content
  str type expected (type=type_error.str)
content
  value is not a valid list (type=type_error.list)

```

- Eg. Error fixed using serialization to json string
```python
def answer_user(state: MessagesState):
  messages = state['messages']
  #print("Message state: ", messages)
  
  # we need to have this ai content message be a string otherwise we will get a pydantic validation error expecting a str after we could deserialize this using `json.laods`
  last_message = json.dumps({"first_graph_message": messages[0].content, "second_graph_message": messages[1].content, "last_graph_message": messages[-1].content})
  return {"messages": [{"role": "ai", "content": last_message}]}
```

- Eg. Error fixed by using AI message for each result and putting those in a list
```python
def answer_user(state: MessagesState):
    messages = state['messages']
    return {
        "messages": [
            {"role": "ai", "content": messages[0].content},
            {"role": "ai", "content": messages[1].content},
            {"role": "ai", "content": messages[-1].content},
        ]
    }
```

## structured output error of returned value not JSON seriazable

```bash
# the error
TypeError: Object of type OutputParserException is not JSON serializable

```

- use better prompting could help but still need to figure out if more efficient in the `system` prompt template or in the `Field(...decription="...")` of the structure output class: `Do not use any markdown code block delimiters (i.e., ``` and ```python) replace those ''. This value MUST be JSON serializable and deserializable, therefore, make sure it is well formatted.`

- or maybe we could get the answer and then check if the pydantic parser can parse it , if not we find a coding way to get what we need from the response.

- will try to just improve prompt wiht good and bad example of JSON output
```bash
{
  "bad_example": "This is a bad example with issues like unescaped quotes in 'keys' and 'values', improper use of ```markdown``` delimiters, and mixed single/double quotes."
}
{
  "good_example": "This is a good example where quotes are properly escaped, like this: \"escaped quotes\", and no markdown code block delimiters are used."
}
```

- or use this in the prompt:
```
"You are a Python script expert and will return a Python script formatted strictly as a valid JSON object. Do NOT include any explanations, markdown, or non-JSON text. The script should be returned as a JSON object with the key 'script', and the value should be a string containing the Python script. The Python code should be in a single block and fully executable. Replace any markdown delimiters (such as ``` or ```python) with an empty string (''). Example JSON format:

{
  'script': 'import requests\n\n...'
}

Ensure that the output is a valid JSON object and does not contain any additional text or explanation."

```

# asking llm to return valid json but llm oevr-escapes stuff
```bash
{'name': 'gemma\\_3\\_7b', 'reason': 'Both the script and requirements for gemma\\_3\\_7b are correct and it fulfills the initial intent. The other scripts have the same code and will also work, but since we can only choose one, we will go with gemma\\_3\\_7b.'}
```
- just worked onthe business logic and feeding the llm with only the name without `_` and then in another node having a list of the real name and getting the real name that I want just checking if there a slice of the name created in it and keep going.
```python
...
# conditional edge that decides if we 'rewrite_or_create_requirements_decision', we split the name and get rid of the '_' to keep only the main name of the llm:
    for elem in valid_code:
      for llm, script in elem.items():
        with open(f"./docker_agent/agents_scripts/agent_code_execute_in_docker_{llm}.py", "w", encoding="utf-8") as llm_script:
          llm_script.write("#!/usr/bin/env python3")
          # splitting the name here and getting what we need 
          llm_name = llm.split("_")[0]
          llm_script.write(f"# code from: {llm_name} LLM Agent\n# User initial request: {os.getenv('USER_INITIAL_REQUEST')}\n'''This code have been generated by an LLM Agent'''\n\n")
          llm_script.write(script)

...

# in conditional edge 'create_requirements_or_error' naming the scripts:
  # list created with real names
  llm_name_list_to_pick_from = ["gemma_3_7b", "llama_3_8b", "llama_3_70b"]
  
  if "success" in outcome: # Dict[Dict[name, reason]]
    # update state with the name of the llm only that will be passed to next node so it can find that file code and execute it in docker
    llm_name = outcome["success"]["llm_name"]
    print("LLM CHOSEN TO BE EXECUTED IN DOCKER: ", llm_name)
    # looping through to get those match made with the real names
    for name in llm_name_list_to_pick_from:
      if llm_name in name:
        llm_agent = name 
    # updating staes
    state["messages"].append({"role": "system", "content": json.dumps({"llm_name": llm_agent})})
```

_____________________________

##### CHAT VERSION
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


prompt = SystemMessage(content="You are a nice pirate")
new_prompt = (
    prompt + HumanMessage(content="hi") + AIMessage(content="what?") + "{input}"
)

new_prompt.format_messages(input="i said hi")

Should Output an Object Like:
`[SystemMessage(content='You are a nice pirate', additional_kwargs={}),
 HumanMessage(content='hi', additional_kwargs={}, example=False),
 AIMessage(content='what?', additional_kwargs={}, example=False),
 HumanMessage(content='i said hi', additional_kwargs={}, example=False)]`
 
 chain = LLMChain(llm=model, prompt=new_prompt)

chain.run("i said hi")

### Errors to handle list:
#### GROQ Errors
- calling api: groq.InternalServerError: Error code: 502 - {'error': {'type': 'internal_server_error', 'code': 'service_unavailable'}}   -> solution implement retries exponential and with max retry times and prepare fallback local lmstudio or ollama maybe... and trace to know when this happen

#### LLM calls Chaining vs direct call
Direct call is better as no issues with formatting of input, LLM understands more what to do.
- for the query:
```bash
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
             """,
             "human": "I have been in the ecommerce site chikarahouses.com and want to know if they have nice product for my Japanese garden"
```
- chaining and then invoke returns:
```bash
{
    'url': 'https' + user_query.split('://')[-1] if any(x in user_query for x in ['http', 'www']) else '',
    'pdf': user_query.split('.pdf')[-2] if '.pdf' in user_query else '',
    'text': user_query if not ('http' in user_query or '.pdf' in user_query) else user_query.split('.pdf')[0] if '.pdf' in user_query else user_query.split('/')[-1],
    'question': re.sub(r'[\[\]\(\)\{\}\|@\\/:;><\']+', ' ', user_query.strip()).strip() if any(x in user_query for x in ['how', 'what', 'where', 'when', 'who', 'which', 'why', 'is', 'are', 'does', 'do', 'did', 'was', 'were', 'has', 'have', 'had']) else ''
}
```
- Direct call retuns:
```bash
{
  'url': 'https://chikarahouses.com',
  'pdf': '',
  'text': 'I have been in the ecommerce site chikarahouses com and want to know if they have nice product for my Japanese garden',
  'question': 'Do they have nice products for a Japanese garden at Chikara Houses?'
}
```

##### Structure of document data when pdf content is parsed
- function : `+pdf_to_sections`
```code

 {
  'text': 'Exploring the cellular and molecular mechanisms of temperature sensation. One of my favourite things to learn in high school chemistry was how temperature workedâ \\200\\224the average kinetic energy of substances being responsible for how hot or cold we m ight feel something to be. This was simple enough. Fast molecules that bump a lot = hotter. Slow molecules that donâ\\200\\231t bump as often = colder. But that doesnâ\\200\\231t exactly explain how our brain can sense and relay information about the temperature of a substance . When you pick up an ice cube, how does your brain know what you’re touching is cold? How do we turn that measure of kinetic energy into signals that we can regularly identify as co', 
  'section': 'Feb 22, 2024', 
  'document': './doc/feel_temperature.pdf'
  }, 
  {
   'text': 'Thatâ\\200\\231s what Iâ\\200\\231m aiming to explain today, and having taken a course in cellu lar and molecular biology, this has only just started to make sense to me.', 
   'section': 'ld?', 
   'document': './doc/feel_temperature.pdf'
   }, 
  {
   'text': 'The main challenge here is figuring out how thermal and kinetic energy is converted into el ectrical signals that can inform the brain of the temperature of a substance. When it comes to receiving or transmitting any sort of signals, it’s a safe assumption that receptors of some sort play a massive role in the travel of information. Thatâ\\200\\231s no different he', 
   'section': 'Thermoreceptors', 
   'document': './doc/feel_temperature.pdf'
   }, 
   {
    'text': 'A simplified diagram of thermoreceptors in the dermis (layer of skin) A simplified diagram of thermoreceptors in the dermis (layer of skin) | Source Thermoreceptors are free nerve endings (FNE), which extend until the mid-epidermis, or the outermost layer of our skin. The ganglionsâ\\200\\224the structures responsible for receiving information from extracellular stimuliâ\\200\\224are not enclosed within a membrane, which a llows them to detect various physical stimuli through interactions involving our skin. Additionally, we have 2 types of thermoreceptors, cold and warm receptors, that can be vari ed in concentration throughout the body. Iâ\\200\\231m sure youâ\\200\\231ve noticed that your ears and face get excessively cold in the winter, thatâ\\200\\231s due to a greater presence', 
     'section': 'Introducing, the thermoreceptor!', 
     'document': './doc/feel_temperature.pdf'
    }, 
   {
    'text': 'To understand how these FNE thermoreceptors work, it is important to first understand a lit tle bit about neurotransmission. The strength of a signal processed by the brain is depende nt on the frequency of a neuronâ\\200\\231s firing. Of course, we are always interacting with substances and objects, which all have surface temperatures that our thermoreceptors detec t. However, these tend to be at room temperature, corresponding to normal firing rates. Int eractions with cold or hot stimuli change the firing rate of their corresponding thermorece', 
     'section': 'of cold receptors in those areas.', 
     'document': './doc/feel_temperature.pdf'
    }, 
    {
     'text': 'Touching something with a temperature between around 5â\\200\\22330Â°C will increase cold rec eptor firing and decrease warm receptor firing. Accordingly, stimuli of temperature between 30â\\200\\22345Â°C will increase warm receptor firing and decrease cold receptor firing. Thi s changing pattern of neural impulses by FNEs is what indicates to your brain (and therefor e, you) that youâ\\200\\231re touching something cold or hot. In fact, when you touch something hot enough to hurt, another type of FNEs called nocicepto rs are activated, which signals to your brain that something youâ\\200\\231re touching is cau sing you pain. That, coupled with a high warm receptor firing rate, makes you aware that yo u might burn yourself if you keep in contact with the given substance.feel_temperature.txt Thu May 09 19:02:26 2024 2', 
      'section': 'ptors.', 
      'document': './doc/feel_temperature.pdf'
     }, 
     {
      'text': 'So, weâ\\200\\231ve covered how we feel one type of hot, but what about that burning feeling', 
      'section': 'Ion Channels', 
      'document': './doc/feel_temperature.pdf'
     }, 
     {
      'text': 'A diagram of TRP ion channels sensitive to temperature and chemical heat sources A diagram of TRP ion channels sensitive to temperature and chemical heat sources | Source The heat of a chilli pepper is slightly different from what weâ\\200\\231ve talked about befo re. These plants contain capsaicin, a chemical agent that acts on interior surfaces in your mouth (and tongue, especially) to make it feel like your tongue is burning. As aforementio ned, for this change in sensation in your mouth to happen, there must be some receptor invo lved that can turn a physical or chemical stimulus into signals your brain can decipher. In this case, this receptor happens to be the TRPV1 receptor (crazy name, just think of it as a warm receptor), which causes an influx of sodium and calcium into nerve cells. This init iates the firing of associated nerves that indicate to the brain that something is on fire in your mouth, figuratively but Iâ\\200\\231m sure that’s what it feels like, more or less. T RPM8 receptors are similar in mechanism, but instead conduct messages about cold stimuli wh en cooling agents like methanol bind to the receptors. Heat and temperatures are so fascinating to me, how we can perceive so strongly the change in movements of such small molecules. If you want to learn more about different types of re ceptors that signal for responses to other physical stimuli like pressure or touch, check o ut the following short video that inspired me to make this article!', 
       'section': 'right after eating an unexpectedly spicy chilli?', 
       'document': './doc/feel_temperature.pdf'
      }
  ]
```

#### Next
- change all ```python ``` by ```markdown ``` in prompts as the stupid llm sometimes interpret the ```python ```` as being creation of python code... - OK
- see if pdf_parser shouldn't be simplified and just use same process as webpage parser and get all text and then chunk it and then summarize it... not sure yet as pdf parser that we have take into consideration tables and more ... we keep it like that for the moment  
- test this function: process_query form app.py and make all necessary imports to app.py - OK
- Where is the function that stores to the DB the url/pdf fetched - OK function created need to be tested
- export prompts to the prompt file and import those to be used in the functions that summarize text and make tiles - OK


# PSQL

### postgresql 
```bash
# connect
sudo -u postgres psql
```

```sql
# create user
CREATE USER creditizens WITH PASSWORD 'your_secure_password';
# set priviledges for user to have hands on tables and pgvector
ALTER USER creditizens CREATEDB;
GRANT ALL PRIVILEGES ON DATABASE your_database_name TO creditizens;
GRANT USAGE ON SCHEMA public TO creditizens;
GRANT CREATE ON SCHEMA public TO creditizens;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO creditizens;
# ensure user can activate pgvector
GRANT USAGE ON SCHEMA vector TO creditizens;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA vector TO creditizens;
```

```sql
# list all users
\du
# list all databases
\l
```

```sql
# check priviledge of specific user on specific database
SELECT grantee, privilege_type 
FROM information_schema.role_table_grants 
WHERE table_catalog = 'your_database_name' 
AND grantee = 'your_username';
```

```sql
# list all databases and their owners
SELECT datname AS "Database",
       pg_catalog.pg_get_userbyid(datdba) AS "Owner"
FROM   pg_catalog.pg_database
ORDER BY 1;
```

```sql
# get detailed information about user privileges on all databases
SELECT datname AS "Database",
       pg_catalog.pg_get_userbyid(datdba) AS "Owner",
       pg_catalog.array_to_string(datacl, E'\n') AS "Access Privileges"
FROM   pg_catalog.pg_database
ORDER BY 1;
```

```bash
# terminal command for scripting eg.
# eg.: 2
psql -U postgres -c "\du"
# eg.: 1
psql -U postgres -d your_database_name -c "YOUR_SQL_QUERY_HERE"
```

```bash
# check collection creation:
psql -U creditizens -d creditizens_vector_db -c "SELECT relname FROM pg_class WHERE relname = 'langgraph_collection_test';"
```

```bash
# describe table structure:
psql -U creditizens -d creditizens_vector_db -c "\d+ langgraph_collection_test"
```

```bash
# check collection content:
psql -U creditizens -d creditizens_vector_db -c "SELECT * FROM langgraph_collection_test LIMIT 10;"
```

```bash
# check if collection has been created:
psql -U creditizens -d creditizens_vector_db -c "SELECT relname FROM pg_class WHERE relname = 'langgraph_collection_test';"

```

```bash
# delete all content from some tables, can add `CASCADE` at the end if thsoe tabled have foreign keys
TRUNCATE TABLE documents, langchain_pg_collection, langchain_pg_embedding;
```

```bash
# list users
\du
# list databases
\l
# select database
\c creditizens_vector_db;
# list for tables in that db
\dt
```

#### psycopg2 (get also psycopg2-binary) security against SQL injection

```python
import psycopg2
from psycopg2 import sql
# then: Using sql.SQL and related classes is recommended when you need to insert identifiers like table names or column names dynamically. This ensures that these values are safely quoted and prevents SQL injection
```

# Next
- keep in mind the pdf parser that might need to be refactored to chunk using same process as the webpage parser one.
- test function that stores data in database- OK
- create all functions that each nodes will need (tools and other functions)
- create states and use workflow functions to save in each state, name the state the name of the node and the name of the edge fo easy logic and limit confusion, therefore, separate states (have many mini states) and find a way to empty those at the right moment with one function at the end of graph or some depending on the logic.
- See if you need function to get rid of DB info. Create the function that resets the db to zero and the cache to zero as well so that we have the option to delete everything for some future task that doesn't need the data to persist forever in the DB.

## `os.system("<terminal command>")` vs subprocess
- the `os.system("<terminal command>")`: only returns the exit status of the command, not the output of the command itself.

- subprocess is going to be able to get the stdout and stderr
eg:
```python
import subprocess

result = subprocess.run(
    ["psql", "-U", "creditizens", "-d", "creditizens_vector_db", "-c", "SELECT * FROM test_doc_recorded"],
    capture_output=True,
    text=True
)

print(result.stdout)
```

# RETURNED OUPUT FOR CERTAIN FUNCTIONS
- eg.: Chunk size = 4000 `create_chunks_from_db_data(fetch_data_from_database, 4000)` 
code```
[
  [
    {'UUID': '0d3531c5-bf35-4e3b-b695-b2c2eaf9f22b', 'content': "To understand FNE thermoreceptors, you need to know about neurotransmission. The brain's signal strength relies on neuron firing frequency. We interact with objects at room temperature, which causes normal firing rates in thermoreceptors. However, cold or hot stimuli change the firing rate of their corresponding thermoreceptors."}, 
    {'UUID': '112074b3-09e0-4759-b32e-c1df851fa219', 'content': "To understand FNE thermoreceptors, you need to know about neurotransmission. The brain's signal strength relies on neuron firing frequency. We interact with objects at room temperature, which causes normal firing rates in thermoreceptors. However, cold or hot stimuli alter the firing rate of their corresponding thermoreceptors."}, 
    {'UUID': '1e6a1648-227f-4843-bf2b-7073a192f8cf', 'content': 'Thermoreceptors in the dermis are free nerve endings that detect temperature changes. They extend to the mid-epidermis and interact with extracellular stimuli. There are two types: cold and warm receptors, with varying concentrations throughout the body. A higher concentration in ears and face can cause excessive coldness in winter.'}, 
    {'UUID': '240c0f7e-1d39-423c-b03d-91dc4aee8473', 'content': 'The challenge is converting thermal and kinetic energy into electrical signals for the brain. Receptors are key in signal transmission, and they play a massive role here as well.'}, 
    {'UUID': '2fdd1c07-c465-4b5e-887b-d7856d179325', 'content': "Exploring the cellular and molecular mechanisms of temperature sensation goes beyond the average kinetic energy of substances. When you touch something cold, like an ice cube, how does your brain know it's cold? This involves understanding how the measure of kinetic energy is converted into signals that our brain can identify as cold.\n\nAt the molecular level, temperature sensation is mediated by ion channels in specialized nerve endings called nociceptors. These channels are sensitive to changes in temperature and open or close in response to these changes, allowing ions to flow in and out of the cell. This ion flow generates electrical signals that are transmitted to the brain, where they are interpreted as temperature sensations.\n\nOne example of a temperature-sensitive ion channel is TRPM8, which is activated by cold temperatures. When the temperature drops, TRPM8 channels open, allowing calcium ions to flow into the cell. This ion flow triggers a series of events that ultimately leads to the generation of an electrical signal that is transmitted to the brain.\n\nOverall, the cellular and molecular mechanisms of temperature sensation involve the detection of changes in kinetic energy by specialized ion channels in nerve endings, which convert these changes into electrical signals that are transmitted to the brain."},
    {'UUID': '569ce0d4-d21e-4f7f-a8a3-ffc8c30a4e1e', 'content': 'The challenge is converting thermal and kinetic energy into electrical signals for the brain. Receptors are key in signal transmission, and they play a massive role here as well.'}, 
    {'UUID': '583686a4-3708-4333-8507-1fe84adad58a', 'content': 'Perception of temperature is determined by the firing rates of cold and warm receptors. When touching an object between 5-30°C, cold receptor firing increases and warm receptor firing decreases, indicating cold. When touching an object between 30-45°C, warm receptor firing increases and cold receptor firing decreases, indicating warmth. If an object is hot enough to cause pain, nociceptors are activated, signaling pain and further indicating potential injury.'}, 
    {'UUID': '625ccd1b-d646-4dd2-af2c-4e9ae6047074', 'content': 'The challenge is converting thermal and kinetic energy into electrical signals for the brain. Receptors are key in signal transmission, and they play a massive role here as well.'}, 
    {'UUID': '6b9f21ed-2123-40e2-89ad-fa5ed55d8a87', 'content': 'TRP ion channels and temperature/chemical heat sources\n- TRPV1 receptor (warm receptor) responsible for burning sensation from capsaicin\n- Influx of sodium and calcium into nerve cells\n- Initiates firing of nerves signaling to brain\n- TRPM8 receptors conduct messages about cold stimuli'}, 
    {'UUID': '7079049c-f4d3-47a3-af9c-9c660e801665', 'content': '`"Explaining a topic related to cellular and molecular biology is the goal today. This subject has recently begun to make sense after taking a course in this field."`'}, 
    {'UUID': '775ee194-fe66-439a-b1b5-048f41bb0e58', 'content': '`"Explaining a topic related to cellular and molecular biology is the goal today. This subject has recently begun to make sense after taking a course in this field."`'}
  ], 
  [
    {'UUID': '775ee194-fe66-439a-b1b5-048f41bb0e58', 'content': '`"Explaining a topic related to cellular and molecular biology is the goal today. This subject has recently begun to make sense after taking a course in this field."`'}, 
    {'UUID': '78132fee-d8c0-40d0-ab0d-b813bb90b12b', 'content': 'Sure, let\'s discuss the "burning" sensation! It\'s important to note that this feeling is different from the "hot" sensation we previously covered. A burning sensation can be a sign of irritation or injury to the skin or mucous membranes. It can be caused by a variety of factors, such as exposure to chemicals, extreme temperatures, or certain medical conditions. If you\'re experiencing a burning sensation, it\'s important to identify the cause and seek appropriate treatment to prevent further damage.'}, 
    {'UUID': '79612291-e138-486b-baaa-c3a20b7853db', 'content': 'Perception of temperature is determined by the firing rates of cold and warm receptors. When touching an object between 5-30°C, cold receptor firing increases and warm receptor firing decreases, indicating cold. When touching an object between 30-45°C, warm receptor firing increases and cold receptor firing decreases, indicating warmth. If an object is hot enough to cause pain, nociceptors are activated, signaling pain and further indicating potential injury.'}, 
    {'UUID': '7ce310f6-fb52-4e99-9485-6ef31f0004e8', 'content': 'Thermoreceptors in the dermis are free nerve endings that extend to the mid-epidermis. They are not enclosed within a membrane, allowing them to detect physical stimuli through interactions with our skin. There are two types of thermoreceptors: cold and warm receptors, which can vary in concentration throughout the body. For example, the ears and face have a greater concentration of cold receptors, making them more susceptible to feeling cold in winter.'}, 
    {'UUID': '7d14e6f1-54d6-4cd3-9946-96e9a377e05b', 'content': 'TRP ion channels and temperature perception | Source\n-------------------------------------------------\n\nTRP ion channels are sensitive to temperature and chemical heat sources, such as capsaicin in chilli peppers. The TRPV1 receptor, also known as the "warm receptor," is responsible for the burning sensation in the mouth caused by capsaicin. This receptor allows an influx of sodium and calcium into nerve cells, initiating the firing of nerves that signal to the brain that something is burning. Similarly, TRPM8 receptors conduct messages about cold stimuli when cooling agents like methanol bind to them. The ability of the body to perceive changes in temperature through these receptors is fascinating, as it demonstrates the body\'s ability to translate physical or chemical stimuli into signals that the brain can understand. For more information on other types of receptors that signal responses to physical stimuli, check out the following short video.'}, 
    {'UUID': '8bfb6c93-9766-4fb5-aa68-cfb5cf7aa58a', 'content': 'Thermoreceptors in the dermis are free nerve endings that extend to the mid-epidermis. They are not enclosed within a membrane, allowing them to detect physical stimuli through interactions with our skin. There are two types of thermoreceptors: cold and warm receptors, which can vary in concentration throughout the body. For example, the ears and face have a greater concentration of cold receptors, making them more prone to feeling cold in winter.'}
  ], 
  [
    {'UUID': '8bfb6c93-9766-4fb5-aa68-cfb5cf7aa58a', 'content': 'Thermoreceptors in the dermis are free nerve endings that extend to the mid-epidermis. They are not enclosed within a membrane, allowing them to detect physical stimuli through interactions with our skin. There are two types of thermoreceptors: cold and warm receptors, which can vary in concentration throughout the body. For example, the ears and face have a greater concentration of cold receptors, making them more prone to feeling cold in winter.'}, 
    {'UUID': '8f038119-5328-4bcf-8c3f-fa6229c31a94', 'content': 'Sure, let\'s discuss the "burning" feeling you mentioned. It\'s possible that you\'re referring to a sensation of heat that is more intense and persistent than the typical warmth we feel. This type of heat can be caused by a variety of factors, such as inflammation, infection, or nerve damage.\n\nInflammation is a natural response of the body to injury or infection, and it can cause a localized increase in temperature, redness, and swelling. This is why we might feel a burning sensation when we have a cut or scrape that becomes infected.\n\nNerve damage can also cause a burning sensation, as damaged nerves may send incorrect signals to the brain. This is known as neuropathic pain, and it can be caused by conditions such as diabetes, shingles, or multiple sclerosis.\n\nRegardless of the cause, a burning sensation can be uncomfortable and even debilitating. If you\'re experiencing this type of heat, it\'s important to speak with a healthcare professional to determine the underlying cause and develop an appropriate treatment plan.'}, 
    {'UUID': '98f1b511-bd03-43d6-93af-be88306cc4e8', 'content': 'Sure, let\'s discuss the "burning" feeling you mentioned. It\'s possible that you\'re referring to a sensation of heat that is more intense and persistent than the typical warmth we feel. This type of heat can be caused by a variety of factors, such as inflammation, infection, or nerve damage.\n\nInflammation is a natural response of the body to injury or infection, and it can cause a localized increase in temperature, redness, and swelling. This is why we might feel a burning sensation when we have a cut or scrape that becomes infected.\n\nNerve damage can also cause a burning sensation, as damaged nerves may send incorrect signals to the brain. This is known as neuropathic pain, and it can be caused by conditions such as diabetes, shingles, or multiple sclerosis.\n\nRegardless of the cause, a burning sensation can be uncomfortable and even debilitating. If you\'re experiencing this type of heat, it\'s important to speak with a healthcare professional to determine the underlying cause and develop an appropriate treatment plan.'}, 
    {'UUID': 'aab58a62-2d9a-47ac-8d1b-e788e4318d35', 'content': "To understand FNE thermoreceptors, you need to know about neurotransmission. The brain's signal strength relies on neuron firing frequency. We interact with objects at room temperature, which causes normal firing rates in thermoreceptors. However, cold or hot stimuli alter the firing rate of their corresponding thermoreceptors."}
  ], 
  [
    {'UUID': 'aab58a62-2d9a-47ac-8d1b-e788e4318d35', 'content': "To understand FNE thermoreceptors, you need to know about neurotransmission. The brain's signal strength relies on neuron firing frequency. We interact with objects at room temperature, which causes normal firing rates in thermoreceptors. However, cold or hot stimuli alter the firing rate of their corresponding thermoreceptors."}, 
    {'UUID': 'bd5c4782-2a08-4185-93b2-a9d91bc08ee6', 'content': "Exploring the cellular and molecular mechanisms of temperature sensation goes beyond the average kinetic energy of substances. When you touch something cold, like an ice cube, how does your brain know it's cold? This involves understanding how the measure of kinetic energy is converted into signals that our brain can identify as cold.\n\nAt the molecular level, temperature sensation is mediated by ion channels in specialized nerve endings called nociceptors. These channels are sensitive to changes in temperature and open or close in response to these changes, allowing ions to flow in and out of the cell. This ion flow generates electrical signals that are transmitted to the brain, where they are interpreted as temperature sensations.\n\nOne example of a temperature-sensitive ion channel is the TRPM8 channel, which is activated by cold temperatures. When the temperature drops, the TRPM8 channel opens, allowing calcium ions to flow into the cell. This ion flow generates an electrical signal that is transmitted to the brain, signaling the sensation of cold.\n\nOverall, the cellular and molecular mechanisms of temperature sensation involve the activation of temperature-sensitive ion channels in specialized nerve endings, which convert changes in temperature into electrical signals that are transmitted to the brain."}, 
    {'UUID': 'e38b458b-28e4-4e99-84e4-f8b0c16ad308', 'content': 'Temperature sensation involves intricate cellular and molecular mechanisms. High school chemistry explains the relationship between temperature and the average kinetic energy of molecules, but not how our brains sense and relay this information. The process of interpreting cold, such as when touching an ice cube, involves a series of cellular and molecular events that enable temperature perception.'}, 
    {'UUID': 'e8ace0ea-0f00-4c7c-80a9-de593bcc1289', 'content': 'Perception of temperature is determined by the firing rates of cold and warm receptors. When touching an object between 5-30°C, cold receptor firing increases and warm receptor firing decreases, indicating cold. When touching an object between 30-45°C, warm receptor firing increases and cold receptor firing decreases, indicating warmth. If an object is hot enough to cause pain, nociceptors are activated, signaling pain and further indicating potential injury.'}, 
    {'UUID': 'f00d1e2c-32ff-47ae-b17e-14f8fee12a76', 'content': '`"Explaining a topic related to cellular and molecular biology is the goal today. This subject has recently begun to make sense after taking a course in this field."`'}, 
    {'UUID': 'f93afd2f-fc8e-44c2-9957-458fc837f26e', 'content': 'TRP ion channels are sensitive to temperature and chemical heat sources, such as capsaicin in chilli peppers. The TRPV1 receptor, also known as the "warm receptor," is responsible for the burning sensation in the mouth caused by capsaicin. This receptor allows an influx of sodium and calcium into nerve cells, initiating the firing of nerves that signal to the brain that something is burning. Similarly, TRPM8 receptors conduct messages about cold stimuli when cooling agents like methanol bind to them. The way we perceive changes in temperature is fascinating, as it involves the detection of small movements of molecules. For more information on other physical stimuli receptors, check out this short video.\n\nSource: <https://www.khanacademy.org/science/biology/human-biology/sensory-system/v/sensory-receptors-and-transduction>'}
  ]
]
```

# ERROR PSYCOPG2 CONNECTION
- Need to be careful and close the connection of the db at the end of the workflow an
```bash
psycopg2.InterfaceError: connection already closed
```

- OR need to get rid of the conn parameter and put inside the function the connection to be opened
```bash
conn = connect_db()
```

- OR use a `try/except/finally` and do the whole stuff that needs to be done in the `try` and close the connection in the `finally`
def fetch_documents(table_name: str) -> List[Dict[str, Any]]:
    conn = connect_db()
    try:
        # Fetch data
        # ...
    finally:
        conn.close()

# ollama embeddings
If using `ollama` for embeddings from `langchain.community` you need to find the `ollama.py` file in the virtual env and change the model for the one that you want to use for the embeddings. it is by default set to `llama2`
- location: `/home/creditizens/langgraph/langgraph_venv/lib/python3.10/site-packages/langchain_community`
- filename: `ollama.py`
- approx. line number: `37`
- change to be done: change `model: str = "llama2"` for `model: str = "mistral:7b"`

if you omit that you can check if you want to install ollama and do embedding using its `pip` installation	


# Next
- keep in mind the pdf parser that might need to be refactored to chunk using same process as the webpage parser one.
- test retrieval from data saved by our test.py OK
- create all functions that each nodes will need (tools and other functions)
- create states and use workflow functions to save in each state, name the state the name of the node and the name of the edge fo easy logic and limit confusion, therefore, separate states (have many mini states) and find a way to empty those at the right moment with one function at the end of graph or some depending on the logic.
- See if you need function to get rid of DB info. Create the function that resets the db to zero - OK
- reset the cache to zero as well so that we have the option to delete everything for some future task that doesn't need the data to persist forever in the DB.




8b4a37ce-0e43-4b40-aa05-79c09d07e48d | docs/feel_temperature.pdf | 'Decoding ld?: A Cellular & Molecular Perspective'                                                                                                                                                                                                                                                                                                                                  | `"Explaining a topic related to cellular and molecular biology is the goal today. This subject has recently begun to make sense after taking a course in this field."`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      

 d9472503-1aaa-41f3-bc16-329617b091e9 | docs/feel_temperature.pdf | "Converting Temperature to Electric Signals: Thermoreceptor Function"                                                                                                                                                                                                                                                                                                                 | The challenge is converting thermal and kinetic energy into electrical signals for the brain. Receptors are key in transmitting signals, and here's no exception. They play a massive role in receiving and transmitting temperature information.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | f

 18237035-48d0-490f-8d23-80f609d932bd | docs/feel_temperature.pdf | "Thermoreceptors in Skin: Free Nerve Endings for Heat & Cold"                                                                                                                                                                                                                                                                                                                         | Thermoreceptors in the dermis are free nerve endings that extend to the mid-epidermis. They are not enclosed within a membrane, allowing them to detect physical stimuli through interactions with our skin. There are two types of thermoreceptors: cold and warm receptors, which can vary in concentration throughout the body. For example, the ears and face have a greater concentration of cold receptors, making them more susceptible to feeling cold in winter.

List[Tuple[Documents, score]]
[
 (
 Document(
   page_content=
     '[
       {"UUID": "124d2bec-5a6d-4ba7-9180-94245e78cfad", "content": "Explore AI tools by Chikara Houses in the Openai GPT store. Improve remote work and well-being with their GPTs. Visit their unique shopping rooms, too."},
       {"UUID": "28902448-443b-41a8-8e35-e6e3cb571fa8", "content": "Explore Chikara\'s social media rooms for shopping tours and AI assistant demos.\\nTrack followers, sales, and enhance your lifestyle with Chikara Houses.\\n[Shop Now](Shop Now!)| [Play & Work with AI](Amazing Chikara AI Assistants!)"
       }
     ]'
   ), 
   0.584969858173907
 ),
 (
 Document(
   page_content=
     '[
       {"UUID": "28902448-443b-41a8-8e35-e6e3cb571fa8", "content": "Explore Chikara\'s social media rooms for shopping tours and AI assistant demos.\\nTrack followers, sales, and enhance your lifestyle with Chikara Houses.\\n[Shop Now](Shop Now!)| [Play & Work with AI](Amazing Chikara AI Assistants!)"},
       {"UUID": "97a4b55f-8c22-4b97-bd20-cc245304b5ac", "content": "Get exclusive access to [Different Collection Rooms](http://www.example.com/collectionrooms) for non-subscribers! Visit the description page to enhance your home specifically. Discover [intriguing stories and articles](http://www.example.com/articles)."
       }
     ]'
   ),
   0.6058349904951864
 )
]


# Next
- keep in mind the pdf parser that might need to be refactored to chunk using same process as the webpage parser one.
- data saved to adb and workflow fine but need now to work on the redis cache retrieval part and vectordb search and falldown to a internet search with a separate func for next agent - OK
- create all functions that each nodes will need (tools and other functions)
- create states and use workflow functions to save in each state, name the state the name of the node and the name of the edge fo easy logic and limit confusion, therefore, separate states (have many mini states) and find a way to empty those at the right moment with one function at the end of graph or some depending on the logic.
- See if you need function to get rid of DB info. Create the function that resets the db to zero - OK
- reset the cache to zero as well so that we have the option to delete everything for some future task that doesn't need the data to persist forever in the DB.

# REDIS-CLI

```bash
# from redi-cli, get all keys stored
KEYS *
# or
SCAN 0

# from bash terminal, iterate through keys and get their values `key -> value`
redis-cli --scan --pattern "*" | while read key; do echo "$key -> $(redis-cli get $key)"; done

# delete all keys everythign from the redis server
redis-cli FLUSHALL

# delete keys on a selected redis database only
redis-cli FLUSHDB
```


# PGVECTOR Deprecation new import (https://api.python.langchain.com/en/latest/vectorstores/langchain_postgres.vectorstores.PGVector.html)[docs]
- before: `from langchain_community.vectorstores.pgvector import PGVector`
- now: `from langchain_postgres import PGVector` OR `from langchain_postgres.vectorstores import PGVector`
```bash
# install
pip install -qU langchain-postgres
# run in docker if wanted (we will use our own postgresql datrabase on the server and activate the pgvector extension in this project)
docker run --name pgvector-container -e POSTGRES_USER=langchain -e POSTGRES_PASSWORD=langchain -e POSTGRES_DB=langchain -p 6024:5432 -d pgvector/pgvector:pg16
```


# Next
- keep in mind the pdf parser that might need to be refactored to chunk using same process as the webpage parser one.
- make the tool for internet search fall node which will be the falldown of our query cache/vectordb search fail. This will start next point here do nodes and tools...
- create all functions that each nodes will need (tools and other functions) -OK
- create states and use workflow functions to save in each state, name the state the name of the node and the name of the edge for easy logic and limit confusion, therefore, separate states (have many mini states) and find a way to empty those at the right moment with one function at the end of graph or some depending on the logic.
- reset the cache to zero as well so that we have the option to delete everything for some future task that doesn't need the data to persist forever in the DB.


# PROMPTS TEMPLATE AND LLM CALL

### we will use this format of prompt template and will fill it depending of which kind of llm call we need to perform using this format: `<name_of_function>_prompt`
```python
<name_of_function>_prompt = {
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
```

### we will use this function to see if we are calling llm using question/answer only or chat version system/human/ai
```python
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
```

### Then the llm calling function will call the llm in different way (formatting the call as needed)
```python
def call_chain(model: ChatGroq, prompt: PromptTemplate, prompt_input_vars: Optional[Dict], prompt_chat_input_vars: Optional[Dict]):
  # only to concatenate all input vars for the system/human/ai chat
  chat_input_vars_dict = {}
  for k, v in prompt_chat_input_vars.items():
    for key, value in v.items():
      chat_input_vars_dict[key] = value
  print("Chat input vars: ",  chat_input_vars_dict)

  # default question/answer
  if prompt and prompt_input_vars:
    print("Input variables found: ", prompt_input_vars)
    chain = ( prompt | model )
    response = chain.invoke(prompt_input_vars)
    print("Response: ", response, type(response))
    return response.content.split("```")[1].strip("markdown").strip()
    
  # special for chat system/human/ai
  elif prompt and prompt_chat_input_vars:
    print("Chat input variables found: ", prompt_chat_input_vars)
    chain = ( prompt | model )
    response = chain.invoke(chat_input_vars_dict)
    print("Response: ", response, type(response))
    return response.content.split("```")[1].strip("markdown").strip()
```
**Step of the prompt template use for llm call:**
.1 import the right prompt template normal question/response or chat special system/human/aione
.2 use the template and the function argument to create the prompt
.3 use make_normal_or_chat_prompt_chain_call(llm_client, prompt_input_variables_part: Dict, prompt_template_part: Optional[Dict], chat_prompt_template: Optional[Dict])
.4 use internet search tool if nothing is found or go to the internet search node

Eg.:
```python
from prompts.prompts import test_prompt_tokyo, test_prompt_siberia, test_prompt_monaco, test_prompt_dakar

print("SIBERIA: \n", make_normal_or_chat_prompt_chain_call(groq_llm_llama3_8b, test_prompt_siberia["input_variables"], test_prompt_siberia["template"], {}))
time.sleep(0.5)
print("TOKYO: \n", make_normal_or_chat_prompt_chain_call(groq_llm_llama3_8b, test_prompt_tokyo["input_variables"], test_prompt_tokyo["template"], {}))
time.sleep(0.5)
print("DAKAR: \n", make_normal_or_chat_prompt_chain_call(groq_llm_llama3_8b, {}, {}, test_prompt_dakar))
time.sleep(0.5)
print("MONACO: \n", make_normal_or_chat_prompt_chain_call(groq_llm_llama3_8b, {}, {}, test_prompt_monaco))

'''
# prompts that also fit the function as question/answer but we have decided to use to next example one to have same template and use or fill what is needed from it
test_prompt_siberia = {"template": "What is the weather like in {test_var} usually.\nAnswer only with the schema in markdown between ```markdown ```.", "input_variables": {"test_var": "Siberia"}}
test_prompt_tokyo = {"template": "We is the weather like in Japan in summer usually.\nAnswer only with the schema in markdown between ```markdown ```.", "input_variables": {}}

# Prompt System/Human/Ai that will be used from now on for all prompts
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
'''
```

**Now We have a prompt template and will use the same and will call llms using this kind of example if AI message is needed if will be added but in this example we use just human and system message**

##### prompt template
```code
<function_name>_prompt = {
  "system": {
    "template": """
      <system_message_template_with_or_without_input_variables>
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
```

##### formatting llm call taking what is needed from the imported prompt template
```python
system_message_tuple = ("system",) + (prompt["system"]["template"],)
human_message_tuple = ("human",) + (query.strip(),)
messages = [system_message_tuple, human_message_tuple]

llm_called = llm.invoke(messages)

llm_called_answer = llm_called.content.split("```")[1].strip("markdown").strip()
```



### list of function for node or tool:
#### 1) User initial query management and files or url if exist
- process_query: returns tuple (df_final, content_dict) : will decompose user query and create df of url or pdf OR returns Dict[str,str] {message: user query text to answer to as no url/pdf processing needed}
state saved: question/pdf or question/url or text and webpage or url df

- store_dataframe_to_db: returns Dict[str, str] {error/success: message}
state saved: df saved bool True/False
custom_chunk_and_embed_to_vectordb: returns Dict[str, str] {error/success: message}
state saved: db data chunked and embedded bool True/False

#### 2) Query Retrieval
- query_redis_cache_then_vecotrdb_if_no_cache: returns List[Dict[str,Any]] | str | Dict[str,str] -->
  - List[Dict[str,Any]] is found answer cache/vectordb; 
  - str if nothing found;
  - Dict[str,str] if error {error:message}
state saved: retrieved redis or vectordb -->
  - query hash retrieved List[DICT[str,Any]]
  - query vector retrieved List[DICT[str,Any]]
  - vectordb retrieved List[DICT[str,Any]]
  - nothing retrieved: bool
 
#### Extras
- delete_table
- subprocess function to run code safely in docker tools


### Format of how to make tools
# tool template
class <name_arg_template>(BaseModel):
      <arg_name>: <input_type> = Field(default=<default_value>, description="<description>")

def <func_name>(<arg_name>: <input_type> = <default_value_if_any>) -> <return_type>:
  """
    <func description> 
    
    Parameter: 
    <arg_name> <input_type> : '<descrption>' = {<default_value_if_any>}
    
    Returns: 
    <return_type> <description> 
  """
  <func_logic>
  return <returned_object_stuff>

<var_tool_name> = StructuredTool.from_function(
  func=<func_name>,
  name="<name_of_tool>",
  description=  """
    <description_of_tool>
  """,
  args_schema=<name_arg_template>,
  # return_direct=True, # returns tool output only if no TollException raised
  # coroutine= ... <- you can specify an async method if desired as well
  # callback==callback_function # will run after task is completed
)


# Visualize graph
```python
# install
sudo apt-get install graphviz graphviz-dev
pip install pygraphviz
# install and display graph command
pip install ipython
display(Image(app.get_graph().draw_png()))
```


# Agent Tools Workflow

Just use @tool decorator on any function and use ToolNode.
Agent node which has binded tools will choose which tool is good to use and the next node ToolNode will just execute the tool choosen by the Agent which has already field the schema with the query and have the right schema for that tool to be executed by the ToolNode.
Then save returned values to the state for easy graph workflow.

```bash
# query
Do you need any help? any PDF doc or webpage to analyze? please search online about the latest restaurant opened in Azabu Juuban?

# Step 1
{'get_user_input': {'messages': ['please search online about the latest restaurant opened in Azabu Juuban?']}}

# Step 2
{'agent': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_m6a0', 'function': {'arguments': '{"query": "latest restaurant opened in Azabu Juuban"}', 'name': 'search'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 34, 'prompt_tokens': 3180, 'total_tokens': 3214, 'completion_time': 0.11005241, 'prompt_time': 0.984584956, 'queue_time': 0.0012828750000000166, 'total_time': 1.094637366}, 'model_name': 'llama3-groq-70b-8192-tool-use-preview', 'system_fingerprint': 'fp_ee4b521143', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-77b04c0e-c49c-437f-9257-31ccd94ce4e8-0', tool_calls=[{'name': 'search', 'args': {'query': 'latest restaurant opened in Azabu Juuban'}, 'id': 'call_m6a0', 'type': 'tool_call'}], usage_metadata={'input_tokens': 3180, 'output_tokens': 34, 'total_tokens': 3214})]}}

# Step 3
{'tool_search_node': {'messages': [ToolMessage(content='{"messages": ["Fukasaka is a sushi restaurant that opened in March 2024, led by a chef who trained at renowned establishments like the Michelin-starred Sushi Hashimoto in Shintomicho. ... Azabu-juban Station Directions from station 5 min. walk from Azabu-juban Station Exit 1 ... Aoyama is a vibrant district where you can experience the latest trends, unique ... A main dish of yum pam, phu penang curry, kai chao mu sap or muta kai tote will satisfy the palate. This eatery also offers a course menu with desserts. Blue Papaya Thailand is located near the Tokyo Metro Nanboku Line Azabu-Juban Station. It provides dinner and lunch services Monday through Saturday. The world-renowned restaurant, which features cuisine that is uncommon in Japan, opened its doors with a menu developed by Ichiro Ozaki, the owner-chef of Azabu Juban Ozaki, a popular sushi kappo restaurant located on the first floor of the same building, to offer new Japanese-style vegan dishes. The restaurant is a 5-minute walk from Azabu-Juban Station and offers a stylish atmosphere with 19 seats available. The opening hours are from Tuesday to Sunday, 17:30 to 23:30, with the last order for food at 22:00 and for drinks at 22:30. The restaurant is closed on certain days of the month, specifically the first and third Sundays. At this ... The restaurant \\"Principio\\" is located in Azabu-Juban, Minato-ku, Tokyo, and is accessible on foot from Azabu-Juban Station. The interior of the restaurant has a cozy and tranquil atmosphere, with 5 tables and a total of 10 seats available."]}', name='search', tool_call_id='call_m6a0')]}}

# intermediary print of the state messages
Message state:  [HumanMessage(content='message initialization', id='dba33398-1603-4a31-8650-5ba76542f1aa'), HumanMessage(content='please search online about the latest restaurant opened in Azabu Juuban?', id='5b451ab8-c184-462e-beba-391e82701732'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_m6a0', 'function': {'arguments': '{"query": "latest restaurant opened in Azabu Juuban"}', 'name': 'search'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 34, 'prompt_tokens': 3180, 'total_tokens': 3214, 'completion_time': 0.11005241, 'prompt_time': 0.984584956, 'queue_time': 0.0012828750000000166, 'total_time': 1.094637366}, 'model_name': 'llama3-groq-70b-8192-tool-use-preview', 'system_fingerprint': 'fp_ee4b521143', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-77b04c0e-c49c-437f-9257-31ccd94ce4e8-0', tool_calls=[{'name': 'search', 'args': {'query': 'latest restaurant opened in Azabu Juuban'}, 'id': 'call_m6a0', 'type': 'tool_call'}], usage_metadata={'input_tokens': 3180, 'output_tokens': 34, 'total_tokens': 3214}), ToolMessage(content='{"messages": ["Fukasaka is a sushi restaurant that opened in March 2024, led by a chef who trained at renowned establishments like the Michelin-starred Sushi Hashimoto in Shintomicho. ... Azabu-juban Station Directions from station 5 min. walk from Azabu-juban Station Exit 1 ... Aoyama is a vibrant district where you can experience the latest trends, unique ... A main dish of yum pam, phu penang curry, kai chao mu sap or muta kai tote will satisfy the palate. This eatery also offers a course menu with desserts. Blue Papaya Thailand is located near the Tokyo Metro Nanboku Line Azabu-Juban Station. It provides dinner and lunch services Monday through Saturday. The world-renowned restaurant, which features cuisine that is uncommon in Japan, opened its doors with a menu developed by Ichiro Ozaki, the owner-chef of Azabu Juban Ozaki, a popular sushi kappo restaurant located on the first floor of the same building, to offer new Japanese-style vegan dishes. The restaurant is a 5-minute walk from Azabu-Juban Station and offers a stylish atmosphere with 19 seats available. The opening hours are from Tuesday to Sunday, 17:30 to 23:30, with the last order for food at 22:00 and for drinks at 22:30. The restaurant is closed on certain days of the month, specifically the first and third Sundays. At this ... The restaurant \\"Principio\\" is located in Azabu-Juban, Minato-ku, Tokyo, and is accessible on foot from Azabu-Juban Station. The interior of the restaurant has a cozy and tranquil atmosphere, with 5 tables and a total of 10 seats available."]}', name='search', id='74eb112e-76a8-428a-be4a-37bb9aed685b', tool_call_id='call_m6a0')]

# Step 4
{'answer_user': {'messages': [{'role': 'ai', 'content': '{"messages": ["Fukasaka is a sushi restaurant that opened in March 2024, led by a chef who trained at renowned establishments like the Michelin-starred Sushi Hashimoto in Shintomicho. ... Azabu-juban Station Directions from station 5 min. walk from Azabu-juban Station Exit 1 ... Aoyama is a vibrant district where you can experience the latest trends, unique ... A main dish of yum pam, phu penang curry, kai chao mu sap or muta kai tote will satisfy the palate. This eatery also offers a course menu with desserts. Blue Papaya Thailand is located near the Tokyo Metro Nanboku Line Azabu-Juban Station. It provides dinner and lunch services Monday through Saturday. The world-renowned restaurant, which features cuisine that is uncommon in Japan, opened its doors with a menu developed by Ichiro Ozaki, the owner-chef of Azabu Juban Ozaki, a popular sushi kappo restaurant located on the first floor of the same building, to offer new Japanese-style vegan dishes. The restaurant is a 5-minute walk from Azabu-Juban Station and offers a stylish atmosphere with 19 seats available. The opening hours are from Tuesday to Sunday, 17:30 to 23:30, with the last order for food at 22:00 and for drinks at 22:30. The restaurant is closed on certain days of the month, specifically the first and third Sundays. At this ... The restaurant \\"Principio\\" is located in Azabu-Juban, Minato-ku, Tokyo, and is accessible on foot from Azabu-Juban Station. The interior of the restaurant has a cozy and tranquil atmosphere, with 5 tables and a total of 10 seats available."]}'}]}}

```

- Step 1 Output: The user's query ("please search online about the latest restaurant opened in Azabu Juuban?") is captured and added to the MessagesState.

- Step 2 Output: The agent node invokes the LLM with the query. The LLM identifies that it should use the search tool to handle this query, so it generates a tool call with the appropriate arguments (e.g., {'query': 'latest restaurant opened in Azabu Juuban'}). This step is crucial because it shows that the LLM successfully recognized that the search tool should be used and passed the correct query string to it.

- Step 3 Output: The tool_search_node executes the search function using the query provided by the LLM. The search tool performs the internet search and returns the results.

- Step 4 Output: The answer_user node takes the result of the internet search and prepares it for display to the user.


eg. of tools:
```python
groq_llm_mixtral_7b = ChatGroq(
    temperature=float(os.getenv("GROQ_TEMPERATURE")),
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-groq-70b-8192-tool-use-preview",
    max_tokens=int(os.getenv("GROQ_MAX_TOKEN")),
)

# some other tools defined
tool
def get_proverb(query: str, state: MessagesState = MessagesState()):
  """Will transform user query into a funny proverb"""
  system_message = SystemMessage(content="You are an expert in creating funny short proverbs from any query.")
  human_message = HumanMessage(content=f"Create a funny proverb please: {query}")
  response = groq_llm_mixtral_7b.invoke([system_message, human_message])
  return {"messages": [response]}

@tool
def find_link_story(query: str, state: MessagesState = MessagesState()):
  """Will find a link between user query and the planet Mars"""
  system_message = SystemMessage(content="You are an expert in finding links between Mars planet and any query.")
  human_message = HumanMessage(content=f"{query}")
  response = groq_llm_mixtral_7b.invoke([system_message, human_message])
  return {"messages": [response]}


# Define the search tool
internet_search_tool = DuckDuckGoSearchRun()
tool_internet = Tool(
    name="duckduckgo_search",
    description="Search DuckDuckGO for recent results.",
    func=internet_search_tool.run,
)

@tool
def search(query: str, state: MessagesState = MessagesState()):
    """Call to surf the web."""
    search_results = internet_search_tool.run(query)
    return {"messages": [search_results]}

tool_search_node = ToolNode([get_proverb, find_link_story])

# Bind the tool to the LLM
llm_with_internet_search_tool = groq_llm_mixtral_7b.bind_tools([get_proverb, find_link_story])
```

# Function to have graph output in `.stream()` mode beautified
```python
# function to beautify output for an ease of human creditizens reading
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

# Example of how to use this function
count = 0
for step in app.stream(
    {"messages": [HumanMessage(content="message initialization")]},
    config={"configurable": {"thread_id": 42}}):
    count += 1
    if "messages" in step:
        print(f"Step {count}: {beautify_output(step['messages'][-1].content)}")
    else:
        print(f"Step {count}: {beautify_output(step)}")
```


# Hypothetical Workflow Scenario
Here we will create hypothetical desired workflow for the graph
```python
"""
--- START 
    0--- ask user query input
    1--- Analyze User Query To Get the Webpage or Document or just text (to know if we need to perform webpage parsing or pdf doc parsing) 
         --- save to state 'webpage' or 'pdf' or 'text'
        1.1--- If/Else conditional edge 'db_doc_check':
           --- output = 'webpage/pdf title IN db'
           1.1.1--- If webpage title or pdf title in db --- If/Else conditional edge 'cache/vectordb query answer check'
                                                        1.1.1.1--- If answer in cache/vectordb --- answer user query --- END
                                                        1.1.1.2--- Else Internet Search about subject
                                                                   --- Embed query/internet_answer and cache it
                                                                       --- answer user query with internet_search result
                                                                           --- END
           --- output = 'webpage/pdf title NOT IN db'
           1.1.2--- Else --- If/Else conditional edge 'webpage or pdf processing':
                         --- output = 'url'
                         1.1.2.1--- If webpage --- parse webpage
                                                   --- store to database
                                                       --- create chunks from database rows
                                                           --- embed chunks
                                                               --- save to state 'webpage embedded'
                                                                   --- go to 1.1.1.1 flow which will search in cache and won't found it and make a vector search and answer-> END
                         --- output = 'pdf'
                         1.1.2.2--- If pdf     --- parse pdf
                                                   --- store to database
                                                       --- create chunks from database rows
                                                           --- embed chunks
                                                               --- save to state 'pdf embedded'
                                                                   --- go to 1.1.1.1 flow which will search in cache and won't found it and make a vector search and answer-> END
                         --- output = 'text'
                         1.1.2.3--- If text    --- perform internet search to get response of `text only` query
                                                   --- format internet result and answer -> END

    2--- Analyze User Query to extract the question from it and rephrase the question to optimize llm information research/retrieval --- save to state 'query/question'
         3--- Retrieve answer from Query in embedded collection
              --- save to state answer_retrieved
                  4--- Internet Search user query
                       --- save to state internet search_answer
                           5--- answer to user in markdown report format using both states answer_retrieved/search_answer
"""
```

# Next
- keep in mind the pdf parser that might need to be refactored to chunk using same process as the webpage parser one.
- incorporate the internet tool in the graph - OK
- graph accepts x3 states but we will only use here `MessagesStates` already present in LangGraph library. We could create other states using `NewState(TypeDict)` but we will keep it simple in this project - MAYBE BUT NOT NOW
- delete `parquet` files (`df_final` compacted) after they have been processed and saved to db to save space and not have file building up and taking space - OK
- reset the cache to zero as well so that we have the option to delete everything for some future task that doesn't need the data to persist forever in the DB.
- fix redis that doesn't save anything as value for key. find in the code where is the issue


# Test Internet Search Implementation and markdown report generation

Initial query: what is the first result when we search for 'sakura flower' though the internet?

**The Beauty of Sakura: A Report on Japanese Cherry Blossom Trees and Their Culture**
=====================================================

### Introduction
Sakura, the Japanese word for cherry blossom, is a flower that has captivated the hearts of many around the world. With its delicate petals and vibrant colors, it's no wonder why cherry blossom trees have become a symbol of Japan's rich culture and history. In this report, we will delve into the different varieties of cherry blossom trees, their lifespan, and the cultural significance of hanami, or cherry blossom viewing.


### Types of Cherry Blossom Trees

There are many different varieties of cherry blossom trees, with flowers that range in color from white to deep pink. Most sakura trees are small to medium-sized, with a lifespan of 15-30 years, though some can live longer. Their bark is smooth and reddish-brown, and their green leaves turn vibrant shades of yellow, red, or crimson in autumn.\n\n### Hanami Culture

Hanami, which translates to viewing flowers, is a centuries-old tradition in Japan where people gather under the Sakura tree for picnics and parties. This tradition is a testament to the unifying power of nature and the beauty of cherry blossoms. In Japan, hanami is a time for people to come together and appreciate the fleeting beauty of the cherry blossoms.

### History and Culture of Sakura\n\nSakura is not only a symbol of Japan's natural beauty but also a representation of its rich culture and history. The National Cherry Blossom Festival in Washington, D.C. is a testament to the unifying power of cherry blossoms, and it has become a popular event around the world. Cherry blossom festivals have become increasingly popular outside of Japan, with many countries hosting their own sakura festivals.

### Conclusion
In conclusion, the beauty of sakura is not just about the flower itself, but about the culture and history that surrounds it. Whether you're in Japan or around the world, cherry blossom trees are a symbol of the fleeting nature of life and the importance of appreciating the beauty that surrounds us. So, take a moment to stop and smell the sakura!

### Advice

* If you're planning to visit Japan during cherry blossom season, make sure to book your accommodations and transportation in advance, as it's a popular time to visit.
* Take a moment to appreciate the beauty of the cherry blossoms, and don't forget to take plenty of photos to capture the moment.
* If you're unable to visit Japan, consider attending a cherry blossom festival in your local area, or planting your own cherry blossom tree to enjoy the beauty of the flowers in your own backyard.



# Binding Tools To LLM and Structured Output

Create a class using `BaseModel` and define variables, types of those variables and use also the `Field` to add descriptions.

```python
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.utils.function_calling import convert_pydantic_to_openai_function
from langchain_core.pydantic_v1 import BaseModel, Field

class InternetSearchStructuredOutput(BaseModel):
    """Internet research Output Response"""
    internet_search_answer: str = Field(description="The result of the internet research in markdon format about user query when an answer has been found.")
    source: str = Field(description="The source were the answer has been fetched from in markdown format, it can be a document name, or an url")
    error: str = Field(description="An error messages when the search engine didn't return any response or no valid answer.")

functions = [format_tool_to_openai_function(tool) for tool in tools]
functions.append(convert_pydantic_to-openai_function(InternetSearchStructuredOutput))
model.bind_functions(function)
result = model.invoke(
  {
    "messages": [HumanMessage(content="Search in internet about Chikarahouses.com, I need some more information")]
  }
)
```

but didn't worked!

the next example didn't work also but we have a kind of a boilerplate the the right output format but nothing if filled. Just empty
```python
from langchain.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage

# Define structured output schema using Pydantic's BaseModel
class InternetSearchStructuredOutput(BaseModel):
    internet_search_answer: str = Field(description="The result of the search in markdown format.")
    source: str = Field(description="The source of the information.")
    error: str = Field(description="Error if any occurred during the search.")

# Initialize the Groq model and set it to return structured output
model = ChatGroq(model="mixtral-7b").with_structured_output(InternetSearchStructuredOutput)

# Create a query asking for structured output directly
query = HumanMessage(content="Provide information about the next satellite launch in the following structure: \
'internet_search_answer', 'source', and 'error'.")

# Invoke the model and get the structured output
try:
    response = model.invoke([query])
    print("Structured Output:", response)
except Exception as e:
    print(f"An error occurred: {e}")
```
Outputs:

```python
Structured Output: internet_search_answer='' source='' error=''
```

# Structured Output Using Only Prompting Which Has Worked Fine
Trying without the structured_output s***, just prompting as I did before and it worked fine without the complexity of using something that is documented in 1000 different ways and not one working, even chatgpt is tired!
```python
# Initialize the Groq model and set it to return structured output
model = groq_llm_mixtral_7b

# Create a query asking for structured output directly
query_system = SystemMessage(content="Help user by answering always with 'Advice' (your advice about the subject of the question and how user should tackle it), 'Answer' (The answer in French to the user query) , and 'error' (The status when you can't answer), put it in a dictionary between mardown tags like ```markdown{Advice: your advice ,Answer: the answer to the user query in French,error:if you can't answer}```.")
query_human = HumanMessage(content="What is the story behind th epythagorus theorem, it seems like they have lied and it has been stollen knowledge from the Egyption where pythagore have studied with black people and came back to greece and said that it is from his own knwoledge, today french people are teaching that it is greek when it was black egyptian knowledge. this to keep the white supremacy idea which is a bad ideology")

# Invoke the model and get the structured output
try:
    response = model.invoke([query_system, query_human])
    print("Structured Output:", response)
except Exception as e:
    print(f"An error occurred: {e}")
```

That have worked, therefore, we just need to prompt well no need al those crazy stuff and it is very consistant with this mini llm 7b mistral therefore the bigger ones paid one will work fine, this is our assumption and that is why we are doing it all using small llms
Outputs:
```python
Structured Output: content="```markdown{\nAdvice: It's important to approach history with an open mind and a critical eye. While it's true that Pythagoras studied in Egypt, there is no concrete evidence to support the claim that he stole the theorem from Egyptian or black mathematicians. The development of mathematical knowledge is often a collaborative process, with ideas and concepts building upon each other over time. It's crucial to give credit where it's due, but it's also important to avoid oversimplifying or distorting historical narratives to fit a particular ideology.\nAnswer: Le théorème de Pythagore est un théorème de géométrie dans un triangle rectangle, qui affirme que le carré de la longueur de l'hypoténuse (le côté opposé à l'angle droit) est égal à la somme des carrés des longueurs des deux autres côtés. Ce théorème est souvent attribué à Pythagore, un mathématicien grec du VIe siècle avant J.-C., mais il est possible que des versions de ce théorème aient été connues et utilisées par d'autres cultures avant lui.\nError: There is no error in your question, but it's important to note that the development of mathematical knowledge is a complex and nuanced process, and attributing specific discoveries to particular individuals or cultures can be challenging and sometimes contentious.\n}```" response_metadata={'token_usage': {'completion_tokens': 319, 'prompt_tokens': 202, 'total_tokens': 521, 'completion_time': 0.520485911, 'prompt_time': 0.011388187, 'queue_time': 0.003033524000000001, 'total_time': 0.531874098}, 'model_name': 'mixtral-8x7b-32768', 'system_fingerprint': 'fp_c5f20b5bb1', 'finish_reason': 'stop', 'logprobs': None} id='run-19184a14-1ff9-404c-9721-687fd9f81320-0' usage_metadata={'input_tokens': 202, 'output_tokens': 319, 'total_tokens': 521}
```
- adapted and transformed as funciton which can be used as `node` function or as a `@tool`, need to create the prompts
```python
def structured_output_report(state: MessagesState):
  query_system = SystemMessage(content="Help user by answering always with 'Advice' (your advice about the subject of the question and how user should tackle it), 'Answer' (The answer in French to the user query) , and 'error' (The status when you can't answer), put it in a dictionary between mardown tags like ```markdown{Advice: your advice ,Answer: the answer to the user query in French,error:if you can't answer}```.")
  query_human = HumanMessage(content="What is the story behind th epythagorus theorem, it seems like they have lied and it has been stollen knowledge from the Egyption where pythagore have studied with black people and came back to greece and said that it is from his own knwoledge, today french people are teaching that it is greek when it was black egyptian knowledge. this to keep the white supremacy idea which is a bad ideology")

  # Invoke the model and get the structured output
  try:
    response = groq_llm_llama3_8b.invoke([query_system, query_human])
    print(f"structured_output:{response}")
  except Exception as e:
    return f"An error occurred: {e}"

  return {"messages": [response]}
```

```python
@tool
def structured_output_report(prompt: str = structured_outpout_report_prompt, state: MessagesState):
  """
  This function will structure the output in order to have the format of answer required for quality deliverability
  
  Parameters:
  prompt str: the system prompt that will instruct in how the output of the answer should be structured
  
  Returns:
  messages dict: a dictionary that will update the state with the reponse from the llm having structure the message
  """
  messages = state["messages"]
  last_message = messages[-1].content

  query_system = prompt
  query_human = HumanMessage(content=last_message))

  # Invoke the model and get the structured output
  try:
    response = groq_llm_llama3_8b.invoke([query_system, query_human])
    print(f"structured_output:{response}")
  except Exception as e:
    return f"An error occurred: {e}"

  return {"messages": [response]}

tool_search_node = ToolNode([structured_output_report])

# tool_choice="any" only supported for the moment for MistraiAI, Openai, Groq, FireworksAI. for grow should ane `None` or `auto`
llm_with_structured_output_report_tool = groq_llm_mixtral_7b.bind_tools([structured_output_report])
```

# Structure Outputs all different ways and feedback
- Some of the important import needed:
```python
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.output_parsers.openai_tools import JsonOutputToolsParser
from langchain.output_parsers.openai_functions import PydanticOutputFunctionsParser
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
```
- Define your desired data structure.
```python
class AddCount(BaseModel):
    """Will always add 10 to any number"""
    initialnumber: List[int] = Field(default= [], description="Initial numbers present in the prompt.")
    calculation: str = Field(default="", description="Details of the calculation that have create the 'result' number.")
    result: int = Field(default= 0, description="Get the sum of the all the numbers. Then add 10 to the sum which will be final result. ")
    error: str = Field(default= "", description="An error message when no number has been found in the prompt.")

class CreateBulletPoints(BaseModel):
    """Creates answers in the form of 3 bullet points."""
    bulletpoints: str = Field(default="", description="Answer creating three bullet points that are very pertinent.")
    typeofquery: str = Field(default="", description="tell if the query is just a sentence with the word 'sentence' or a question with the word 'question'.")
```

- **First way**
```python
# Set up a parser + inject instructions into the prompt template.
parser = PydanticOutputParser(pydantic_object=CreateBulletPoints)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# And a query intended to prompt a language model to populate the data structure.
prompt_and_model = prompt | groq_llm_mixtral_7b | parser
response = prompt_and_model.invoke({"query": "Tokyo is number 1 city to have more than 20 million habitants. Read description of tool and provide right answer for each fields please."})
print("First way: ", response)

Outputs condensed bullet points but has respected the schema and in case and in case of calculaiton does it right:
First way:  bulletpoints='• Tokyo has the largest population of any city, with over 20 million inhabitants.\n• It is a major international financial and cultural center.\n• Tokyo is known for its efficient public transportation system and high standard of living.' typeofquery='sentence'
First way:  initialnumber=[1, 20000000] calculation='Summed the initial number and added 10' result=2000010 error=''

```

- **Second way**
```python
parser = PydanticOutputFunctionsParser(pydantic_schema=CreateBulletPoints)

openai_functions = [convert_to_openai_function(CreateBulletPoints)]
chain = prompt | groq_llm_mixtral_7b.bind(functions=openai_functions) | parser

print("Second way: ", chain.invoke({"query": "Tokyo is number 1 city to have more than 20 million habitants. Read description of tool and provide right answer for each fields please."}))

Outputs no bullet points and does not the calculation at all, miss interprets:
Second way:  bulletpoints='Tokyo is the largest city in Japan and one of the 14 cities in the world with a population of more than 20 million people. It is a major international financial and cultural center. Tokyo has a population of over 37 million people, making it the most populous metropolitan area in the world.' typeofquery='sentence'
Second way:  initialnumber=[20000000, 1] calculation='Added initialnumber values to get sum' result=0 error=''

```

- **Third way**
```python
# for AddCount
#query_system = SystemMessage(content="Help user by answering always with 'initialnumber' (Initial numbers present in the prompt.), 'calculation' (Details of the calculation that have create the 'result' number), 'result' (Get the sum of the all the numbers. Then add 10 to the sum which will be final result) and 'error' {An error message when no number has been found in the prompt.), put it in a dictionary between mardown tags like ```markdown{initialnumber: list of Initial numbers present in the prompt. ,calculation: Details of the calculation that have create the 'result' number, result: Get the sum of the all the numbers. Then add 10 to the sum which will be final result, error: An error message when no number has been found in the prompt.}```.")
# for CreateBulletPoints
query_system = SystemMessage(content="Help user by answering always with 'bulletpoints' (Answer creating three bullet points that are very pertinent.), 'typeofquery' (tell if the query is just a sentence with the word 'sentence' or a question with the word 'question'.), put it in a dictionary between mardown tags like ```markdown{bulletpoints: Answer creating three bullet points that are very pertinent, typeofquery: tell if the query is just a sentence with the word 'sentence' or a question with the word 'question'.}```.")
query_human = HumanMessage(content="Tokyo is number 1 city to have more than 20 million habitants. Read description of tool and provide right answer for each fields please.")

# Invoke the model and get the structured output
response = groq_llm_mixtral_7b.invoke([query_system, query_human])
print(f"Third way: {response.content.split('```')[1].strip('markdown').strip()}")

Outputs the best structure but sometimes missing comma in dictionary str and also no calculation made in my second test so not sure. Here the example has worked but later in the day it didn't. and follows instruction well with nice markdown formatting and in case of call calculations sees `20 million` as `20` and not `20_000_000`:
Third way: {
bulletpoints: 
- Tokyo is the most populous city in the world with over 20 million inhabitants.
- It is the capital of Japan and is known for its bustling streets, rich culture, and technological advancements.
- Tokyo is a popular tourist destination, attracting millions of visitors each year with its many attractions, including the Tokyo Tower, the Tokyo Skytree, and the historic Asakusa district.
typeofquery: sentence
}
Third way: {
initialnumber: [20, 1], 
calculation: The sum of the two initial numbers is 21. Then add 10 to the sum which gives a final result of 31.
result: 31, 
error: There are initial numbers present in the prompt, so no error message is needed.
}
```

- **Fourth way**
```python
model = groq_llm_mixtral_7b.bind_tools([CreateBulletPoints])
prompt = ChatPromptTemplate.from_messages(
    [("system", "You are helpful assistant"), ("user", "{input}")]
)
parser = JsonOutputToolsParser()
chain = prompt | model | parser
response = chain.invoke({"input": "Tokyo is number 1 city to have more than 20 million habitants. Read description of tool and provide right answer for each fields please."})
print(f"Fourth way: {response}")

Outputs no bullet points and in case of calculation doesn't do it right:
Fourth way: [{'args': {'bulletpoints': 'Tokyo is the number 1 city with more than 20 million inhabitants, it is known for its rich culture and history, and it has a vibrant economy.', 'typeofquery': 'statement'}, 'type': 'CreateBulletPoints'}]
Fourth way: [{'args': {'calculation': 'Adding 10 to the population of Tokyo (20 million) to find the new total.', 'initialnumber': [20000000], 'result': 0}, 'type': 'AddCount'}]

```

- **Fifth way**
```python
# for CreateBulletPoints
response_schemas = [
    ResponseSchema(name="bulletpoints", description="Answer creating three bullet points that are very pertinent."),
    ResponseSchema(name="typeofquery", description="tell if the query is just a sentence with the word 'sentence' or a question with the word 'question'."),
]
# for AddCount
response_schemas2 = [
    ResponseSchema(name="initialnumber", description="Initial numbers present in the prompt.")
    ResponseSchema(name="calculation", description="Details of the calculation that have create the 'result' number.")
    ResponseSchema(name="result", description="Get the sum of the all the numbers. Then add 10 to the sum which will be final result.")
    ResponseSchema(name="error", description="An error message when no number has been found in the prompt.")
]
parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = parser.get_format_instructions()
prompt = PromptTemplate(
    template="answer the user query.\n{format_instructions}\n{query}",
    input_variables=["query"],
    partial_variables={"format_instructions": format_instructions},
)
model = groq_llm_mixtral_7b
chain = prompt | model | parser
response = chain.invoke({"query": "Tokyo is number 1 city to have more than 20 million habitants. Read description of tool and provide right answer for each fields please."})
print(f"Fith way: {response}")

Outputs good structure like `First way` but in a Dictionary so even better for value extraction
Fith way: {'bulletpoints': '• Tokyo is the city with the largest population of over 20 million people.\n• The query asks for a description or information about a tool, but it seems there is a misunderstanding as the topic is about a city.\n• The response should provide relevant and factual information about Tokyo, such as its landmarks, culture, or economy.', 'typeofquery': 'sentence'}

```

- **Feedback on output parsers**
Best is the **First way** for all and the **Fifth way** for text only but not for calculation. **Third way** maybe with better optimal prompting.
- For text: I will be using probably the **Fifth way** as it is more consistent and the output is a `dict` and we can use a function that we already have to transform it into a dict if it is not as th estructure is perfect and fetch what we need.
- For calculation or more complexity: I will be using probably the **First way** as it has not failed any kind of jobs so may even use it for everything and keep the **Fifth way** as backup alternative but knowing its limitations.


# Initial Plan Archive
```python
###### CREATE WORKFLOW TO PLAN NODE AND EDGES NEEDS #######
## 7- CREATE WORKFLOW
"""
Here we will create hypothetical desired workflow for the graph

--- START 
    0--- ask user query input
    1--- Analyze User Query To Get the Webpage or Document or just text (to know if we need to perform webpage parsing or pdf doc parsing) 
         --- save to state 'webpage' or 'pdf' or 'text'
        1.1--- If/Else conditional edge 'db_doc_check':
           --- output = 'webpage/pdf title IN db'
           1.1.1--- If webpage title or pdf title in db --- If/Else conditional edge 'cache/vectordb query answer check'
                                                        1.1.1.1--- If answer in cache/vectordb --- answer user query --- END
                                                        1.1.1.2--- Else Internet Search about subject
                                                                   --- Embed query/internet_answer and cache it
                                                                       --- answer user query with internet_search result
                                                                           --- END
           --- output = 'webpage/pdf title NOT IN db'
           1.1.2--- Else --- If/Else conditional edge 'webpage or pdf processing':
                         --- output = 'url'
                         1.1.2.1--- If webpage --- parse webpage
                                                   --- store to database
                                                       --- create chunks from database rows
                                                           --- embed chunks
                                                               --- save to state 'webpage embedded'
                                                                   --- go to 1.1.1.1 flow which will search in cache and won't found it and make a vector search and answer-> END
                         --- output = 'pdf'
                         1.1.2.2--- If pdf     --- parse pdf
                                                   --- store to database
                                                       --- create chunks from database rows
                                                           --- embed chunks
                                                               --- save to state 'pdf embedded'
                                                                   --- go to 1.1.1.1 flow which will search in cache and won't found it and make a vector search and answer-> END
                         --- output = 'text'
                         1.1.2.3--- If text    --- perform internet search to get response of `text only` query
                                                   --- format internet result and answer -> END

    2--- Analyze User Query to extract the question from it and rephrase the question to optimize llm information research/retrieval --- save to state 'query/question'
         3--- Retrieve answer from Query in embedded collection
              --- save to state answer_retrieved
                  4--- Internet Search user query
                       --- save to state internet search_answer
                           5--- answer to user in markdown report format using both states answer_retrieved/search_answer
"""

## 8- CREATE NODES, EDGES AND CONDITIONAL EDGES

"""
creditizens_doc_report_flow = StateGraph(MessagesState)
creditizens_doc_report_flow.set_entry_point("<node_name>")
creditizens_doc_report_flow.add_node("<node_name>", <function_associated_to_node_action>)
# condition adge = conditional route from one node to a function output which will determine next node
creditizens_doc_report_flow.add_conditional_edges(
    # From which node
    "<node_name>",
    # function that will determine which node is called next
    <function_with_condition>,
    # function called output will determine next node
    {
      "output1": "<node_name>", # if output == "output1" go to specific node
      "output2": "<node_name>",  # if output == "output2" go to specific node
    },
)
# edge are route so from node to another
creditizens_doc_report_flow.add_edge("<node_name1>", "<node_name2>")
# checkpointer for memory of graph and compile the graph
checkpointer = MemorySaver()
creditizens_graph_flow_app = workflow.compile(checkpointer=checkpointer)
# inject user query in the creditizens_doc_report_flow: "can maybe here think of an app that ask user input to collect it and use it to launch the creditizens_graph_flow_app"
user_query_to_answer_using_creditizens_graph_flow = creditizens_graph_flow_app.invoke(
    {"messages": [HumanMessage(content="what is the weather in sf")]},
    config={"configurable": {"thread_id": 42}}
)
user_query_to_answer_using_creditizens_graph_flow["messages"][-1].content

________________________________________________________________________________________
creditizens_doc_report_flow = StateGraph(MessagesState)

Nodes:
creditizens_doc_report_flow.add_node("ask_user_input", <function_associated_to_node_action>)
creditizens_doc_report_flow.add_node("webpage_or_pdf_embedding_or_vector_search", query_analyzer_module.detect_content_type)
creditizens_doc_report_flow.add_node("user_query_analyzer_type_of_doc_extraction", <function_associated_to_node_action>)
creditizens_doc_report_flow.add_node("user_query_extract_question_and_rephrase", <function_associated_to_node_action>)
creditizens_doc_report_flow.add_node("check_if_document_in_db", doc_db_search)
creditizens_doc_report_flow.add_node("cache_vectordb_search", query_matching.handle_query_by_calling_cache_then_vectordb_if_fail)
creditizens_doc_report_flow.add_node("answer_user_query", <function_associated_to_node_action>) # this will END
creditizens_doc_report_flow.add_node("internet_search", <function_associated_to_node_action>)
# can add also save query and internet search to DB and put in title a llm generated title for this query/answer, and have doc_name being =internet search
creditizens_doc_report_flow.add_node("save_to_db", <function_associated_to_node_action>)
creditizens_doc_report_flow.add_node("embed", <function_associated_to_node_action>)
creditizens_doc_report_flow.add_node("webpage_parser", webpage_parser.get_df_final)
creditizens_doc_report_flow.add_node("pdf_parser", pdf_parser.get_final_df)
creditizens_doc_report_flow.add_node("create_chunk", <func>)
creditizens_doc_report_flow.add_node("markdown_report_creation", <func>)

0              creditizens_doc_report_flow.set_entry_point("ask_user_input")
1              creditizens_doc_report_flow.add_edge("user_query_analyzer_type_of_doc_extraction", "webpage_or_pdf_embedding_or_vector_search")
               creditizens_doc_report_flow.add_conditional_edges(
                 "webpage_or_pdf_embedding_or_vector_search",
                 query_analyzer_module.detect_content_type,
                 {
                   "url": "check_if_document_in_db", 
                   "pdf": "check_if_document_in_db",
                   "text": "cache_vectordb_search", # this will go to the flow 2
                 },
               )
1.1            creditizens_doc_report_flow.add_edge("webpage_or_pdf_embedding_or_vector_search", "check_if_document_in_db")
               creditizens_doc_report_flow.add_conditional_edges(
                 # From which node
                 "check_if_document_in_db",
                 # function that will determine which node is called next
                 doc_db_search,
                 # function called output will determine next node
                 {
                   "title_found": "cache_vectordb_search", # we go search in cache/vectordb
                   "title_not_found": "webpage_or_pdf_embedding_or_vector_search",
                 },
               )
1.1.1          creditizens_doc_report_flow.add_edge("check_if_document_in_db", "cache_vectordb_search")
               creditizens_doc_report_flow.add_conditional_edges(
                 "cache_vectordb_search",
                 query_matching.handle_query_by_calling_cache_then_vectordb_if_fail,
                 {
                   "answer_found": "answer_user_query", # this will answer and END
                   
                   "answer_not_found": "internet_search", # here we will start a long process of nodes and edges
                 },
               )
1.1.1.1        creditizens_doc_report_flow.add_edge("cache_vectordb_search", "answer_user_query") # this will END

1.1.1.2        creditizens_doc_report_flow.add_edge("cache_vectordb_search", "internet_search")
               # save to db here query and internet search result, llm generated title for this query/answer, and have doc_name being =internet search
               creditizens_doc_report_flow.add_edge("internet_search", "save_to_db")
               creditizens_doc_report_flow.add_edge("save_to_db", "embed")
               creditizens_doc_report_flow.add_edge("embed", "answer_user_query") # this will END
               
1.1.2          creditizens_doc_report_flow.add_edge("check_if_document_in_db", "webpage_or_pdf_embedding_or_vector_search")
               creditizens_doc_report_flow.add_conditional_edges(
                 "webpage_or_pdf_embedding_or_vector_search",
                 query_analyzer_module.detect_content_type,
                 {
                   "url": "webpage_parser",
                   "pdf": "pdf_parser",
                   "text": "user_query_extract_question_and_rephrase", # this will go to the flow 2
                 },
               )
1.1.2.1        creditizens_doc_report_flow.add_edge("webpage_or_pdf_embedding_or_vector_search", "webpage_parser")
               creditizens_doc_report_flow.add_edge("webpage_parser", "save_to_db")
               creditizens_doc_report_flow.add_edge("save_to_db", "create_chunks")
               creditizens_doc_report_flow.add_edge("create_chunks", "embed")
               creditizens_doc_report_flow.add_edge("embed", cache_vectordb_search)
               creditizens_doc_report_flow.add_edge("cache_vectordb_search", "answer_user_query") # this will END

1.1.2.2        creditizens_doc_report_flow.add_edge("webpage_or_pdf_embedding_or_vector_search", "pdf_parser")
               creditizens_doc_report_flow.add_edge("pdf_parser", "save_to_db")
               creditizens_doc_report_flow.add_edge("save_to_db", "create_chunks")
               creditizens_doc_report_flow.add_edge("create_chunks", "embed")
               creditizens_doc_report_flow.add_edge("embed", cache_vectordb_search)
               creditizens_doc_report_flow.add_edge("cache_vectordb_search", "answer_user_query") # this will END

2              creditizens_doc_report_flow.add_edge("user_query_extract_question_and_rephrase", "webpage_or_pdf_embedding_or_vector_search")
               creditizens_doc_report_flow.add_edge("webpage_or_pdf_embedding_or_vector_search", "cache_vectordb_search")
               creditizens_doc_report_flow.add_edge("cache_vectordb_search", "internet_search")
3              creditizens_doc_report_flow.add_edge("internet_search", "markdown_report_creation")
4              creditizens_doc_report_flow.add_edge("markdown_report_creation", END)

For all conditional edge to make it simpler for some, save to state the value of the function return and get just the state value in the conditional edge
"""
```

# Special Function
- Function to call llm noramlly or chat (AI/System/Human) with our templates: `we don't use it but it is good to have it one the side`
```python
# Functions to call llm using templates, nofrmal llm call amd chat llm call (human, system, ai)
"""
1- import the right prompt template normal question/response or chat special system/human/aione
2- use the template and the function argument to create the prompt
3- use make_normal_or_chat_prompt_chain_call(llm_client, prompt_input_variables_part: Dict, prompt_template_part: Optional[Dict], chat_prompt_template: Optional[Dict])
4- use internet search tool if nothing is found or go to the internet search node
"""
# create prompts format, we can pass in as many kwargs as wanted it will unfold and place it properly
# we have two schema one for normal question/answer prompts and another for chat_prompts system/human/ai
def call_chain(model: ChatGroq, prompt: PromptTemplate, prompt_input_vars: Optional[Dict], prompt_chat_input_vars: Optional[Dict]):
  # only to concatenate all input vars for the system/human/ai chat
  chat_input_vars_dict = {}
  for k, v in prompt_chat_input_vars.items():
    for key, value in v.items():
      chat_input_vars_dict[key] = value
  print("Chat input vars: ",  chat_input_vars_dict)

  # default question/answer
  if prompt and prompt_input_vars:
    print("Input variables found: ", prompt_input_vars)
    chain = ( prompt | model )
    response = chain.invoke(prompt_input_vars)
    print("Response: ", response, type(response))
    return response.content.split("```")[1].strip("markdown").strip()
    
  # special for chat system/human/ai
  elif prompt and prompt_chat_input_vars:
    print("Chat input variables found: ", prompt_chat_input_vars)
    chain = ( prompt | model )
    response = chain.invoke(chat_input_vars_dict)
    print("Response: ", response, type(response))
    return response.content.split("```")[1].strip("markdown").strip()

  print("Chat input variables NOT found or missing prompt!")
  chain = ( prompt | model )
  response = chain.invoke(input={})
  print("Response: ", response, type(response))
  return response.content.split("```")[1].strip("markdown").strip()

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

# just to test
from prompts.prompts import test_prompt_tokyo, test_prompt_siberia, test_prompt_monaco, test_prompt_dakar

# prompts
"""
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
"""

# test prints
print("SIBERIA: \n", make_normal_or_chat_prompt_chain_call(groq_llm_llama3_8b, test_prompt_siberia["input_variables"], test_prompt_siberia["template"], {}))

print("TOKYO: \n", make_normal_or_chat_prompt_chain_call(groq_llm_llama3_8b, test_prompt_tokyo["input_variables"], test_prompt_tokyo["template"], {}))

print("DAKAR: \n", make_normal_or_chat_prompt_chain_call(groq_llm_llama3_8b, {}, {}, test_prompt_dakar))

print("MONACO: \n", make_normal_or_chat_prompt_chain_call(groq_llm_llama3_8b, {}, {}, test_prompt_monaco))
```

# example of object saved to cache when prior to it have been retrieved in vector db
- we got here key formatted: `hash...`
- we got the value formatted: `Dict[embeddings: List Vector, response: List[Dict[UUID:..., socre:..., content:..., row_data: Dict[id:..., doc_name:..., tile:..., content:...]]]]`

```bash
2) "1672c73c05c52c59dfe7976808f01949e8a3f599774abe32e820ba64de3c759f"
127.0.0.1:6379> egt "1672c73c05c52c59dfe7976808f01949e8a3f599774abe32e820ba64de3c759f"
(error) ERR unknown command 'egt', with args beginning with: '1672c73c05c52c59dfe7976808f01949e8a3f599774abe32e820ba64de3c759f' 
127.0.0.1:6379> get "1672c73c05c52c59dfe7976808f01949e8a3f599774abe32e820ba64de3c759f"
"[{\"UUID\": \"cfcad20f-a015-41e5-aa7e-a1b0f3efc642\", \"score\": 0.5771155852749257, \"content\": \"Perception of temperature is determined by the firing rates of cold and warm receptors. When touching an object between 5-30\\u00b0C, cold receptor firing increases and warm receptor firing decreases, indicating cold. When touching an object between 30-45\\u00b0C, warm receptor firing increases and cold receptor firing decreases, indicating warmth. If an object is hot enough to cause pain, nociceptors are activated, signaling pain and further indicating potential injury.\", \"row_data\": {\"id\": \"cfcad20f-a015-41e5-aa7e-a1b0f3efc642\", \"doc_name\": \"docs/feel_temperature.pdf\", \"title\": \"'Neural Responses: Feeling Hot and Cold Temperatures'\", \"content\": \"Perception of temperature is determined by the firing rates of cold and warm receptors. When touching an object between 5-30\\u00b0C, cold receptor firing increases and warm receptor firing decreases, indicating cold. When touching an object between 30-45\\u00b0C, warm receptor firing increases and cold receptor firing decreases, indicating warmth. If an object is hot enough to cause pain, nociceptors are activated, signaling pain and further indicating potential injury.\"}}, {\"UUID\": \"e14be89d-e330-4dcf-b401-b251dbcd3c26\", \"score\": 0.5771155852749257, \"content\": \"To understand FNE thermoreceptors, you need to know about neurotransmission. The brain's signal strength relies on a neuron's firing frequency. We interact with objects of varying temperatures, but thermoreceptors typically detect room temperature, maintaining normal firing rates. Cold or hot stimuli alter the firing rate of corresponding thermoreceptors.\", \"row_data\": {\"id\": \"e14be89d-e330-4dcf-b401-b251dbcd3c26\", \"doc_name\": \"docs/feel_temperature.pdf\", \"title\": \"'Frequency of Firing in Cold Thermoreceptors'\", \"content\": \"To understand FNE thermoreceptors, you need to know about neurotransmission. The brain's signal strength relies on a neuron's firing frequency. We interact with objects of varying temperatures, but thermoreceptors typically detect room temperature, maintaining normal firing rates. Cold or hot stimuli alter the firing rate of corresponding thermoreceptors.\"}}, {\"UUID\": \"cfcad20f-a015-41e5-aa7e-a1b0f3efc642\", \"score\": 0.5771155852749257, \"content\": \"Perception of temperature is determined by the firing rates of cold and warm receptors. When touching an object between 5-30\\u00b0C, cold receptor firing increases and warm receptor firing decreases, indicating cold. When touching an object between 30-45\\u00b0C, warm receptor firing increases and cold receptor firing decreases, indicating warmth. If an object is hot enough to cause pain, nociceptors are activated, signaling pain and further indicating potential injury.\", \"row_data\": {\"id\": \"cfcad20f-a015-41e5-aa7e-a1b0f3efc642\", \"doc_name\": \"docs/feel_temperature.pdf\", \"title\": \"'Neural Responses: Feeling Hot and Cold Temperatures'\", \"content\": \"Perception of temperature is determined by the firing rates of cold and warm receptors. When touching an object between 5-30\\u00b0C, cold receptor firing increases and warm receptor firing decreases, indicating cold. When touching an object between 30-45\\u00b0C, warm receptor firing increases and cold receptor firing decreases, indicating warmth. If an object is hot enough to cause pain, nociceptors are activated, signaling pain and further indicating potential injury.\"}}, {\"UUID\": \"e14be89d-e330-4dcf-b401-b251dbcd3c26\", \"score\": 0.5771155852749257, \"content\": \"To understand FNE thermoreceptors, you need to know about neurotransmission. The brain's signal strength relies on a neuron's firing frequency. We interact with objects of varying temperatures, but thermoreceptors typically detect room temperature, maintaining normal firing rates. Cold or hot stimuli alter the firing rate of corresponding thermoreceptors.\", \"row_data\": {\"id\": \"e14be89d-e330-4dcf-b401-b251dbcd3c26\", \"doc_name\": \"docs/feel_temperature.pdf\", \"title\": \"'Frequency of Firing in Cold Thermoreceptors'\", \"content\": \"To understand FNE thermoreceptors, you need to know about neurotransmission. The brain's signal strength relies on a neuron's firing frequency. We interact with objects of varying temperatures, but thermoreceptors typically detect room temperature, maintaining normal firing rates. Cold or hot stimuli alter the firing rate of corresponding thermoreceptors.\"}}]"
```

# example object when also storing the vector of the query to be able to do semantic search on the cache
- we got here the key formatted: `vector:<hash>...`
- we got the value formatted: `Dict[embeddings: List Vector, response: List[Dict[UUID:..., socre:..., content:..., row_data: Dict[id:..., doc_name:..., tile:..., content:...]]]]`

```bash
127.0.0.1:6379> get "vector:dd20197abb9415478a7b812afc0fd4abf7c16b4f87d4d9eb048ecd46750be487"
"{\"embedding\": [-2.1154770851135254, 0.2666240334510803, 1.8142385482788086, 0.41379299759864807, 3.7191996574401855, -1.8433380126953125, -6.4350056648254395, -2.6746723651885986, 2.1849188804626465, 0.8504980802536011, -2.0410654544830322, -5.0169806480407715, -5.043974876403809, 2.425056219100952, -3.4651873111724854, -4.78447151184082, 2.419900894165039, -3.6159238815307617, 4.253776550292969, 0.19824063777923584, 10.479557037353516, 2.950863838195801, -1.4985581636428833, -2.0088255405426025, -1.9604425430297852, 0.2954627275466919, 0.995936930179596, -1.1401079893112183, -6.8272576332092285, 0.9981338977813721, -3.9778831005096436, 1.8016393184661865, -4.791261672973633, -2.8917245864868164, -3.1821258068084717, 4.468486785888672, -2.136037826538086, 1.6961394548416138, 2.0226945877075195, 0.6157699227333069, 4.89280366897583, -10.114105224609375, -0.25386953353881836, -2.6433420181274414, -1.683518648147583, -4.796107292175293, -2.9774363040924072, -6.530832290649414, -9.462904930114746, 3.6073992252349854, 11.475140571594238, 5.46569299697876, 0.013687835074961185, -76.71723175048828, 7.6744537353515625, 2.5621261596679688, 10.927386283874512, 4.14035177230835, -1.8862158060073853, -3.7766270637512207, 1.0553762912750244, 4.798726558685303, 3.637312889099121, 5.1764445304870605, -3.736956834793091, 3.6463444232940674, -3.210813283920288, 9.502100944519043, 6.249601364135742, -4.599900245666504, 2.137345790863037, -0.45854705572128296, 1.457496166229248, 7.369807720184326, -3.364116668701172, -11.127277374267578, 1.4971153736114502, -0.5136719942092896, -1.21785569190979, 1.449489712715149, 1.3507319688796997, 1.756458044052124, 1.9732142686843872, 3.66756534576416, -5.227423667907715, -1.0611759424209595, -3.066880941390991, 9.167658805847168, 0.7826506495475769, -7.134324073791504, 10.806538581848145, 7.291152477264404, 3.117990493774414, 15.893864631652832, -3.5487444400787354, -4.487802505493164, 6.912250518798828, -0.6939936876296997, 2.5132124423980713, -5.052292346954346, 3.076533794403076, 0.9888189435005188, 3.3470001220703125, 1.4530214071273804, 1.8028861284255981, -2.358039379119873, -0.571504533290863, -9.550314903259277, 10.739927291870117, 4.197419166564941, 3.2606308460235596, -10.151429176330566, 6.272375106811523, 2.986173629760742, -4.839128494262695, -1.23955237865448, -0.90947425365448, 4.988474369049072, 2.8298721313476562, 2.5593936443328857, 1.641748070716858, -4.11443567276001, -12.589862823486328, -8.799067497253418, -1.3627833127975464, -4.258091449737549, 1.1109275817871094, 5.089455604553223, -2.573355197906494, -7.463119029998779, 2.9681382179260254, -5.1480207443237305, -6.359829425811768, 2.5862324237823486, -2.448847770690918, -1.981150507926941, 3.7750768661499023, 3.1306376457214355, 6.969830513000488, -6.0898919105529785, 10.023141860961914, 9.770341873168945, -0.39061450958251953, -2.446850538253784, -5.683545112609863, 5.269312381744385, -2.4141831398010254, -0.9437898993492126, -1.1566307544708252, -7.56643533706665, -4.230492115020752, -6.225662708282471, -0.9155670404434204, 5.1924147605896, 4.279030799865723, -13.304464340209961, -9.85759449005127, 0.09595820307731628, -7.825308322906494, 7.614359378814697, -8.59860897064209, -4.052281856536865, 5.968053340911865, -3.9262683391571045, -8.795631408691406, -1.6330193281173706, -3.8246631622314453, 2.6415979862213135, -3.924058675765991, 1.0764172077178955, 4.00065279006958, -6.925377368927002, 0.7590333819389343, 6.189158916473389, 5.31646728515625, 3.009547710418701, 3.378314256668091, 5.016688346862793, 4.884634971618652, -9.9271821975708, 3.4048826694488525, -3.3766720294952393, -3.527749538421631, -3.8086297512054443, -1.2112174034118652, -4.600334644317627, -4.196463584899902, 2.4234931468963623, 3.4725046157836914, 8.913261413574219, 3.314694881439209, 2.026087999343872, 1.6560182571411133, 10.49744987487793, 6.267114639282227, 7.273893356323242, -6.871723651885986, 1.3731560707092285, 2.8133459091186523, -6.063552379608154, -3.0512936115264893, -3.124145746231079, 1.7837774753570557, -2.5019357204437256, -13.640339851379395, 3.4200656414031982, 3.324592351913452, 7.25059700012207, 1.2486670017242432, -8.377777099609375, -5.262974739074707, 3.276991605758667, -6.972144603729248, -2.555938959121704, -3.9784493446350098, -9.596701622009277, 1.0356677770614624, -0.5672755241394043, 8.906291961669922, -8.099299430847168, -1.538496494293213, 4.350433826446533, 1.8916876316070557, 6.895309925079346, 1.0705398321151733, 0.33727383613586426, 3.418853282928467, 8.352394104003906, 4.612673759460449, -1.1434094905853271, -8.475414276123047, 7.284993648529053, -2.7563138008117676, 2.1071157455444336, -8.905340194702148, -3.6448447704315186, 8.803337097167969, 2.607454299926758, -4.567862033843994, 3.3251421451568604, 6.463708877563477, 2.4738547801971436, -0.4179770052433014, 4.970046520233154, -1.878089427947998, 3.0031802654266357, 6.149051666259766, -5.131364822387695, -4.291234970092773, -1.1427321434020996, 4.077669620513916, -3.006751775741577, -5.8936381340026855, -6.396185398101807, -3.666843891143799, 6.146566390991211, 5.960037708282471, 0.7459564208984375, 8.39448356628418, 4.455574989318848, 2.802408456802368, 0.4525696039199829, -2.0099360942840576, 0.1898597776889801, -4.691901683807373, -7.97169303894043, -3.1460132598876953, 6.558750629425049, -7.068082809448242, -5.170548439025879, -8.50394344329834, 0.8200567960739136, 5.474276542663574, -0.7757282257080078, 6.891482353210449, -0.43312782049179077, 9.012090682983398, -4.576420307159424, 0.7488460540771484, -1.8130629062652588, -0.7786442637443542, 0.7985100150108337, 3.558379888534546, -1.1457371711730957, 4.090900421142578, -3.441136121749878, 8.834359169006348, 7.7821784019470215, 6.602021217346191, 6.331871509552002, 10.983039855957031, -2.1970632076263428, -11.380983352661133, 8.870372772216797, -1.9518046379089355, -4.363503932952881, -6.129969120025635, -6.02894926071167, -0.07136949896812439, -2.509955644607544, -7.644101619720459, -0.3551779091358185, 1.3913148641586304, 7.106586456298828, 6.808965682983398, -1.2090057134628296, 4.336454391479492, -0.4365794360637665, -2.9427506923675537, 0.9546144604682922, -4.5823774337768555, -5.342405319213867, 0.8968706130981445, -4.572734832763672, 1.761574625968933, -3.3622493743896484, -4.394671440124512, -6.5501933097839355, 3.7137131690979004, -5.965291500091553, -1.9303102493286133, -8.637052536010742, 11.050804138183594, -3.960310459136963, 2.395639181137085, -3.3112077713012695, -0.2566807270050049, -0.002635351615026593, -3.019263744354248, -7.047951698303223, 2.76119327545166, 13.00218391418457, 5.417057037353516, -1.9265458583831787, -1.2750266790390015, -2.2192652225494385, -0.24576599895954132, -0.2731328308582306, -0.12423281371593475, 4.489023685455322, -3.1736888885498047, -0.3673487603664398, 8.673925399780273, 4.258652210235596, -3.7548277378082275, -13.224361419677734, -0.5483219623565674, 4.418467044830322, -8.386537551879883, -2.6672353744506836, 3.761147975921631, -1.7582842111587524, 3.15716814994812, 9.225749969482422, 3.642484426498413, -1.5947767496109009, 2.503856897354126, -2.607936143875122, -6.8209381103515625, 0.2882096469402313, 4.111293792724609, -2.060176372528076, 4.1141839027404785, 1.0868586301803589, 2.0466485023498535, -9.64832592010498, -7.272172451019287, -0.7269318699836731, 4.338057518005371, 8.754449844360352, -0.681445837020874, 1.4934327602386475, -7.622511863708496, -2.667973518371582, -10.78758430480957, -3.2034952640533447, 7.748733043670654, -3.3846006393432617, 2.6668264865875244, 7.106943607330322, -5.3851399421691895, -10.352439880371094, -2.191425085067749, 2.9706428050994873, -7.4818620681762695, -3.3700459003448486, -2.7471632957458496, 2.585388660430908, 0.5343074202537537, 2.797511339187622, -5.168315887451172, -5.787874698638916, 0.7687143087387085, -1.6478859186172485, -8.879276275634766, 4.396716117858887, -1.2814162969589233, 3.2449467182159424, -7.043951511383057, 2.622570514678955, -4.569204807281494, 3.275502920150757, -4.797573566436768, -2.9406991004943848, -4.385513782501221, 1.2551188468933105, -2.5743796825408936, -11.506731033325195, -4.730318069458008, 1.0679314136505127, 3.6128745079040527, -4.260937690734863, -0.8879402279853821, 0.17270272970199585, -6.096515655517578, 0.6286020874977112, 1.7379995584487915, -1.4679888486862183, 6.293557167053223, 6.421231746673584, -5.73332405090332, 2.0106074810028076, -9.237091064453125, 7.4514288902282715, -0.5809891223907471, -5.885160446166992, 0.9649484753608704, 0.5636865496635437, -3.1567468643188477, 0.821254312992096, -3.7090837955474854, 1.0848298072814941, -6.057090759277344, -2.5601015090942383, -1.5856757164001465, 3.8897647857666016, -0.2479500025510788, -4.050027370452881, 8.99778938293457, 4.5784592628479, -6.429805278778076, 1.1285480260849, 3.8926479816436768, -1.29817795753479, 1.7761651277542114, -0.10675311088562012, -2.079789161682129, 7.569039821624756, 1.841854453086853, -4.474747180938721, 5.644819736480713, 5.315709590911865, 4.960631847381592, 1.5423575639724731, 7.456075668334961, -5.455245018005371, 1.347091555595398, -10.697203636169434, -2.3436806201934814, 5.468256950378418, 3.333378791809082, 11.901856422424316, -3.3373234272003174, 5.136353492736816, 3.7078256607055664, -1.3091434240341187, 3.071502685546875, 3.873483657836914, 3.314527988433838, -0.4408353567123413, 0.02042100578546524, -9.343400001525879, 1.8644146919250488, -0.629574716091156, -5.154595851898193, 7.596637725830078, -1.7578442096710205, 3.9958178997039795, 3.1242361068725586, 1.107578158378601, 3.932246208190918, 8.917515754699707, 3.2587273120880127, 7.321430683135986, -0.9429886937141418, 3.119689702987671, 2.362708568572998, -5.448171615600586, -12.737812042236328, -2.819366931915283, -0.5113097429275513, -2.752528667449951, -4.449990749359131, -1.9883086681365967, 2.9010348320007324, 2.461944580078125, 5.328717231750488, -5.65261173248291, -7.8747429847717285, 3.650050640106201, -6.053752422332764, 3.4977428913116455, 7.303974628448486, -1.3448013067245483, -3.2642593383789062, 4.4083733558654785, -1.0252076387405396, -0.10558686405420303, -9.020748138427734, 6.2086687088012695, -8.057377815246582, 2.2888224124908447, -8.038345336914062, -3.57131290435791, -0.9437611699104309, -4.888121604919434, -7.481921195983887, 5.662231922149658, 3.0674424171447754, -3.101686716079712, 0.7212220430374146, -5.267399311065674, 6.615294933319092, 3.711341381072998, 7.939525604248047, 1.5728626251220703, 2.103660821914673, -5.139779090881348, -7.9155426025390625, 3.2819931507110596, -1.182469367980957, -5.17984676361084, -1.8880473375320435, 3.7503204345703125, 4.97205114364624, -5.902846336364746, -2.4139585494995117, -7.478500843048096, -5.196979999542236, -9.087559700012207, 7.0634965896606445, 0.7897523641586304, 2.023524045944214, 2.398250102996826, -0.647570013999939, -2.4968457221984863, 0.9938361644744873, -6.1876959800720215, -3.3791656494140625, 11.098970413208008, -4.649514198303223, -0.45078811049461365, 5.073394298553467, 0.23361492156982422, -3.3122262954711914, 6.968973159790039, -2.350506544113159, -2.0977981090545654, 5.802864074707031, 6.195112705230713, 0.07683045417070389, -4.5769500732421875, 4.388314247131348, 4.242493152618408, -3.8150100708007812, -3.8363540172576904, 3.8344292640686035, -1.7400648593902588, -4.06889533996582, 0.9197235107421875, -0.7076756954193115, 2.936460256576538, -3.068610668182373, 5.053619384765625, -4.965609550476074, -9.465067863464355, 4.7278852462768555, -1.7401151657104492, -15.43666934967041, -6.375912666320801, 5.017955780029297, -1.5680763721466064, 6.878398418426514, -3.869126081466675, 0.6811327934265137, 1.5255823135375977, 1.4720455408096313, -11.19786548614502, 1.4090542793273926, -2.7422101497650146, -5.488183498382568, -0.8650081753730774, 3.405118465423584, -3.6447877883911133, -2.0522890090942383, 1.2736114263534546, -2.3816287517547607, 9.445066452026367, 7.711741924285889, 1.1128439903259277, -3.4461171627044678, 3.987131118774414, -0.16869087517261505, 5.508786201477051, -9.939517974853516, -3.0135385990142822, 6.558826923370361, -1.5726606845855713, 2.4434914588928223, -2.6456971168518066, 1.8754067420959473, 4.016862869262695, -0.8777281641960144, 2.859072208404541, -5.165999412536621, -5.951833248138428, -0.6646500825881958, -0.41255223751068115, -11.45466423034668, 3.0315871238708496, -3.4532573223114014, 4.374546527862549, -4.5204620361328125, 12.495386123657227, -8.237041473388672, 5.795152187347412, -6.847842216491699, 4.1446099281311035, -0.8886786699295044, -4.755643844604492, -8.366244316101074, 2.7840664386749268, -0.5968969464302063, 6.198591232299805, 4.625657081604004, 5.3545098304748535, -0.9153268337249756, 2.988820791244507, -3.0996944904327393, -0.33392661809921265, 4.049753665924072, -4.46754264831543, -0.8093128800392151, -5.344274520874023, 6.865234375, 2.3867685794830322, -6.456553936004639, 4.639861583709717, 5.21353006362915, -10.386605262756348, 3.942021131515503, -1.66847825050354, -5.437163829803467, -0.5006295442581177, 1.5399959087371826, -2.738104820251465, 9.969926834106445, -8.141016960144043, -1.1800955533981323, 3.636476755142212, 8.225411415100098, 5.420187950134277, -5.839600086212158, 2.214369297027588, -4.021771430969238, -0.9917019605636597, 9.526320457458496, 2.1227612495422363, 1.7792401313781738, -3.6679909229278564, -4.291693687438965, 1.4447449445724487, -1.9710028171539307, 1.5693886280059814, -8.906804084777832, 4.784017562866211, 3.9138848781585693, 0.9575879573822021, 1.062117576599121, 3.5498712062835693, 13.58403205871582, 4.885873317718506, 12.118412017822266, 6.185111999511719, -5.3699116706848145, -8.275288581848145, -3.543217182159424, 4.17440128326416, -13.846840858459473, -2.660540819168091, 3.179144859313965, 2.9492015838623047, -3.731841564178467, -3.9432590007781982, -5.232688903808594, 1.8842567205429077, 8.951403617858887, -5.34540319442749, -0.6092826128005981, 6.168370246887207, 0.7895645499229431, -6.884348392486572, 3.417961597442627, -4.073558330535889, -7.971942901611328, 5.002597332000732, -1.4555760622024536, -0.02466920018196106, 1.0556243658065796, -6.3870158195495605, 0.5250009298324585, -1.228265404701233, 1.8122795820236206, -3.1831984519958496, -1.7283201217651367, -0.2504817247390747, -8.941797256469727, 1.4187544584274292, 9.001832008361816, -1.356674075126648, -3.978215217590332, 6.382408142089844, 1.400849461555481, 4.303965091705322, 0.22297100722789764, -10.548646926879883, 6.417967796325684, 1.5460968017578125, -9.445170402526855, -1.5313398838043213, 2.6822617053985596, -5.4130330085754395, 2.3730151653289795, 6.344557762145996, -9.798235893249512, -2.1427218914031982, -3.5474159717559814, -0.3454231917858124, 1.5255141258239746, 0.8305297493934631, 0.7645174860954285, 5.516700744628906, -1.207244873046875, 1.2739282846450806, -1.7767301797866821, -2.517014503479004, -2.1683640480041504, 0.9568350911140442, -5.1163859367370605, -0.9959267377853394, -0.9225832223892212, -2.450101613998413, 11.921368598937988, 4.210698127746582, 1.8957549333572388, 2.3663291931152344, 3.6416306495666504, -1.560180902481079, 0.050700441002845764, -1.444579005241394, 2.9457266330718994, -3.761026382446289, -6.256514549255371, 7.870824813842773, 6.6717071533203125, 3.587799549102783, 0.36027413606643677, 2.628976583480835, -3.096654176712036, 2.670351028442383, 2.2155909538269043, 6.122061252593994, -0.6450045108795166, -2.0590028762817383, 0.47159573435783386, -1.9623721837997437, -1.4702402353286743, -2.9069206714630127, -2.5838241577148438, -7.011819362640381, 2.6535685062408447, -3.0263562202453613, -0.3957422375679016, 2.85475492477417, -3.0329060554504395, 0.9009623527526855, -7.232650279998779, 5.885009765625, 3.568760633468628, -4.699648380279541, 3.0762338638305664, 7.514856815338135, 3.603055953979492, -4.946926116943359, 9.291884422302246, 8.065838813781738, 7.1748762130737305, 2.7758750915527344, 3.4827747344970703, -6.395135879516602, 2.392976760864258, 0.256680428981781, -4.711451053619385, 4.175591468811035, -2.957705497741699, -4.810981750488281, 0.16920289397239685, 2.6125223636627197, 2.571398973464966, -8.081473350524902, 11.808499336242676, 2.626856565475464, -1.008447527885437, -0.9885134696960449, -6.779516696929932, 1.0881470441818237, 2.5473880767822266, -7.143502235412598, -6.954800605773926, -1.89171302318573, -4.327291965484619, 0.05857738107442856, -0.16615474224090576, -4.269750118255615, -0.8052784204483032, -1.7420645952224731, 0.30012354254722595, 5.441363334655762, -1.041688323020935, 7.195712089538574, -0.1125170886516571, -7.721090793609619, 0.42478495836257935, -11.387710571289062, -3.1958281993865967, 12.967026710510254, 5.051912784576416, -0.8812114000320435, -1.0732502937316895, -1.5638043880462646, -6.647467613220215, 3.7677736282348633, -7.3357648849487305, 3.3244853019714355, -8.456549644470215, 8.26382064819336, -0.9432892799377441, 2.4236838817596436, 1.4264832735061646, -9.950126647949219, 7.637003421783447, -9.441670417785645, 0.3528495728969574, 5.606325149536133, -2.919182777404785, -1.2312864065170288, 0.41313743591308594, -5.3635687828063965, 6.361586093902588, -5.037479400634766, 2.0851798057556152, -5.715176105499268, 1.411189079284668, -3.106828212738037, 0.9067454934120178, -2.0942726135253906, -0.15424327552318573, -0.21957051753997803, -5.485583782196045, 8.086264610290527, -0.9870492815971375, -7.8009562492370605, 5.392215728759766, 1.1963461637496948, -2.5937459468841553, 1.3623006343841553, 11.47292423248291, -3.7152798175811768, -6.909605026245117, -12.244454383850098, -4.791685581207275, -5.707043170928955, -4.673007011413574, 11.617910385131836, 5.374368667602539, 1.6744967699050903, -6.828189373016357, 3.520840644836426, 5.851720333099365, 1.0851562023162842, -7.762465476989746, -2.979703187942505, -13.31988525390625, -4.052704334259033, -7.738262176513672, 0.31449830532073975, -3.6810495853424072, -2.7013745307922363, 8.125940322875977, 7.313613414764404, -8.716473579406738, -0.6013986468315125, -1.4164142608642578, -2.9166455268859863, -9.359694480895996, -6.827966690063477, 5.639047622680664, 3.6509575843811035, -6.121668338775635, -2.2770187854766846, -1.6623380184173584, 4.142375946044922, 1.320954442024231, 0.2234630435705185, 1.2481658458709717, -0.8898120522499084, -1.4310177564620972, -0.8865837454795837, 1.4115039110183716, 9.417068481445312, 9.138361930847168, 1.668752670288086, -10.757185935974121, -1.254386067390442, -0.6201754808425903, -0.12347432971000671, 4.143718719482422, -4.154735088348389, 9.243478775024414, -4.1949992179870605, -5.2376484870910645, -4.430365562438965, 1.5118341445922852, -2.2992353439331055, 6.229128360748291, -3.591578483581543, 0.22012217342853546, -9.140486717224121, 5.7449951171875, -0.011232038028538227, -1.5110435485839844, -4.04435920715332, -1.7900222539901733, 7.532114028930664, 0.35921725630760193, 4.0584001541137695, -8.734010696411133, 2.5132758617401123, 2.5325193405151367, 5.523129940032959, 3.0758168697357178, 0.21718363463878632, -1.2582837343215942, 4.295350074768066, 0.3684622049331665, 3.9355483055114746, 2.727802276611328, 4.114770412445068, -2.6345696449279785, 9.654861450195312, -1.9117631912231445, 4.719241619110107, -7.881397247314453, 6.353817462921143, 2.963472843170166, 0.9637120366096497, -4.216114521026611, -8.321287155151367, 4.103122711181641, 4.262922763824463, -6.538720607757568, 3.8536136150360107, 1.062076210975647, 9.014778137207031, -7.407046318054199, 4.005308151245117, -12.723373413085938, 0.516621470451355, 0.6028758883476257, 8.117925643920898, 3.1633286476135254, 1.6494592428207397, -0.9363406300544739, -1.0251761674880981, -4.997838497161865, 0.284356951713562, 0.9212022423744202, 1.4721126556396484, 7.754382133483887, -4.503316402435303, 3.000627279281616, 2.434640645980835, -12.016887664794922, 9.809136390686035, -6.086399555206299, 6.5199503898620605, 5.947683811187744, 1.2397358417510986, -1.5773783922195435, 0.09636740386486053, -3.1737945079803467, -3.9981844425201416, 0.2642410099506378, -2.8132827281951904, -6.95528507232666, 6.226634979248047, -10.60312271118164, -5.62723970413208, -2.4752895832061768, -0.07294023782014847, 1.6640256643295288, -0.32212984561920166, 1.8420501947402954, -0.0701979324221611, 1.2881187200546265, -7.223742961883545, 4.026197910308838, -0.23309749364852905, -10.292202949523926, 0.7394614815711975, -5.765864372253418, 7.572016716003418, 7.676930904388428, 2.6965551376342773, 0.15494757890701294, 4.270870685577393, 11.987629890441895, 5.857852935791016, -3.879239797592163, 4.399377822875977, -32.33058166503906, 4.86812686920166, 1.0442662239074707, -6.283731460571289, -4.2263264656066895, -1.8725494146347046, 7.790778636932373, -6.637210845947266, -6.182269096374512, -5.301817893981934, 7.8778462409973145, -8.20280647277832, -1.991141438484192, 2.244225025177002, -4.859358310699463, 0.791408896446228, 3.7160263061523438, -6.294924736022949, 1.5384211540222168, -2.569380283355713, -2.8170254230499268, 0.46008843183517456, 0.18578694760799408, -11.941380500793457, 3.7293777465820312, 7.4397406578063965, 11.529582023620605, -1.1016095876693726, 3.821901321411133, 0.5122168660163879, -0.8282760977745056, -2.837294340133667, -3.4729106426239014, 3.3842530250549316, -3.373133659362793, -0.3459382951259613, -7.925261497497559, 3.003058910369873, -1.4130229949951172, -5.776421070098877, -2.970705032348633, 15.1466703414917, -2.586467742919922, -1.3105982542037964, 0.38181835412979126, 1.3990068435668945, 4.315018177032471, -8.985695838928223, 9.885797500610352, 2.5314857959747314, 1.9212242364883423, 2.466468334197998, 10.25478744506836, 8.939194679260254, 3.8464202880859375, 4.040561199188232, 3.9191133975982666, -6.311733245849609, 8.871800422668457, -6.071556091308594, 1.3008140325546265, 6.785214900970459, 2.4861998558044434, -4.371450424194336, -1.5928136110305786, 3.04058837890625, 0.09134242683649063, 3.3701140880584717, -0.9047510027885437, 0.5032384991645813, 8.469747543334961, 1.1306674480438232, -3.6710922718048096, 4.5213623046875, 2.7056961059570312, 1.8682631254196167, 0.946036696434021, -1.980879306793213, 3.097740411758423, -3.0032708644866943, -7.176522254943848, 7.611574649810791, 8.501612663269043, -0.8368028402328491, -4.238127708435059, -5.977559566497803, -11.743282318115234, -1.9452011585235596, -7.796038627624512, 2.6787562370300293, 8.730775833129883, 6.4580793380737305, -0.4184795916080475, -0.35848814249038696, 2.794405937194824, 15.508477210998535, -0.7366970777511597, 3.689714193344116, -1.7963190078735352, -4.2170023918151855, 3.311678171157837, -6.461713790893555, -0.9133634567260742, 4.222382068634033, -4.385276794433594, 3.057454824447632, -7.094723224639893, -2.223970413208008, 5.251640796661377, -4.946123123168945, 5.3101301193237305, 0.5090909004211426, 2.064438819885254, -0.07691388577222824, -1.1435751914978027, -6.735994338989258, -6.431974411010742, -6.487405300140381, -0.18019810318946838, -4.249401569366455, -2.7743289470672607, -10.702430725097656, -2.1139063835144043, -2.2529029846191406, 4.034787178039551, -2.7837541103363037, -2.014755964279175, 5.295164585113525, 1.6615452766418457, 2.245114803314209, -2.218018054962158, -0.1675652265548706, -5.281707286834717, 10.512951850891113, -0.07815919816493988, 14.432589530944824, -5.2215118408203125, 2.6379942893981934, -6.431938171386719, 1.5415277481079102, 2.1290907859802246, -8.019121170043945, -2.9740312099456787, 7.719696521759033, 8.463704109191895, 0.8942158222198486, -6.414251804351807, -1.1660689115524292, 1.1778041124343872, 9.921342849731445, 5.356453895568848, -3.5645270347595215, -3.4002902507781982, -2.082014560699463, -0.07426142692565918, 2.472649097442627, 6.551882743835449, 5.199862003326416, -0.5723954439163208, -6.3547797203063965, 2.9403741359710693, -9.559602737426758, -2.9569504261016846, 4.766416072845459, -1.7847379446029663, 20.44550895690918, 3.3806087970733643, -1.9659379720687866, 4.734086036682129, -1.142043948173523, -9.822616577148438, 5.67675256729126, -6.739339828491211, 0.5696721076965332, 1.8533927202224731, 1.6739777326583862, 10.539061546325684, 2.0768325328826904, 0.5952350497245789, -0.1404012143611908, -1.1520148515701294, 1.3476351499557495, 2.750436544418335, 0.5575017929077148, 6.584780693054199, 9.70002269744873, 6.995615005493164, 2.129540205001831, 3.5922038555145264, -4.612889766693115, -4.075323581695557, 2.363990306854248, -5.859037399291992, 15.36210823059082, 4.119567394256592, 0.8169272541999817, -6.15542459487915, 3.497023582458496, -2.8246102333068848, -8.887914657592773, 2.183997392654419, -3.989197015762329, -5.465310573577881, -4.474886894226074, 6.005536079406738, 1.3535877466201782, 6.785554885864258, -8.548320770263672, -5.628816604614258, 1.1205029487609863, 2.1564230918884277, -9.349031448364258, -6.455171585083008, 1.1708413362503052, 1.2799028158187866, -5.016806125640869, -8.611011505126953, 1.8889224529266357, 2.0100557804107666, 1.894161581993103, -6.829017639160156, 0.8977801203727722, -5.8785810470581055, -8.257587432861328, 7.070044040679932, -8.665374755859375, -7.24973201751709, 3.6099629402160645, -4.894780158996582, 4.515666961669922, 0.08710324019193649, 4.3798508644104, 1.371551752090454, 6.438158988952637, 5.859018325805664, -2.803959369659424, 2.439488649368286, 0.4614081084728241, -1.0083035230636597, 12.727752685546875, -2.192967414855957, 5.367458343505859, 7.710994243621826, 5.059550762176514, -3.9106380939483643, -5.705082893371582, -6.40839958190918, 10.788564682006836, 2.489797592163086, 3.5850913524627686, -0.2574334442615509, -3.2666869163513184, -7.396895408630371, -4.660221099853516, 6.478566646575928, -2.5688371658325195, -5.898740291595459, 3.422091245651245, 3.2364275455474854, 4.507379055023193, -1.8974164724349976, 3.7640225887298584, 0.0554826445877552, -1.2898622751235962, 2.382331371307373, 5.051426410675049, -1.7053184509277344, 0.9963715672492981, 5.508366107940674, -8.958622932434082, -6.960050582885742, 5.4909796714782715, 1.9535503387451172, -2.4779467582702637, -5.505771636962891, 0.12546353042125702, 2.5612194538116455, -2.6876039505004883, 0.15308573842048645, 0.021808767691254616, -0.1511056125164032, 2.295506477355957, 3.1265807151794434, 6.978455543518066, -0.6727911829948425, -5.079349994659424, -2.04144549369812, 2.4306082725524902, 0.39950865507125854, -0.2828090786933899, 6.213051795959473, -3.6164987087249756, 0.8358901143074036, -3.955533266067505, 0.3259658217430115, 0.8281449675559998, -5.82692813873291, 1.4318327903747559, 8.027263641357422, -6.952198505401611, -6.462225914001465, 1.8005306720733643, -1.4519550800323486, -1.468376874923706, -1.4110037088394165, -4.70123815536499, 1.658057689666748, 0.19849713146686554, -3.7186012268066406, 4.3824028968811035, -1.784825325012207, 6.022528171539307, 9.05272102355957, 6.84311580657959, 5.337957382202148, 3.086432695388794, 0.5504314303398132, -4.93668794631958, -4.246894836425781, -7.0680952072143555, 6.813638687133789, 0.9593474268913269, 2.4224138259887695, 2.4746510982513428, -7.503381252288818, 0.7413538694381714, 5.220674991607666, -1.0139516592025757, 0.15270498394966125, -1.8279262781143188, -0.5297585725784302, 4.018081188201904, -6.579409122467041, -0.5664267539978027, 1.7590866088867188, 4.7150702476501465, 6.3627543449401855, 2.1607346534729004, 0.3479463756084442, -7.538704872131348, 5.489866733551025, -6.032626152038574, -12.588827133178711, 11.137181282043457, -11.098912239074707, -82.05732727050781, -4.788456916809082, -0.17037717998027802, -0.9383730292320251, 3.043156623840332, 0.1618274301290512, -6.393364429473877, -5.722598075866699, 6.911561489105225, -0.28540322184562683, -4.193253517150879, -5.149852752685547, -5.6208648681640625, 2.2171356678009033, 1.052855372428894, -2.624039649963379, 1.3988958597183228, -1.745236873626709, -5.1148152351379395, -3.07004714012146, -3.058583974838257, 4.178343296051025, 4.497433185577393, -6.8772664070129395, 0.012583626434206963, 6.226858139038086, 1.0173325538635254, 7.1484551429748535, 0.7141234874725342, -2.9544341564178467, -0.47450414299964905, 2.7016618251800537, -5.66455602645874, -3.0492470264434814, -2.3723323345184326, 3.7636749744415283, -0.42863595485687256, 6.941139221191406, 0.5012333989143372, 2.533830165863037, 3.3069345951080322, -1.006085991859436, 5.557435989379883, 5.474684238433838, 0.9351380467414856, 3.815765857696533, 2.3747713565826416, 4.35952615737915, 1.3931297063827515, 4.493973255157471, -4.606626033782959, -2.3458902835845947, -0.8331977725028992, -1.305907130241394, 1.7380976676940918, 2.9734933376312256, 3.342012643814087, 5.892128944396973, -6.229981422424316, -3.2134335041046143, -4.3072099685668945, 8.900846481323242, 1.5910621881484985, 2.5214478969573975, 0.2397020012140274, 2.6331164836883545, -4.715708255767822, 1.7888479232788086, 5.797229766845703, 3.5072429180145264, -1.603487491607666, -1.194203495979309, -1.8808772563934326, -2.943247079849243, 0.688110888004303, 9.781319618225098, -6.4715256690979, -7.387362480163574, -5.442581653594971, -7.5479865074157715, 2.208036184310913, 0.9257990121841431, -13.4207181930542, 1.3568044900894165, -4.949552059173584, 3.2335879802703857, 6.807335376739502, 6.403373718261719, 4.997054100036621, -6.760860443115234, -2.5297298431396484, 1.451393961906433, -7.06500244140625, 4.5404133796691895, -4.207004547119141, 8.32922077178955, 0.6955351829528809, -7.460292816162109, 0.4065014123916626, -3.5872695446014404, 0.9628138542175293, -12.637899398803711, 4.973163604736328, 1.5074487924575806, -3.3179662227630615, 0.9895960092544556, 0.10615742951631546, -1.9215803146362305, -6.713132858276367, -10.055006980895996, -0.035479363054037094, 5.553498268127441, 0.5516218543052673, -0.2219793051481247, 3.72649884223938, -1.0278702974319458, -0.634374737739563, 2.585289716720581, 3.3566536903381348, 3.482713222503662, 8.962584495544434, -7.055723190307617, 1.1129258871078491, -6.518229961395264, -6.655190467834473, 0.1470477432012558, 0.26723089814186096, 2.0167713165283203, 0.5891448855400085, 1.039359450340271, 8.67173957824707, -3.4349710941314697, 5.31878662109375, 3.4924490451812744, -2.4111509323120117, 8.148701667785645, -1.9062268733978271, -7.394252777099609, -2.5714080333709717, -7.1333327293396, 4.468681335449219, -2.970177412033081, -4.684604644775391, -0.9502295851707458, 2.1933887004852295, 1.5689975023269653, 0.20606353878974915, 2.3156211376190186, 2.491605520248413, 3.261272668838501, -0.6763677000999451, 0.5542884469032288, -4.095242500305176, -0.34055641293525696, -4.5048651695251465, 1.0122615098953247, -3.8971686363220215, 0.49479952454566956, -4.431601524353027, 4.302191734313965, 3.0518436431884766, -3.5184054374694824, 1.4106686115264893, 4.352621555328369, 3.187655448913574, 1.3109023571014404, 6.593615531921387, 7.180336952209473, 5.836637020111084, 5.751585960388184, 7.8723673820495605, -2.2049543857574463, 9.483125686645508, 1.6447073221206665, -6.469018459320068, -8.701164245605469, 1.2721257209777832, 4.876550197601318, -8.78732967376709, 5.260993003845215, 4.594779968261719, 1.0963826179504395, 2.3402318954467773, 0.7217970490455627, 2.0828959941864014, 0.7876444458961487, -7.91013240814209, -3.619209051132202, 1.2513855695724487, 1.701169490814209, -4.712588310241699, 0.015277707949280739, 3.257946729660034, -2.480367422103882, 2.8466646671295166, -1.5794587135314941, -3.903984546661377, 4.100405216217041, -11.115825653076172, -4.052207946777344, 0.7767931818962097, 0.8072082996368408, 6.831361293792725, -7.208249568939209, -1.8489125967025757, -4.970929145812988, 0.22853347659111023, -3.701061725616455, 2.2051401138305664, 1.9655028581619263, -7.564729690551758, 4.293459415435791, -1.3789396286010742, -7.597092151641846, -4.391363143920898, 3.6886544227600098, -8.497156143188477, 13.555009841918945, -0.6178596019744873, 9.167497634887695, 0.0210945475846529, -1.4123157262802124, -2.7142691612243652, -10.445683479309082, -5.872443199157715, 9.36269474029541, -3.699916362762451, -5.0591511726379395, -7.612243175506592, 6.630790710449219, 6.660477161407471, 0.5419482588768005, -6.156142711639404, -2.3039350509643555, -5.806320667266846, 3.8462541103363037, 3.481830596923828, 8.786192893981934, -1.4032433032989502, -0.41891810297966003, 7.156350612640381, 2.124528408050537, 3.8298351764678955, 1.7653326988220215, -0.24931904673576355, 3.424028158187866, 2.5943124294281006, 2.6826367378234863, -0.6206409931182861, -3.013826847076416, 7.217200756072998, -6.325819492340088, 4.728094100952148, -8.63920783996582, 8.062665939331055, -4.062845706939697, -8.645848274230957, -8.27880859375, -5.30798864364624, 1.3192898035049438, 4.707692623138428, 6.503304481506348, 7.436670303344727, 2.998400926589966, 0.10950830578804016, 5.08843469619751, 29.95798110961914, -0.34298110008239746, 9.378805160522461, -2.0395846366882324, 16.704795837402344, 2.2209911346435547, 0.12738065421581268, -0.13940882682800293, -5.578154563903809, -7.3323283195495605, -8.567764282226562, -2.002261161804199, -1.1054370403289795, 8.946907997131348, -12.404013633728027, -2.852922201156616, 1.8817986249923706, 5.419755935668945, -3.704695224761963, -0.5036599636077881, -19.90562629699707, 6.752607345581055, -0.2669754922389984, -0.7951309680938721, -1.7148581743240356, -7.6839985847473145, -0.8728824257850647, -2.293184280395508, -3.157714366912842, 2.944556951522827, 0.06449513882398605, -0.36647307872772217, 6.485435485839844, 1.5738763809204102, 10.140625953674316, 0.7560250163078308, -4.840073585510254, -5.614883899688721, -6.410682678222656, 2.7133255004882812, 5.334702014923096, -9.172367095947266, 2.427084445953369, 5.407151222229004, -0.5647382140159607, 8.833412170410156, -1.4404518604278564, -11.247354507446289, 4.73686408996582, -2.5919370651245117, 10.855850219726562, 6.39351224899292, 11.995538711547852, 5.766940116882324, 5.034917831420898, 2.6215591430664062, -1.814947247505188, 5.811673641204834, -1.4585576057434082, -12.55423641204834, 7.330566883087158, 11.106618881225586, 7.503872394561768, 5.132339000701904, 6.3170599937438965, -5.87890100479126, -0.1783411204814911, -3.735678195953369, 1.217046856880188, 2.607386827468872, -1.3985832929611206, 3.881721258163452, -0.4477299451828003, 7.963733196258545, -13.228585243225098, 0.760584831237793, -0.10846001654863358, -12.580995559692383, -0.5045907497406006, -1.1258158683776855, -1.3787782192230225, 3.8184521198272705, 2.1716184616088867, -4.253499507904053, -9.946510314941406, -4.0441460609436035, -3.71645188331604, -1.848026156425476, -1.9247286319732666, 2.366941452026367, 4.7816691398620605, 6.116094589233398, 4.931093692779541, -3.6673824787139893, -7.975381851196289, 13.095673561096191, 5.533498764038086, -2.2647244930267334, 2.575939178466797, 1.264838457107544, -1.8779242038726807, 1.3428144454956055, 4.517369270324707, -3.7928922176361084, -8.846400260925293, -6.314815521240234, -7.175861835479736, -4.67591667175293, 5.8678388595581055, -4.083699703216553, 0.8946910500526428, 0.8103740811347961, 7.596882343292236, -7.079899311065674, -1.976540207862854, 1.188730239868164, 3.571431875228882, -1.3075988292694092, -3.716888904571533, 3.7460970878601074, 9.878021240234375, 1.2311666011810303, 6.552992820739746, 2.9249930381774902, 7.279390811920166, 9.364476203918457, 3.676598310470581, 4.279722213745117, -3.955373764038086, 0.3846887946128845, 0.7557975053787231, -9.061583518981934, 0.21000374853610992, -4.220467567443848, 6.063459396362305, 7.553685188293457, 11.063345909118652, -1.155003309249878, 4.001014709472656, -2.5806665420532227, 10.746743202209473, -3.1504905223846436, -0.5842392444610596, -1.5839165449142456, -2.7377054691314697, 9.925010681152344, -1.427864670753479, 7.576898574829102, 1.3379409313201904, 1.0350902080535889, -0.45166948437690735, 1.0020267963409424, 2.5034244060516357, -11.156294822692871, 1.5433058738708496, 2.755023241043091, 0.7231131792068481, -6.108284950256348, -5.313846111297607, -3.3511791229248047, 0.04894424229860306, -3.1842617988586426, 0.21078646183013916, 6.090966701507568, 6.168371200561523, -1.6561484336853027, 7.0806684494018555, 2.312981367111206, -0.3423471748828888, 3.1111295223236084, -6.82194709777832, -6.837200164794922, 0.20145349204540253, -0.40567001700401306, 3.843313455581665, 3.724759817123413, 2.9403014183044434, -7.742326259613037, 5.2932353019714355, 2.697692632675171, -6.454414367675781, -10.834227561950684, 3.7125613689422607, -4.978463172912598, 2.5610013008117676, -0.06823080033063889, 13.171170234680176, 1.6261322498321533, -0.6774977445602417, 2.549868106842041, -5.782252311706543, 2.1163504123687744, -1.8252074718475342, 0.6341719627380371, -3.482025146484375, -2.38891339302063, 2.9160819053649902, -0.6601163744926453, 2.963719129562378, -5.149934768676758, -6.06765079498291, -2.5519139766693115, 4.462373733520508, 1.2690739631652832, -9.32047176361084, 1.6063578128814697, 6.9988298416137695, 13.57036304473877, 0.23826394975185394, 5.362762928009033, 3.1034650802612305, -3.707587480545044, 7.749011516571045, -10.750136375427246, 9.220061302185059, 5.473282814025879, 8.014338493347168, -5.687890529632568, 4.702625274658203, 3.6546003818511963, -5.006949424743652, 3.6953186988830566, 7.0202250480651855, 5.90675687789917, 0.2865101099014282, -3.545126438140869, 4.975337505340576, 1.8268334865570068, 8.532724380493164, -4.90933084487915, -2.744011402130127, 0.03099835105240345, 2.032498598098755, -1.9847739934921265, -2.2431187629699707, 1.7440482378005981, -1.135912299156189, 0.031995341181755066, 2.1009926795959473, 0.5128008127212524, 0.22702839970588684, -5.01190185546875, -4.112567901611328, -2.2125244140625, -0.25421878695487976, 6.1173624992370605, -6.347054958343506, 0.7576329708099365, -1.223098874092102, -2.1665303707122803, -3.9196360111236572, 12.349183082580566, 5.404465675354004, 2.222412109375, 7.493913650512695, -3.241349220275879, 2.6382639408111572, 3.147674560546875, 5.598055839538574, -4.6478095054626465, 3.434390068054199, 0.5460378527641296, 5.940290451049805, 8.04273796081543, 2.2013142108917236, 1.2917121648788452, -1.5546801090240479, 7.6971659660339355, 2.2824604511260986, 8.670120239257812, -5.240869045257568, 4.4283857345581055, -12.485926628112793, 0.25313299894332886, 5.231209754943848, 1.170452356338501, -0.35399219393730164, 3.4981777667999268, 2.9966530799865723, 0.8225046396255493, -1.8447438478469849, 10.386313438415527, 0.4905513525009155, 1.3980070352554321, -1.9938026666641235, 0.1349761039018631, -11.302772521972656, -1.5075441598892212, -0.4974627196788788, -2.6918394565582275, 1.5601928234100342, -2.157785415649414, -0.032632991671562195, 8.403029441833496, -0.29938367009162903, -0.3149770498275757, 2.409909725189209, -1.2707124948501587, -4.476701259613037, -5.236104488372803, -0.47512221336364746, -4.029428958892822, 3.656639337539673, -0.6525851488113403, 6.670122146606445, -5.816606521606445, -8.170393943786621, -6.069986820220947, 0.20478439331054688, -3.883512258529663, 9.994892120361328, 6.2454352378845215, 6.659798622131348, 5.644711971282959, 2.6463327407836914, -8.024613380432129, 1.9327565431594849, 6.402573108673096, 3.2187530994415283, -0.30389612913131714, -2.338346242904663, 1.3068186044692993, 0.10785672068595886, -1.9980499744415283, -3.872138500213623, 12.493192672729492, -3.754105567932129, 2.0974926948547363, -4.195617198944092, -0.043038006871938705, -3.943798303604126, 2.269866943359375, -2.1370928287506104, -5.902876377105713, -0.7955707311630249, 0.07383431494235992, -4.925253868103027, 10.50757884979248, -9.952003479003906, -0.33999761939048767, -10.802566528320312, 1.4603919982910156, -5.918643474578857, 0.6137365698814392, 3.276437282562256, 1.1201820373535156, 1.0443342924118042, -4.757183074951172, 2.3068952560424805, -7.786762714385986, 4.168882846832275, -8.211960792541504, -0.6678643822669983, -9.482634544372559, 1.8932850360870361, -6.81020450592041, 3.3490211963653564, -2.5224759578704834, -3.758261203765869, 3.385653257369995, 4.591480255126953, 2.8922905921936035, 0.5851041674613953, -2.6068387031555176, -0.9574869871139526, 3.540404796600342, -6.522865295410156, 2.1610848903656006, 10.137327194213867, -9.438067436218262, 8.508150100708008, 1.2471257448196411, 0.9276853203773499, -3.254603624343872, 4.2419867515563965, 7.1605072021484375, -0.670481264591217, -2.22784686088562, 0.11634236574172974, 5.494442462921143, 2.6018784046173096, 1.3101152181625366, 8.538329124450684, -10.332242965698242, -5.148568153381348, 3.7622854709625244, -3.1808528900146484, -4.874963760375977, -7.911050796508789, -0.1703045517206192, -9.199044227600098, -5.690738677978516, -0.38533562421798706, -0.6590765714645386, -0.5144057273864746, -1.3143845796585083, -3.773439884185791, -1.00202214717865, 1.6183433532714844, -0.5609830021858215, 0.7460587620735168, 10.803604125976562, 3.6545283794403076, 9.368176460266113, -5.45478630065918, 5.666746616363525, 15.86228084564209, 2.7018158435821533, -17.418235778808594, -4.1776442527771, -1.8820571899414062, -7.890570640563965, -0.4158596098423004, 3.4599220752716064, 0.9297565221786499, -0.5437771677970886, -6.965482234954834, -6.449848175048828, -1.3568590879440308, 8.643845558166504, -0.8216598033905029, 1.3962405920028687, -2.5416805744171143, -1.3884919881820679, 3.3741157054901123, 10.597623825073242, -3.1504058837890625, 0.4410170018672943, -6.195828914642334, 0.36784374713897705, 4.62813138961792, -4.961103439331055, 1.5424003601074219, -3.1623761653900146, -5.146805286407471, -4.232203483581543, 7.851058483123779, 5.722301006317139, -0.47123849391937256, 3.9014084339141846, -6.4400153160095215, 5.2847795486450195, -0.40268030762672424, -3.6249186992645264, 1.4983731508255005, -4.6076531410217285, -1.7584316730499268, -2.3106184005737305, -6.173879146575928, -3.7857728004455566, -3.835075855255127, -8.243813514709473, -7.690016746520996, 1.239364504814148, -2.8290281295776367, 9.630073547363281, 2.148190498352051, -2.6859793663024902, -7.551577568054199, -3.1284914016723633, -0.11379191279411316, 8.836214065551758, 3.1721417903900146, 2.1885085105895996, -1.530249834060669, 5.212400913238525, 3.6035706996917725, -0.7476261258125305, -0.788725733757019, 1.8308943510055542, 4.96456241607666, -4.898171424865723, -2.924266815185547, 2.1464667320251465, 4.032341003417969, -0.9508055448532104, 2.7874226570129395, 12.058752059936523, 10.064866065979004, 4.0231733322143555, 2.94218111038208, -4.159183502197266, 3.1181654930114746, -5.276731014251709, 2.500154733657837, 1.1334120035171509, 7.955737113952637, -2.208238124847412, 1.4322192668914795, -1.9102842807769775, 4.352025508880615, 20.644990921020508, 7.5870585441589355, 3.625967264175415, -3.457143783569336, 3.172140121459961, -8.922825813293457, -8.354966163635254, 1.4949918985366821, 1.6716262102127075, 2.9861345291137695, -6.016100883483887, -0.481399804353714, -4.185055732727051, 0.7065613865852356, 4.005082130432129, 3.82147216796875, -3.0137734413146973, 5.075977325439453, 1.9749000072479248, 1.4294798374176025, -0.22740386426448822, -1.3664844036102295, 4.394017219543457, -4.631010055541992, 1.9106152057647705, 0.9765375852584839, -1.5267267227172852, -9.132490158081055, 1.049122929573059, 4.682186603546143, -2.188505172729492, -1.6422739028930664, -0.09155460447072983, -11.179533958435059, 2.0938498973846436, -6.1454243659973145, -0.668179452419281, -1.1559584140777588, -3.0546538829803467, 2.064774990081787, -1.6711782217025757, 1.300973653793335, -5.51448917388916, 14.038406372070312, 5.016665458679199, -6.301114082336426, -8.250288963317871, -6.391846179962158, 1.7893977165222168, -8.240100860595703, -3.371689558029175, 2.8503329753875732, -0.4318239688873291, -4.42338228225708, -1.797902226448059, 3.2979702949523926, 8.097611427307129, 1.5897005796432495, 5.487996578216553, 10.502016067504883, 1.4142489433288574, -1.6941874027252197, 6.98283052444458, -3.754441738128662, -3.8122642040252686, -3.4693520069122314, 0.7518319487571716, -5.045488357543945, 0.8109167814254761, -0.1433819979429245, 4.691427707672119, -2.5732274055480957, -4.309597015380859, -3.453483819961548, 12.231903076171875, -3.4309043884277344, 4.205662250518799, 1.4293317794799805, 1.2462072372436523, -9.532010078430176, -3.071988821029663, -5.322463035583496, -4.428004264831543, 0.8453555703163147, 1.024940013885498, -2.0467803478240967, 0.032180771231651306, 3.4066152572631836, -3.2030677795410156, 1.6014268398284912, -0.2257554531097412, -1.8087689876556396, -4.393264293670654, 2.281071901321411, -3.3718504905700684, -0.7476680874824524, -10.753190040588379, 8.487412452697754, -6.485629081726074, 5.201210021972656, -50.39559555053711, -1.8611364364624023, -4.385014057159424, 1.7067985534667969, -14.0480318069458, 8.80463695526123, -0.22433580458164215, 2.8956074714660645, 2.3860247135162354, 2.724318265914917, 9.384355545043945, 6.355164051055908, 3.437380790710449, -0.8810988068580627, 0.7939432859420776, -5.466229438781738, -5.250516414642334, 5.250451564788818, -1.733055591583252, 0.43709054589271545, 6.7256317138671875, 0.07616573572158813, 3.345623254776001, 4.039398193359375, 3.5651068687438965, -3.3014755249023438, -5.877788543701172, 0.7401096224784851, -2.192018747329712, 4.873026371002197, -4.356153964996338, 2.5985851287841797, 2.01979398727417, 6.394366264343262, -4.978244304656982, -5.512953281402588, 8.709543228149414, 4.191373825073242, -14.831755638122559, -8.086284637451172, -1.3587414026260376, -0.5457950830459595, 3.437100410461426, -1.9649534225463867, 0.226754292845726, 0.27024760842323303, 0.008815671317279339, -4.64288330078125, 4.20919942855835, -3.8965325355529785, -9.17677116394043, 2.206833839416504, -5.39592981338501, -4.5463032722473145, 8.095268249511719, -2.0815162658691406, 3.198282480239868, 0.6820778846740723, -1.819980502128601, 15.630290031433105, -7.233104228973389, -2.857393741607666, -3.710883378982544, 1.2197747230529785, -4.220200538635254, -2.511073112487793, -1.6021437644958496, -5.243414402008057, 1.040252685546875, -0.020958317443728447, -1.6697030067443848, -2.395336866378784, -1.4934264421463013, 0.4346180856227875, 2.641026020050049, -6.7275872230529785, 0.8595126867294312, -2.2245609760284424, 8.111482620239258, 7.751610279083252, 3.353997230529785, 4.990284442901611, 11.018826484680176, -4.036548137664795, 7.1125640869140625, 6.834766864776611, -5.482890605926514, 5.059986114501953, -3.1067283153533936, -4.265890598297119, 0.10039031505584717, -6.797568321228027, -5.106201171875, -0.7304503321647644, -1.594577670097351, 0.27497124671936035, -0.9653935432434082, -5.506191253662109, 6.674372673034668, -1.3133292198181152, -6.1323957443237305, 5.245416164398193, 9.925026893615723, 1.067468523979187, 3.447167158126831, -4.049703598022461, -3.227675199508667, -6.729423522949219, -3.1287550926208496, 2.6180684566497803, -4.469813823699951, 0.040485966950654984, 9.196474075317383, 5.375809669494629, -9.360231399536133, 7.115817546844482, 0.9661825299263, -3.2043910026550293, 9.65139389038086, -1.6615782976150513, -3.5481936931610107, -12.162712097167969, -2.2889864444732666, 1.8706939220428467, 1.3615524768829346, -3.4616050720214844, 5.586165428161621, -6.587236404418945, -0.7145707607269287, 1.4877238273620605, 6.618856430053711, 5.189603328704834, 0.0432039238512516, 3.9167935848236084, -5.057255744934082, -2.314401626586914, -2.1320290565490723, -6.403012752532959, 5.246361255645752, -9.37537670135498, 4.671031475067139, -4.665817737579346, -2.6693949699401855, -6.68015718460083, 6.196184158325195, -3.403022289276123, -8.374082565307617, -0.8034963607788086, 3.7290451526641846, 7.642641544342041, -0.5301600098609924, -1.1053388118743896, 1.8689755201339722, 5.021490097045898, 4.826700687408447, 0.5460107922554016, 0.22729194164276123, -0.0524778887629509, -1.3071244955062866, 5.450984954833984, 3.0668625831604004, -0.6776264905929565, -2.5305123329162598, -8.688786506652832, 0.7181348204612732, 3.059816360473633, -6.416323184967041, -9.408897399902344, 4.032063007354736, 3.2639670372009277, 2.8928449153900146, -0.2216288149356842, 8.028188705444336, -0.5952933430671692, -2.1168792247772217, 8.195563316345215, 1.5635809898376465, 2.2110211849212646, -17.273174285888672, -6.111233234405518, 5.456586837768555, 2.099575996398926, -3.4481709003448486, 5.962944984436035, 0.9762804508209229, -6.496968746185303, -2.966095447540283, 6.04409122467041, 3.941970109939575, -4.712640762329102, -3.571863889694214, 1.0476582050323486, -4.853447914123535, 1.0201780796051025, 2.675300359725952, 0.565985381603241, -1.4856972694396973, -8.56238079071045, -1.9795851707458496, 9.013973236083984, -4.339212417602539, 0.7808768153190613, 2.830920934677124, 9.334824562072754, -6.345805644989014, 2.3710644245147705, 3.283552408218384, 5.829126834869385, 2.494452953338623, 8.49903678894043, 5.87959623336792, -5.611004829406738, -1.379434585571289, 12.390323638916016, -0.3588564097881317, 2.846146583557129, -1.5469721555709839, -3.0785465240478516, 8.222149848937988, -6.413126468658447, -0.810066282749176, -3.6063642501831055, 1.2518819570541382, 10.209441184997559, 8.039594650268555, -8.12662410736084, 7.207838535308838, 1.090761423110962, 4.799162864685059, 11.264665603637695, -2.3818697929382324, 3.8699028491973877, 4.655203819274902, 4.258383750915527, 8.260671615600586, 0.8657758831977844, 0.30517759919166565, -1.3970609903335571, 6.3616437911987305, -1.743674397468567, 5.7575788497924805, -6.243075847625732, 4.627110958099365, -7.308181285858154, -1.1093366146087646, -11.586119651794434, -5.392889976501465, 8.776352882385254, 6.056135654449463, 10.73274040222168, 10.357646942138672, 4.543933391571045, -1.5084160566329956, 0.9037454724311829, 4.441878318786621, -10.158010482788086, 0.7839778065681458, -3.211334705352783, -4.48471736907959, -12.828516006469727, -2.584995985031128, -1.1277742385864258, 3.1762616634368896, 0.8757278919219971, 8.612616539001465, -2.840618133544922, 6.878479957580566, 0.24431446194648743, 8.598444938659668, 7.693471908569336, -4.6813459396362305, -3.2918708324432373, 8.5187349319458, -3.597813844680786, -2.5233352184295654, -2.7061030864715576, 4.626462936401367, 6.229581356048584, 6.671051025390625, -4.280477523803711, 4.702131271362305, -9.70506763458252, -7.726567268371582, 7.077037334442139, -4.818583011627197, 5.81066370010376, -0.459877610206604, 11.263272285461426, -2.163257122039795, 1.035543441772461, -1.6053889989852905, 0.8403013944625854, 16.293794631958008, -0.8247360587120056, 4.853120803833008, -2.3917062282562256, 0.39911285042762756, 9.122088432312012, 4.0958251953125, -3.397883176803589, -0.9234387278556824, -2.017648458480835, -0.5596060752868652, 2.1642799377441406, -9.174612998962402, -16.472476959228516, -6.091018199920654, 4.294957637786865, -0.021814638748764992, 0.7643173933029175, -3.7305548191070557, -3.9471843242645264, -12.917428970336914, 5.71159553527832, -3.4683241844177246, -7.181863784790039, 10.735261917114258, -0.23179706931114197, 0.0053764572367072105, 1.4335949420928955, -2.8684239387512207, 0.3817792534828186, -4.737791538238525, -2.0997674465179443, -2.7829229831695557, -10.568477630615234, 1.8407320976257324, 7.312172889709473, -0.6422687768936157, -1.47006356716156, -15.141032218933105, -0.2385665327310562, 3.6766419410705566, 3.0597729682922363, 0.9391537308692932, 0.6974042057991028, -4.0496673583984375, -1.1157469749450684, 5.028867721557617, -7.055159568786621, 7.689267635345459, 2.7811803817749023, -4.069398403167725, -0.19077950716018677, -3.380105495452881, 6.640451431274414, 2.145159959793091, 1.9318867921829224, -3.7375524044036865, -2.594303607940674, -7.266831398010254, -2.820188045501709, -3.901918888092041, 11.799714088439941, 1.9998455047607422, 4.371318817138672, -0.6957855820655823, -1.1845163106918335, -3.6857669353485107, 4.853773593902588, 4.87429141998291, -6.932374000549316, 1.3967738151550293, 2.654165029525757, -10.042604446411133, -0.19152244925498962, 0.8499866127967834, 1.1401137113571167, 3.138911247253418, 1.427236795425415, -3.6427602767944336, -1.632159948348999, 5.538997173309326, -6.693646430969238, -3.4766998291015625, -4.3802571296691895, 1.6260584592819214, 2.992494821548462, -0.8850663304328918, -7.159111976623535, 3.603266477584839, -0.05856555327773094, 0.8920882940292358, 3.358023166656494, 5.636850833892822, 4.369193077087402, -2.2420854568481445, 6.323705673217773, -1.6479555368423462, -3.776304006576538, -1.7622076272964478, 1.8845553398132324, 3.0685060024261475, 7.906795978546143, 1.2496992349624634, -6.608702182769775, 1.9257532358169556, -8.001249313354492, 0.03547840192914009, -0.3617497980594635, -5.616487503051758, 6.380889892578125, -2.5485641956329346, 3.731358051300049, -0.915494978427887, -5.847352504730225, -2.465024709701538, 0.6532320976257324, 1.876543641090393, -4.140486717224121, -2.34977388381958, 4.368846416473389, -16.297813415527344, 3.8586556911468506, -3.5735483169555664, -2.7850353717803955, -5.5230631828308105, -5.209420204162598, -4.724467754364014, 0.6825169920921326, 4.248857498168945, 2.113887310028076, 3.0559074878692627, 1.2197513580322266, 1.1684284210205078, -1.7202633619308472, -2.7593133449554443, 5.0498199462890625, -2.464179277420044, -4.116235733032227, -5.632472515106201, 7.43614387512207, -3.4178614616394043, 5.615418910980225, 9.862204551696777, 6.072545528411865, -8.854805946350098, 1.471773624420166, -6.020588397979736, -3.2152955532073975, -0.21383044123649597, 0.5253077149391174, -7.7963457107543945, -3.269871711730957, -7.404489994049072, -4.38320255279541, 9.247941017150879, -10.71022891998291, -2.5878920555114746, -3.765536069869995, -11.519434928894043, 14.389220237731934, -3.2563071250915527, 2.183866500854492, -8.111735343933105, -2.6646413803100586, -8.948179244995117, -2.956871271133423, -1.3021507263183594, 7.3828349113464355, -0.638131856918335, 0.7462123036384583, -3.789818286895752, 2.5831873416900635, 0.5690273642539978, 1.1450204849243164, -5.094998836517334, 4.647975444793701, -0.5499288439750671, 3.75002384185791, 2.237236499786377, 5.379354953765869, 2.9279770851135254, -2.1625709533691406, 7.99437141418457, -0.7058091759681702, -3.3080334663391113, 5.938076496124268, 1.2055563926696777, -0.9073737263679504, -2.5461928844451904, 9.474505424499512, -3.4904685020446777, -5.941259860992432, 2.4480159282684326, 3.458817481994629, 6.266468048095703, 4.441790580749512, 11.184127807617188, -1.6042320728302002, -1.7559009790420532, -5.842620372772217, -3.594327449798584, 5.332038402557373, -3.488870620727539, -0.9994077682495117, -4.423095703125, 5.316005229949951, 5.207515716552734, 0.4550359845161438, 8.44304370880127, -1.4262728691101074, 3.136404514312744, 4.055663108825684, -3.9927830696105957, 5.962771892547607, 2.156099319458008, 4.594798564910889, 4.374292850494385, 6.939168453216553, -3.763059377670288, 5.479478359222412, 7.344764709472656, -5.773427963256836, 7.447136878967285, 1.335779070854187, -6.629695892333984, -1.526902198791504, 1.7408125400543213, -3.7176358699798584, -1.922669529914856, -6.099574565887451, 0.9632661938667297, 3.696413516998291, 7.259555339813232, 0.18033966422080994, -3.6566691398620605, -4.312819004058838, -11.0464448928833, 11.592391014099121, -3.0460221767425537, 38.131553649902344, 2.552044153213501, -1.6779675483703613, -10.285417556762695, 11.283995628356934, -0.5683600902557373, -0.8576859831809998, 7.861790180206299, 5.648802280426025, -0.9047216773033142, 1.5293803215026855, -11.146261215209961, 11.07032299041748, -2.1849722862243652, -2.026510000228882, 2.5142972469329834, -5.711244106292725, -5.316822052001953, -10.465091705322266, -10.151570320129395, 3.2560999393463135, 4.792618751525879, -0.30362656712532043, 5.933243751525879, 7.6887030601501465, -5.323581695556641, -0.02319510281085968, 4.297220230102539, 1.435068964958191, 5.134167194366455, -5.259382724761963, 0.642574667930603, -6.303516864776611, -1.3488943576812744, -1.0013750791549683, -6.995568752288818, -0.03778087720274925, -3.2821362018585205, -2.376772165298462, 2.5663232803344727, 0.7014374136924744, 3.066807985305786, 1.5534332990646362, -9.13611125946045, 0.7968536615371704, 0.5469099283218384, -3.4079790115356445, -4.988870143890381, 0.7284576296806335, 2.842536211013794, 8.71204662322998, 0.3967363238334656, 6.572936058044434, 11.129045486450195, 7.4290313720703125, -4.372527122497559, -5.836600303649902, -7.544383525848389, -0.2898513972759247, -0.04413691908121109, -9.67694091796875, -5.88026762008667, 2.0260605812072754, -2.0771336555480957, -3.018754720687866, -1.8317652940750122, -2.419501304626465, 0.9917590022087097, -3.2873804569244385, -10.012910842895508, -2.849943161010742, -2.1878504753112793, 5.364098072052002, -3.833986520767212, 4.477833271026611, -1.58475661277771, -6.809815883636475, 6.325313091278076, 4.5082173347473145, 5.5517778396606445, 2.610930919647217, 1.451372742652893, -10.485095024108887, 6.351578712463379, 0.9694556593894958, -16.598464965820312, -8.423620223999023, 0.16632448136806488, -0.5633096098899841, 2.3413121700286865, 5.018345355987549, -10.378846168518066, 8.083396911621094, -3.1775619983673096, -4.5143561363220215, -1.044906735420227, -4.889166355133057, -2.9501681327819824, 8.997904777526855, -6.90947151184082, 0.8127343654632568, 6.252378940582275, -6.848381042480469, 1.0148546695709229, -0.9081453680992126, -9.481661796569824, -6.903931140899658, 1.8696258068084717, -1.3153878450393677, 3.2058372497558594, -6.750938892364502, 1.6113361120224, 0.3700832724571228, 2.4995830059051514, 4.678071022033691, -4.962343215942383, -5.694946765899658, -0.11641000956296921, 1.2832366228103638, -0.11122256517410278, 0.2038324475288391, 1.0938262939453125, -1.343612790107727, 7.425368785858154, -3.993299722671509, -1.708677887916565, 6.67283821105957, -5.632223129272461, 1.5160712003707886, 2.4032838344573975, -2.317009925842285, 4.457427501678467, -11.916106224060059, -5.0574564933776855, 2.0177993774414062, 4.064550876617432, 2.7939999103546143, -10.598304748535156, 2.1233816146850586, -1.5392091274261475, -4.222309589385986, 5.92805814743042, -2.215517997741699, -4.575994491577148, -10.685018539428711, 2.5738649368286133, -0.3634846806526184, 12.796011924743652, 0.5975685119628906, 4.856284141540527, 1.2643400430679321, -1.9055005311965942, -3.4999165534973145, -6.497641086578369, -4.774866580963135, -2.4492056369781494, -2.7457115650177, -4.676731109619141, -5.058680534362793, -6.858032703399658, -1.826145052909851, -4.368442535400391, -3.965538740158081, -2.1970436573028564, -2.9756557941436768, -6.4446797370910645, 7.708182334899902, -3.016063928604126, 6.12534236907959, 2.5235092639923096, -4.188694000244141, 2.12865948677063, -2.597777843475342, -9.638632774353027, 2.6719276905059814, -3.5339808464050293, 6.597496032714844, 4.618934154510498, -2.301093578338623, -11.912928581237793, 3.2344419956207275, -3.33054518699646, 2.041496753692627, 7.180793762207031, 2.068643569946289, -6.0990777015686035, -1.255948543548584, -3.641489267349243, -0.08793286979198456, 0.7780286073684692, 0.3433116376399994, -1.611122965812683, 8.328953742980957, 3.1096549034118652, 2.063830614089966, -1.4050906896591187, -1.3568142652511597, 0.17183756828308105, -6.809730529785156, -6.986851692199707, -7.241379261016846, 7.436687469482422, 4.284681797027588, 3.478104829788208, 3.71480131149292, 3.2293310165405273, 2.194636106491089, -14.567084312438965, 8.714275360107422, 1.5396944284439087, 7.329025745391846, -8.303911209106445, 7.5208659172058105, 19.178794860839844, -4.052966594696045, -8.415514945983887, 8.200698852539062, -4.000598907470703, 9.207247734069824, 0.6497798562049866, 1.2976738214492798, 0.862798810005188, -2.3619446754455566, -9.657064437866211, 0.05910506844520569, -1.9015803337097168, -0.4910711646080017, 4.39625358581543, -6.6996636390686035, -1.092159628868103, 0.9824756383895874, 3.9317827224731445, -1.3996825218200684, 3.744314432144165, -1.3585740327835083, -0.5090270638465881, 2.893491268157959, 6.011658668518066, -5.077487468719482, -2.1222548484802246, -1.7379471063613892, 8.901634216308594, -0.14504274725914001, -3.484745740890503, 7.8189802169799805, -0.09298243373632431, -8.20407772064209, 6.227977275848389, -0.24980361759662628, -2.199690580368042, 3.719003677368164, -3.4898061752319336, 7.372499942779541, -6.272371292114258, 5.988371849060059, -11.53327751159668, -5.6973490715026855, 4.292196750640869, -5.959405899047852, 6.492128372192383, -6.065316200256348, -1.3460944890975952, 1.3490283489227295, -1.8500537872314453, 1.7290973663330078, 2.606149673461914, -3.6862144470214844, 5.542099475860596, 0.8413124084472656, -1.5826821327209473, -0.9794104695320129, 1.8472983837127686, 4.994433879852295, -2.894390106201172, 4.729425430297852, -2.421959161758423, 0.8454520106315613, -0.4758504629135132, -3.3451039791107178, 5.88836145401001, 2.4800493717193604, -8.721858978271484, 2.8682847023010254, 6.14642333984375, -5.116739273071289, -4.039568901062012, -7.74163293838501, 5.47334098815918, -10.118821144104004, -3.0438497066497803, -1.4885754585266113, 0.35800352692604065, 1.5287566184997559, -3.1298491954803467, -6.958666801452637, -0.44938620924949646, 6.360688209533691, -5.129157066345215, 3.9258856773376465, -5.946986675262451, 4.432071208953857, 1.3419421911239624, 5.763850688934326, 0.216939777135849, 3.0707812309265137, -5.774659633636475, 10.412220001220703, 14.804413795471191, 0.3332963287830353, -8.055782318115234, 2.966651439666748, -12.658102035522461, -0.20831207931041718, -12.14128303527832, 6.158281326293945, -8.314249038696289, -14.274565696716309, 8.190526962280273, -5.218202590942383, -13.142735481262207, -2.836146831512451, -3.602663278579712, 7.8567633628845215, -2.61789870262146, -5.123453617095947, -4.284894943237305, 0.5813682079315186, 5.5172529220581055, 3.621575355529785, -5.637434005737305, -0.5700353384017944, -9.746352195739746, 10.016631126403809, -6.906772613525391, -2.6982779502868652, 3.175290822982788, -5.565893173217773, 7.756863117218018, -3.8041162490844727, 8.848430633544922, 1.5564990043640137, -10.34051513671875, -2.0971081256866455, -3.6976633071899414, 13.579538345336914, -13.310388565063477, -5.585142612457275, -7.02730655670166, 0.2963941991329193, 4.722808361053467, 0.7184737324714661, -1.3375964164733887, 5.159842014312744, 2.617418050765991, -0.6379887461662292, 1.7792994976043701, -4.370608329772949, -4.808828830718994, 22.450353622436523, -17.334224700927734, -0.6478798389434814, 4.36275053024292, 0.7285542488098145, -9.015037536621094, -1.208526372909546, 11.629056930541992, 2.3950002193450928, 1.583113431930542, -3.375511884689331, 3.8133716583251953, -4.410006046295166, 5.5056633949279785, 1.1863107681274414, 2.2433533668518066, -4.766380310058594, -4.273431777954102, 4.8396315574646, 10.24317455291748, 4.176187992095947, 4.079042911529541, -3.992231607437134, 3.6629903316497803, 8.409290313720703, 2.4397878646850586, 3.484492540359497, -3.778050422668457, 9.672240257263184, -4.370842456817627, -2.060602903366089, 2.277407646179199, 1.7405868768692017, -1.0415116548538208, 0.6638882756233215, 1.6949535608291626, -1.2813431024551392, 0.8754168748855591, -0.9317067265510559, 1.9055547714233398, -8.969986915588379, -9.380806922912598, -9.639788627624512, 3.254124164581299, 0.14599740505218506, 9.500332832336426, -7.444479465484619, 9.345562934875488, 4.1402130126953125, -7.168846130371094, -6.219208717346191, 0.06660343706607819, 4.230428695678711, 1.0086039304733276, -2.7104883193969727, 0.49598178267478943, 10.621015548706055, -5.881612777709961, 1.458223819732666, -3.9424240589141846, -6.9808173179626465, -8.770352363586426, 4.021702289581299, 0.45826631784439087, 6.186369895935059, -2.5654211044311523, -0.32170936465263367, -2.0731401443481445, 4.943909645080566, -0.9661452174186707, -5.3742170333862305, -5.550971984863281, 1.239045262336731, -0.6957463622093201, -4.486244201660156, -1.9600391387939453, 2.1464924812316895, 7.079081058502197, -4.258333683013916, 2.8796987533569336, -1.24082612991333, -1.7624180316925049, -5.33553409576416, -4.762716770172119, 3.257713556289673, -4.518344879150391, 4.1227006912231445, 13.904265403747559, -4.912089824676514, 4.558075904846191, 10.004667282104492, 6.873373985290527, -2.057128667831421, 0.7881653904914856, 0.8139739036560059, -2.6189260482788086, -1.3633586168289185, -9.006875038146973, 1.1748746633529663, 1.8903014659881592, 5.117489814758301, -3.8009278774261475, -2.4551198482513428, 7.071656703948975, 1.8452321290969849, 6.3570990562438965, 1.4095063209533691, -3.6739532947540283, 7.869414806365967, 3.8312861919403076, -1.6843974590301514, 5.014459609985352, 1.3915060758590698, 0.13800093531608582, 0.48394256830215454, 2.0356130599975586, 3.310741901397705, -3.0960538387298584, 3.861546039581299, 3.733116388320923, -3.8717894554138184, -2.146306276321411, 0.18541058897972107, 11.770766258239746, 9.580862045288086, 1.1477612257003784, -0.3913283348083496, -5.642099857330322, -2.686527967453003, -1.608290195465088, -1.345474123954773, 0.9300493597984314, 3.850555658340454, -3.7305684089660645, -2.6029794216156006, -4.6004791259765625, -4.430570125579834, 4.696290016174316, -5.954362392425537, -0.8071942329406738, 3.5097062587738037, -2.1790552139282227, 7.052834987640381, 9.047429084777832, -1.7401503324508667, -7.580726146697998, 3.3047564029693604, -9.020546913146973, -1.3556060791015625, 2.2928335666656494, -2.219799041748047, 5.51684045791626, -0.29230108857154846, 1.3286024332046509, -17.69904899597168, -2.186868667602539, -0.0713895708322525, 3.074852705001831, 2.0081398487091064, 5.263954162597656, -7.5581769943237305, 0.2640308737754822, -1.968738079071045, 2.995889186859131, -2.859492778778076, -7.695080757141113, 0.9158164262771606, 1.7638434171676636, -3.9663217067718506, 1.351850986480713, 3.24483585357666, -2.012460947036743, 3.8505160808563232, -0.4462963938713074, -7.254281044006348, -7.224381923675537, 1.865036964416504, 0.2757194936275482, -2.914334774017334, -5.929925918579102, 3.6124284267425537, 10.717086791992188, -2.8086676597595215, 9.549051284790039, -2.774702310562134, -0.9334283471107483, 12.069696426391602, 8.174936294555664, -0.6681568622589111, 10.030679702758789, -5.435582637786865, -2.308053731918335, 0.09988108277320862, -1.3262797594070435, 1.1913142204284668, -10.033819198608398, 0.44543468952178955, -3.4645347595214844, -3.6483988761901855, -0.5152750611305237, 1.4407702684402466, -2.1009182929992676, 3.026599645614624, 1.8558162450790405, 3.677381753921509, 12.199577331542969, -1.827062964439392, -1.7524049282073975, -5.869747161865234, 3.1488747596740723, -11.714765548706055, -7.535807132720947, -1.1312873363494873, 2.224990129470825, 2.8834943771362305, -7.7858991622924805, 2.7129828929901123, -1.3894683122634888, 0.12132292985916138, 0.3695107400417328, -3.7707064151763916, -0.1062912717461586, -1.4621926546096802, -3.7416489124298096, -4.515954494476318, -6.069058895111084, 0.5452741384506226, 4.332067966461182, 6.806014060974121, 15.499953269958496, 4.295194625854492, -8.103191375732422, -0.9651873111724854, 6.412024974822998, 6.566161155700684, -6.346358299255371, 1.1360832452774048, 6.654112339019775, 7.171239376068115, -0.7912784814834595, -0.19222372770309448, 5.863457679748535, 4.730157852172852, -3.3979175090789795, 4.98063325881958, 3.1243255138397217, 0.8664831519126892, 0.4265238046646118, 0.22343029081821442, 5.452528476715088, -6.889734268188477, 3.1932318210601807, 2.2205002307891846, -7.495607852935791, -4.583838939666748, -5.871882915496826, -2.9853038787841797, -0.41468170285224915, 1.2081446647644043, -1.0227478742599487, 1.671204686164856, -10.262948036193848, 8.396931648254395, -1.3833297491073608, 5.982661724090576, 5.180063724517822, 4.882195949554443, 4.394138813018799, -0.4942837059497833, -4.6798248291015625, 2.632232904434204, 7.227107048034668, 3.236429214477539, -2.573153257369995, 2.5895819664001465, 2.9507603645324707, -5.717448711395264, 6.7679901123046875, 8.48302936553955, 2.333402633666992, 2.516472578048706, 3.7140705585479736, 2.465036630630493, 6.35284423828125, -3.7296290397644043, -7.98854923248291, -0.6056578755378723, -1.6079387664794922, 3.2970163822174072, 3.0922813415527344, 0.39524826407432556, 3.2671194076538086, -4.200220584869385, 10.956306457519531, 5.695122241973877, 4.877112865447998, 4.990159034729004, -1.461808443069458, 5.940384387969971, 3.36639142036438, -3.8987507820129395, -10.994096755981445, 1.4453943967819214, -27.23230743408203, -11.186095237731934, 2.547839879989624, -0.13365426659584045, 2.6617982387542725, 0.536538302898407, -5.423311233520508, 1.591789722442627, -1.0787855386734009, -3.5244061946868896, -6.677781105041504, -7.301988124847412, 9.95155143737793, -1.1383801698684692, -4.676675796508789, -4.942988395690918, 2.967458486557007, 5.445486068725586, -3.2502057552337646, 6.633937835693359, -3.480347156524658, -6.027463912963867, 9.668591499328613, -0.8872227668762207, -0.7384358644485474, -1.193902850151062, 0.673936128616333, -0.7770046591758728, -1.8126555681228638, -0.15000145137310028, -0.04482724145054817, 0.5078845620155334, -1.9122815132141113, -2.6987922191619873, 0.7667289972305298, 5.531466960906982, 2.040853500366211, -9.248018264770508, 7.3128533363342285, 1.0782461166381836, 1.4889079332351685, 2.0479788780212402, 3.1125833988189697, 2.614135980606079, 7.512434959411621, -2.502997875213623, -6.316851615905762, 4.269517421722412, -2.9300427436828613, 0.6075987219810486, 6.427131652832031, 12.477616310119629, -2.374575614929199, -0.5888037085533142, -0.4628638029098511, 5.912943363189697, -8.899547576904297, 6.82258939743042, 14.91883373260498, -1.339408040046692, 6.081036567687988, 4.39409875869751, -4.351089000701904, -3.2261736392974854, -8.612123489379883, -1.8520042896270752, 2.7143359184265137, 2.1111443042755127, -0.10640641301870346, 12.900793075561523, 5.143711566925049, 0.5094690322875977, -0.29715216159820557, 7.218055725097656, 0.259728342294693, -4.150743007659912, -9.868999481201172, 11.962060928344727, 7.007699489593506, 0.7620428800582886, 10.28177261352539, 6.864035606384277, -5.641931533813477, 0.331670343875885, -1.9383002519607544, 10.803696632385254, -3.946556329727173, -6.195074558258057, 4.593499183654785, -10.431343078613281, -4.926768779754639, -9.411314010620117, -2.916992425918579, 3.4132330417633057, 0.5771741271018982, 2.1504995822906494, -0.134393572807312, 2.1678311824798584, 9.502531051635742, 4.504876136779785, -0.9379554390907288, -8.466510772705078, -3.213083505630493, 5.926571369171143, 0.7584716081619263, -2.913454055786133, 7.346468925476074, 4.274440765380859, -8.491048812866211, 0.9770143628120422, 1.343937873840332, 13.876463890075684, -1.9972316026687622, 2.37562894821167, -1.2523881196975708, 2.567559242248535, -4.170769691467285, 2.4130396842956543, -4.942344665527344, -0.7250237464904785, -5.8810014724731445, 1.5528074502944946, 0.43089014291763306, -3.689021348953247, -4.437033176422119, -6.492863178253174, -8.474010467529297, 1.2642083168029785, -3.8745357990264893, 1.7919214963912964, -2.2100989818573, -3.1891567707061768, -5.3473029136657715, 2.3361310958862305, -1.317252516746521, -4.42912483215332, -5.926272392272949, -1.927207589149475, 12.182745933532715, -1.4908229112625122, -13.192572593688965, 4.700211048126221, 5.068302631378174, 3.311861515045166, 2.3851749897003174, 4.206811428070068, 4.462112903594971, 2.5990946292877197, 4.643074035644531, 0.3834090530872345, -2.583195924758911, -2.9972331523895264, 0.8405271768569946, 0.47973427176475525, 3.2739925384521484, -10.2953519821167, -4.793210506439209, -2.9452643394470215, 2.622985601425171, 10.514643669128418, 1.4764235019683838, -7.454464912414551, -5.03499698638916, -0.30315548181533813, 8.38616943359375, 6.589605331420898, 0.17070510983467102, 2.6858882904052734, 1.9778133630752563, -3.496776819229126, 13.154176712036133, 0.42651069164276123, 7.483608245849609, -2.132404327392578, 7.315162658691406, 9.634343147277832, -0.5890463590621948, 0.31085264682769775, -9.680066108703613, -0.6903921365737915, 2.2970094680786133, 4.838977813720703, -14.049896240234375, -2.4720382690429688, 0.25677940249443054, -7.137359619140625, 7.362846374511719, -2.578260898590088, 2.395252227783203, -1.4841564893722534, 2.5493905544281006, 2.935696601867676, -8.823208808898926, 0.5372560620307922, -6.612060546875, -4.862290859222412, 5.168783664703369, -1.0568031072616577, -5.1547136306762695, -2.569578170776367, 4.819843292236328, -2.2241125106811523, 2.070765495300293, 2.7651383876800537, 10.990682601928711, 1.6309514045715332, 0.07948566228151321, 5.155060768127441, -0.9191169142723083, -3.4878604412078857, -7.302367687225342, -5.460019588470459, -8.609159469604492, 1.5797194242477417, 2.2990505695343018, -1.1615039110183716, -4.869813442230225, -12.232489585876465, 2.6447787284851074, -2.7865779399871826, 1.7449042797088623, 0.7993380427360535, 0.3395232856273651, -4.795962810516357, -3.1228034496307373, -2.2561938762664795, 9.252362251281738, 6.861185550689697, -0.2551722228527069, 6.33748722076416, -3.266408681869507, -4.812090873718262, -8.849515914916992, 9.14822769165039, 7.626561164855957, 1.9559131860733032, 5.52896785736084, 3.3163042068481445, 3.50978422164917, 0.0004372740804683417, -4.03203010559082, 0.0015326819848269224, 0.9943719506263733, 0.08726928383111954, 4.484925746917725, 1.1630383729934692, 2.8807802200317383, -3.831420660018921, 5.920440673828125, -0.17411702871322632, 9.326719284057617, 4.426858901977539, -2.719102144241333, 1.283660650253296, -2.8762929439544678, -6.889167785644531, 3.187249183654785, -2.7556424140930176, 3.9697763919830322, 3.2697501182556152, 2.755258321762085, -0.35834720730781555, -8.125707626342773, 1.4223616123199463, 6.515394687652588, -4.279191970825195, 6.979538440704346, -6.4367194175720215, -2.1400909423828125, -3.5682215690612793, -7.466345310211182, -6.796921730041504, -0.8874156475067139, -3.95837140083313, -7.34474515914917, 3.2462947368621826, -6.213314056396484, -10.99361801147461, -2.1499695777893066, 0.6755049228668213, 5.135923385620117, 1.9503071308135986, -5.214298248291016, -1.9178266525268555, 3.621047019958496, 1.8369332551956177, -4.878200531005859, -2.584083080291748, 1.0342473983764648, 2.9564361572265625, -3.2878634929656982, 9.841550827026367, -0.8809307217597961, 1.6290028095245361, -1.05143404006958, -31.20798110961914, -0.10052969306707382, -2.098829507827759, -14.737350463867188, -0.973476231098175, 5.133737087249756, 4.019116401672363, -0.7278528809547424, 4.047523021697998, 3.6400768756866455, -2.293102264404297, 0.46871891617774963, -4.2683258056640625, -3.7956976890563965, 1.3545043468475342, -3.166585922241211, -1.3856555223464966, 0.28673064708709717, 1.602644681930542, -1.7703781127929688, 7.821396350860596, 4.777087211608887, -7.426138877868652, -4.891678333282471, 2.2155444622039795, -4.9848713874816895, -4.248928070068359, -8.9878511428833, 5.467642307281494, 7.421178340911865, 1.1411017179489136, -0.6139503121376038, 2.2086620330810547, -2.904916763305664, 0.7064898610115051, -6.643998622894287, 2.767721652984619, -1.4821016788482666, -6.858800888061523, -9.177563667297363, -0.31090497970581055, -1.2032558917999268, -5.563230514526367, 3.319352626800537, -17.050050735473633, -5.607141494750977, 6.488053321838379, -0.4195898771286011, 2.4171524047851562, 7.103209495544434, 1.521128535270691, -7.41794490814209, -6.441903591156006, -0.29356610774993896, 0.08271326124668121, -4.948236465454102, -3.18442702293396, 5.095231533050537, -5.186716079711914, 6.217895984649658, -8.339406967163086, 1.684022068977356, 7.027273178100586, -5.780284881591797, -6.145380973815918, -0.6515696048736572, 0.651461124420166, 6.692048072814941, -4.832621097564697, 3.4286203384399414, 1.421154499053955, 0.08948501199483871, -3.4149258136749268, 0.3145088851451874, -0.053856633603572845, 4.4556474685668945, 1.5012151002883911, 3.6588456630706787, 1.8544185161590576, 2.0835940837860107, 3.4781606197357178, 4.926372051239014, -1.0727455615997314, 7.153402805328369, 13.625787734985352, 0.25835636258125305, -8.712910652160645, -1.1618130207061768, -0.9078143835067749, 0.5625174045562744, 2.7816524505615234, 1.8776166439056396, 2.5301265716552734, 1.3940647840499878, 2.293191432952881, 8.704676628112793, -1.3385862112045288, 2.735745668411255, -4.6731133460998535, -0.9444535970687866, 2.2155532836914062, 1.4261701107025146, 8.490647315979004, 7.663692474365234, 6.301230430603027, 0.11328216642141342, -0.3592629134654999, 3.173236608505249, 6.453849792480469, -5.935978412628174, -1.3169294595718384, 3.2377727031707764, -8.430495262145996, 0.19735577702522278, -6.197905540466309, -2.7376606464385986, -2.1789517402648926, 1.5682809352874756, -0.2221555858850479, 3.531125068664551, 2.8646180629730225, -10.116217613220215, -5.3925065994262695, -5.5232391357421875, 2.2504994869232178, -4.905482769012451, -3.4202566146850586, -4.37996768951416, -4.775497913360596, 0.352941632270813, -3.05092716217041, -6.660358428955078, -0.920547604560852, 3.711869239807129, -2.9426417350769043, 3.8946471214294434, -9.311487197875977, 2.4901044368743896, -4.351028919219971, 6.280261993408203, -0.8246848583221436, -3.5738039016723633, 8.263786315917969, 3.824629783630371, -3.6872036457061768, -1.5693687200546265, -4.109642028808594, 5.603827953338623, 4.16615104675293, -4.472997665405273, -1.2762542963027954, 2.6864070892333984, -4.092817306518555, 4.039594650268555, -6.00173282623291, -12.603036880493164, -1.4219098091125488, 8.417047500610352, -2.0105667114257812, 8.01706314086914, 2.714524030685425, 6.2379045486450195, 1.2774378061294556, -2.0547444820404053, -1.3206485509872437, 3.14121675491333, 0.5154063701629639, -9.156267166137695, 2.845766305923462, -0.3079116940498352, 0.4062824845314026, 3.221708059310913, -3.114492654800415, -1.9664186239242554, 3.005056858062744, 2.097703218460083, 11.270328521728516, 0.30917972326278687, -2.1760964393615723, 0.7613063454627991, 6.306088924407959, 8.055379867553711, 5.820364094688557e-05, -0.4038930833339691, 4.545173168182373, -0.15848249197006226, -5.507124423980713, 0.0030146350618451834, 0.7883136868476868, -1.1097396612167358, 0.3535059988498688, -0.2360798567533493, -2.4819605350494385, 2.5697431564331055, -5.554133415222168, -3.944786787033081, 3.034756660461426, 2.3343605995178223, 5.5047173500061035, -2.878103017807007, -10.251089096069336, -5.5851030349731445, 15.407842636108398, 0.9743953347206116, -3.4701099395751953, -0.37493765354156494, -1.468552589416504, -2.5184826850891113, 4.51926326751709, -1.8836241960525513, 1.4511418342590332, 4.296201705932617, 5.352572441101074, 1.0297694206237793, -11.473718643188477, 1.4396905899047852, 6.771527290344238, -4.292426586151123, 3.0289688110351562, 5.517069339752197, 9.843515396118164, 5.620119094848633, 4.519924640655518, -7.576880931854248, 4.9008636474609375, -0.7086119055747986, -3.6241564750671387, 6.674108982086182, -3.8292429447174072, -3.0340182781219482, -8.545395851135254, 0.8500639796257019, 0.6298248767852783, 2.6351802349090576, 4.756223678588867, 0.3068133294582367, -9.269274711608887, -1.2707173824310303, -4.315406322479248, -1.270329236984253, -0.7624404430389404, -1.5585414171218872, -8.273663520812988, -13.299798011779785, 2.922539710998535, -1.7966787815093994, -4.537538528442383, -7.131703853607178, -7.734290599822998, 3.401962995529175, -2.619680166244507, -4.9699296951293945, 0.844380795955658, -5.879693508148193, -3.9842302799224854, 8.007399559020996, 3.454111337661743, 4.338606357574463, -0.3424825668334961, 0.6865366101264954, -6.927158355712891, 6.619456768035889, 4.207339286804199, 4.727344512939453, 8.322007179260254, -4.238423824310303, 1.0720241069793701, -2.179088830947876, 3.108870029449463, 0.8536079525947571, -2.9033970832824707, 6.504937648773193, 17.613494873046875, 0.64699786901474, 0.23930874466896057, 3.3140883445739746, 5.532526969909668, 6.971859455108643, -5.898531436920166, 0.4559820890426636, 4.005783557891846, 3.454050064086914, 1.7579810619354248, -0.022450150921940804, -2.7568061351776123, 10.428108215332031, 3.246994972229004, -1.8419989347457886, 0.4839785099029541, -1.872499942779541, -3.2648043632507324, -0.03526967018842697, -10.698554992675781, -3.1331257820129395, -1.8816192150115967, -6.255736827850342, 1.6316605806350708, -11.972794532775879, -1.7872214317321777, 0.11770229786634445, -4.347874164581299, -0.7713831663131714, 2.2124969959259033, -2.6058359146118164, -8.70969295501709, -0.14515629410743713, -7.471645355224609, -3.600957155227661, -5.502898216247559, -5.258497714996338, -1.0712965726852417, -1.5598149299621582, -6.649387836456299, 8.487722396850586, -5.885616779327393, 4.459370136260986, -2.1313936710357666, -2.935807943344116, -6.622952938079834, -5.954438209533691, -2.8606314659118652, -0.8067338466644287, -8.63303279876709, -8.896941184997559, 8.826458930969238, 1.0825471878051758, 4.065207481384277, -7.79862117767334, 12.822866439819336, 5.32867431640625, -5.34097957611084, -4.1355671882629395, 2.90248703956604, -7.174384593963623, -1.3683228492736816, -3.43160080909729, -8.348855018615723, -3.9092211723327637, 2.291048765182495, -2.165220260620117, -3.1775949001312256, 2.1330196857452393, -1.1082369089126587, -8.796402931213379, -4.083916187286377, -4.522512912750244, -2.702939033508301, -9.020867347717285, -12.126185417175293, -6.384997367858887, -4.280674457550049, 1.0286301374435425, 1.3377994298934937, -13.58321762084961, 2.291396379470825, -5.807153701782227, -0.4868025779724121, -4.234897136688232, 2.8906962871551514, 5.907098770141602, 4.87150239944458, 4.337770462036133, 1.2028088569641113, -1.8561367988586426, -8.512022018432617, -0.8450262546539307, -11.808391571044922, 6.540306091308594, -4.035245418548584, -1.4433025121688843, 5.309558391571045, 2.909183979034424, 2.6435329914093018, -3.9250240325927734, -7.126317501068115, 6.045494079589844, -1.7300986051559448], \"response\": [{\"UUID\": \"cfcad20f-a015-41e5-aa7e-a1b0f3efc642\", \"score\": 0.5771155852749257, \"content\": \"Perception of temperature is determined by the firing rates of cold and warm receptors. When touching an object between 5-30\\u00b0C, cold receptor firing increases and warm receptor firing decreases, indicating cold. When touching an object between 30-45\\u00b0C, warm receptor firing increases and cold receptor firing decreases, indicating warmth. If an object is hot enough to cause pain, nociceptors are activated, signaling pain and further indicating potential injury.\", \"row_data\": {\"id\": \"cfcad20f-a015-41e5-aa7e-a1b0f3efc642\", \"doc_name\": \"docs/feel_temperature.pdf\", \"title\": \"'Neural Responses: Feeling Hot and Cold Temperatures'\", \"content\": \"Perception of temperature is determined by the firing rates of cold and warm receptors. When touching an object between 5-30\\u00b0C, cold receptor firing increases and warm receptor firing decreases, indicating cold. When touching an object between 30-45\\u00b0C, warm receptor firing increases and cold receptor firing decreases, indicating warmth. If an object is hot enough to cause pain, nociceptors are activated, signaling pain and further indicating potential injury.\"}}, {\"UUID\": \"e14be89d-e330-4dcf-b401-b251dbcd3c26\", \"score\": 0.5771155852749257, \"content\": \"To understand FNE thermoreceptors, you need to know about neurotransmission. The brain's signal strength relies on a neuron's firing frequency. We interact with objects of varying temperatures, but thermoreceptors typically detect room temperature, maintaining normal firing rates. Cold or hot stimuli alter the firing rate of corresponding thermoreceptors.\", \"row_data\": {\"id\": \"e14be89d-e330-4dcf-b401-b251dbcd3c26\", \"doc_name\": \"docs/feel_temperature.pdf\", \"title\": \"'Frequency of Firing in Cold Thermoreceptors'\", \"content\": \"To understand FNE thermoreceptors, you need to know about neurotransmission. The brain's signal strength relies on a neuron's firing frequency. We interact with objects of varying temperatures, but thermoreceptors typically detect room temperature, maintaining normal firing rates. Cold or hot stimuli alter the firing rate of corresponding thermoreceptors.\"}}, {\"UUID\": \"cfcad20f-a015-41e5-aa7e-a1b0f3efc642\", \"score\": 0.5771155852749257, \"content\": \"Perception of temperature is determined by the firing rates of cold and warm receptors. When touching an object between 5-30\\u00b0C, cold receptor firing increases and warm receptor firing decreases, indicating cold. When touching an object between 30-45\\u00b0C, warm receptor firing increases and cold receptor firing decreases, indicating warmth. If an object is hot enough to cause pain, nociceptors are activated, signaling pain and further indicating potential injury.\", \"row_data\": {\"id\": \"cfcad20f-a015-41e5-aa7e-a1b0f3efc642\", \"doc_name\": \"docs/feel_temperature.pdf\", \"title\": \"'Neural Responses: Feeling Hot and Cold Temperatures'\", \"content\": \"Perception of temperature is determined by the firing rates of cold and warm receptors. When touching an object between 5-30\\u00b0C, cold receptor firing increases and warm receptor firing decreases, indicating cold. When touching an object between 30-45\\u00b0C, warm receptor firing increases and cold receptor firing decreases, indicating warmth. If an object is hot enough to cause pain, nociceptors are activated, signaling pain and further indicating potential injury.\"}}, {\"UUID\": \"e14be89d-e330-4dcf-b401-b251dbcd3c26\", \"score\": 0.5771155852749257, \"content\": \"To understand FNE thermoreceptors, you need to know about neurotransmission. The brain's signal strength relies on a neuron's firing frequency. We interact with objects of varying temperatures, but thermoreceptors typically detect room temperature, maintaining normal firing rates. Cold or hot stimuli alter the firing rate of corresponding thermoreceptors.\", \"row_data\": {\"id\": \"e14be89d-e330-4dcf-b401-b251dbcd3c26\", \"doc_name\": \"docs/feel_temperature.pdf\", \"title\": \"'Frequency of Firing in Cold Thermoreceptors'\", \"content\": \"To understand FNE thermoreceptors, you need to know about neurotransmission. The brain's signal strength relies on a neuron's firing frequency. We interact with objects of varying temperatures, but thermoreceptors typically detect room temperature, maintaining normal firing rates. Cold or hot stimuli alter the firing rate of corresponding thermoreceptors.\"}}]}"

```



# Next
- fix redis that doesn't save anything as value for key. find in the code where is the issue - OK for the moment works fine - OK
- adapt all functions to the new graph as those are exported to their to make the graph run, but not yet fixed. - OK
- tranfer all `test.py` function/graph/etc/... to get our initial graph to `app.py` - OK
- create all variables that can be put in .env file and create a .env file special for app vars so that we update only that .env file - OK


# returned values fromt he answer_user_report function, to see what are the values of the previous messages
- Message[-3]:  
```bash
content='What are the different types of thermoceptors?' id='e927aa47-337b-40a6-8494-450b9cf80072' 
```
- Message[-2]:
```bash
content='' additional_kwargs={'tool_calls': [{'id': 'call_98bp', 'function': {'arguments': '{"query":"different types of thermoceptors"}', 'name': 'search'}, 'type': 'function'}]} response_metadata={'token_usage': {'completion_tokens': 94, 'prompt_tokens': 12332, 'total_tokens': 12426, 'completion_time': 0.152341942, 'prompt_time': 0.610758584, 'queue_time': 0.0022357619999999745, 'total_time': 0.763100526}, 'model_name': 'mixtral-8x7b-32768', 'system_fingerprint': 'fp_c5f20b5bb1', 'finish_reason': 'tool_calls', 'logprobs': None} id='run-769b9abf-e2f5-411b-b3cb-f1b78966b786-0' tool_calls=[{'name': 'search', 'args': {'query': 'different types of thermoceptors'}, 'id': 'call_98bp', 'type': 'tool_call'}] usage_metadata={'input_tokens': 12332, 'output_tokens': 94, 'total_tokens': 12426}
```
- Message[-1]:
```bash
content='{"messages": ["Different types of stimuli are sensed by different types of receptors. ... Thermoreceptors are free nerve endings which respond to changes in temperature and are primarily located in skin and mucous membranes. Thermoreceptors responding to innocuous (nonharmful) warm signals are found on dendritic branches of unmyelinated fibers and respond to ... The somatosensory system also includes receptors and neurons that convey information about body position and movement to the brain. These proprioceptors are housed in muscle, bone, and tendons and respond to stretch and contraction, tension and release. Examples of different types of receptors located under our skin. Knowing what type of stimulus occurred will depend on which type of receptor was stimulated and what region of the ________ receives the impulses coming from sensory receptors. a. brain stem. b. cerebellum. c. cerebral cortex. c. cerebral cortex. Visceral pain may feel as if it comes from some part of the body other than the part being stimulated. The different types of functional receptor cell types are mechanoreceptors, photoreceptors, chemoreceptors (osmoreceptor), thermoreceptors, electroreceptors (in certain mammals and fish), and nociceptors. Physical stimuli, such as pressure and vibration, as well as the sensation of sound and body position (balance), are interpreted through a ... Thermoreceptors are free nerve endings (FNE), which extend until the mid-epidermis, or the outermost layer of our skin. ... If you want to learn more about different types of receptors that signal ..."]}' name='search' id='94b381cf-0ae4-4857-817f-e9d1b831a586' tool_call_id='call_98bp'
Tool called!: (messages[-2].additional_kwargs['tool_calls'])    [{'id': 'call_98bp', 'function': {'arguments': '{"query":"different types of thermoceptors"}', 'name': 'search'}, 'type': 'function'}]
```


# scenario
Scenario of user requiring a full SEO report on a website using its main page

# Next
- keep in mind the pdf parser that might need to be refactored to chunk using same process as the webpage parser one.
- reset the cache to zero as well so that we have the option to delete everything for some future task that doesn't need the data to persist forever in the DB.
- add a node to manage the retrieved answer to use tools to search about it or to fetched the answer from it and build a response. - OK
- a logic with a new funciton that going to use the same funciton as the one query_matching that will check if the document exist in our db if yes if does a query matching and jumps directly to he retrieval graph , if not it is going to use the normal route process.
- add langfuse
- test the last graph alone to see how report look like - OK
- try to test workflow from retrieval graph to report graph only by isolating those and commenting the other graphs - OK





query test:  What is responsible for detecting temperature changes in our skin? check this documents docs/feel_temperature.pdf



# Next
- keep in mind the pdf parser that might need to be refactored to chunk using same process as the webpage parser one.
- reset the cache to zero as well so that we have the option to delete everything for some future task that doesn't need the data to persist forever in the DB.
- we can check if we are going to add docker image of redis specialized in semantic search through cache. 
- we need to implement deletion of data and reset all also.
- a logic with a new funciton that going to use the same funciton as the one query_matching that will check if the document exist in our db if yes if does a query matching and jumps directly to he retrieval graph , if not it is going to use the normal route process.
- add langfuse


# Plan for Agent Graph that executes code in docker sandbox `BRAINSTORMING`
- two different agents in parallele that will write the script in markdown and we will use structure output the first way (pydantic) in order to have for sure the code and nothing else. Both agents will use different model so that we have the choice between two different codes.
- on agent that will analyze both codes produces and will suggest best code without errors (Here we will infer that they is probably a better easy way to perform the task and that we have two codes to learn from and not do same mistakes, thena s to create a script) here also we need structured output
- one agent that will make the requirements.txt for the code that will run in the docker sandbox for the code created
- one agent that check the code execution log file to see if the code have been successfully executed (structured output here too)
- one agent that need to check the desired code execution outcome and the log file or the folder where the output should be or .... and will tell if it has been successful or not (create custom logic depending on the scenario)

### agents workflow for each stages we will have to use conditional edges
set_key(".var.env", "AGENT_CODE_CREATED_FILE", "agent_code.py")
The graph will be put in a while True loop and stop when it is done and working fine with a retry max number that will be put in the env config file "CODE_EXECUTOR_AGENT_RETRY":
code.py > requirements.txt > check code execution logs > check output file present or code execution intent fulfilled or not vs the initial request
./docker_agent/agent_code.py
    > ./docker_agent/sandbox_requirements.txt and ./docker_agent/.sandbox.env
      > sandbow_docker_execution_logs.md (Will return execution as a prompt that notify agent to provide a fix and say it is an error or OK)
        > custom logic depending on the scenario to check if outcome is there or not
        > use vision agent to see if output is what we want if there is any image involved
          > code stops if "CODE_EXECUTION_AGENT_RETRY" reaches "0" or if the outcome is what we want and we will set env var "CODE_EXECUTION_JOB_SUCCESSFUL" to True
  
        
###  We use APIs, some APIs that can be used

- **CoinGecko API**
Description: Offers comprehensive data on cryptocurrency markets. You can use it to test handling and correcting data related to financial information and numeric precision.
API Endpoint: https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd

- **Random Image API**
Description: Fetches random images. This API can help test your agent's handling of binary data and media formats, along with error handling for image links.
API Endpoint: https://yesno.wtf/api

- **Agify.io**
Description: This API predicts the age based on a given name. It's simple and provides an excellent opportunity to test handling and correcting data format issues in API requests.
API Endpoint: https://api.agify.io?name=[name]

- **Cat Facts API.**
Description: Provides random cat facts. It’s simple and fun for testing.
Endpoint: https://catfact.ninja/fact

- **Yes/No API**
Description: Fetches a random "yes" or "no" response, great for testing binary decision-making.
Endpoint: https://yesno.wtf/api

- **Official Joke API**
Description: Provides random jokes. It’s simple, fun, and useful for handling textual data.
Endpoint: https://official-joke-api.appspot.com/random_joke

- **Dog CEO's Dog API**
Description: Returns random images of dogs, useful for testing image retrieval.
Endpoint: https://dog.ceo/api/breeds/image/random


 i will have an agent that will use this in order to choose which api will be called and another internet agent that will then look for the documentation and come back with how to do that query requested by the user, then another agent that will use docker to create on the fly a dockerfile and execute the code. another agent that will read the logs of the executed code and tell if the task has been completed correctly or not. and if not correctly executed it will go back the agent creating the code with the error message of the fix suggested for the code generator agent to create again the code to fulfil user desired api call using python. so a  self-correcting code agents



[User Request]
     |
[API Selection Agent]
     |
[Go Internet find API Documentation] (maybe can use our graph to have a report in how to do the api call)
          |
[Generate Code with model 1]       [Generates code with model 2] <----------------------------------------------------------|
               |                                 |                                                                          |
           ----------------------------------------------                                                                   |
                    |                                                                                                       |
[Compare both codes and learn from those in order to generate the wanted script]                                            |
     |                                                                                                                      |
[Execute code]                                                                                                              |
     |                                                                                                                      |
[Read log file]                                                                                                             |
            |                                                                                                               |
    -------------------------------------------------------                                                                 |
            |                            |                                                                                  |
[Code is ok, we keep going]    [Error in the execution]                                                                     |
     |                           |                                                                                          |
     |                           |______________---> [get env vars needed (Initial request, api choosen), log returned error, add prompt asking to fix the code]
     |
[Success] ---> [Output Results]


#### Parallele node execution in langgraph

- code example:
```python
# import needed
from langchain_core.runnables import RunnableParallel

# Define models or prompts
model = ChatOpenAI()
joke_chain = ChatPromptTemplate.from_template("Tell me a joke about {topic}") | model
poem_chain = ChatPromptTemplate.from_template("Write a 2-line poem about {topic}") | model

# Run both chains in parallel
parallel_tasks = RunnableParallel(joke=joke_chain, poem=poem_chain)

# Execute in parallel
results = parallel_tasks.invoke({"topic": "AI"})
print(results)

Outputs DICT[str, str]:
{
    "joke": "Why don't scientists trust atoms? Because they make up everything!",
    "poem": "AI so bright, codes through the night."
}

```

- OR
```python
# conditional edge returning a lsit of nodes
def conditional_edge_decision(state: MessagesState):
    # Based on some condition, return multiple nodes to be executed
    if some_condition:
        return ["node_a", "node_b", "node_c"]
    else:
        return ["fallback_node"]
 
# can also create an intermediary node that will get the ouput of the condition edge returned last message and stream that to all other nodes that will run in parallele
# but we will just use a list of nodes in the conditional edge function returned value when success is there
workflow.add_edge("intermediary_node", "llama_3_8b_script_creator")
workflow.add_edge("intermediary_node", "llama_3_70b_script_creator")
workflow.add_edge("intermediary_node", "gemma_3_7b_script_creator")

# we could use a dispacher nodes as well
def dispatcher_node(state: MessagesState):
    nodes_to_run = ["node_a", "node_b", "node_c"]
    results = []

    # Dispatch all tasks in parallel
    for node in nodes_to_run:
        result = run_node(node)
        results.append(result)
    
    # Return results to the next node
    return results

# or an async node, but don't know how it would work, just an example
async def execute_parallel_nodes():
    task_1 = asyncio.create_task(run_node("node_a"))
    task_2 = asyncio.create_task(run_node("node_b"))
    task_3 = asyncio.create_task(run_node("node_c"))

    await asyncio.gather(task_1, task_2, task_3)
```

# Next
- implement `retry` for code_execution graph from a env var - OK

# next
- finish the logic and funcitons coding and utils functions for the code executor graph - IN PROCESS
- keep in mind the pdf parser that might need to be refactored to chunk using same process as the webpage parser one. - NOT OBLIGED
- reset the cache to zero as well so that we have the option to delete everything for some future task that doesn't need the data to persist forever in the DB. - TO BE DONE
- we need to implement deletion of data and reset all also. Like database data so it is stateless app just do job and erase everything after a certain amount of time, so maybe keep track of documents names in a list env var updated in a dict with time start and when the TTL is reached it will be erased automatically (at app run or cronjob?..) - TO BE DONE
- we can check if we are going to add docker image of redis specialized in semantic search through cache.  - NOT OBLIGED BUT KEEP IN MIND FOR FUTURE PROJECTS INTERESTING

- a logic with a new funciton that going to use the same funciton as the one query_matching that will check if the document exist in our db if yes if does a query matching and jumps directly to he retrieval graph , if not it is going to use the normal route process.  - NOT DONE YET BUT TO BE IMPLEMENTED
- add langfuse - TO BE DONE
- finish the logic of the code execution graph and beware of retries, check again to make sure it is all good, draw a diagram so that you don't miss anything in the retry flow.  - TO BE DONE NOW - OK and diagram will be done at the end of first iteration as not that confusing
- brush up logic of which code will be chosen for user at the end if we have several codes being successfully executed maybe have an agent with structured output chosing among those with reason why and render that one to user.. - TO BE DONE NOW

# next
- check the `next` just before this one
- add langfuse
- make the nodes and conditional edges after code execution conditional edge
- make all structured outputs needed
- put all hard coded prompts to the prompt file and import those



# next
- need to review the flow of the user input after it's conditional edge if the there is no documentaiton needed nor code creation so that it returns to the business logic app and create an answer report. but before that we need to check if there is no links or pdf documents improving the user_input function structured outptut to return the yes or no for pdf identified field and url identified field, in thta case we will just forward to normal other graph flows. if no document we need to create a way to make the report using the report graph with env vars filled approprietely.

'''
have created more prompts templates, structured outputs for all llm calls, still need to review which llm for which structured outptu as it may need more context length capacity sometimes, have also created conditional edge for user inital request to be analyzed but need to be improved as it exist the graph if no code nor document generation is needed and will start other graphs depending on if there is a link or pdf file in the query....."
[main aedd52e] have created more prompts templates, structured outputs for all llm calls, still need to review which llm for which structured outptu as it may need more context length capacity sometimes, have also created conditional edge for user inital request to be analyzed but need to be improved as it exist the graph if no code nor document generation is needed and will start other graphs depending on if there is a link or pdf file in the query.....

'''

- graph work fine until after the parallele code generation and then comes up with an error, so fix it and continue debugging and improving until graph finisehd and test all cases with url and more... check that the documentation written by agent is used by agents when doing parallele coding:
```bash
  File "/home/creditizens/langgraph/graphs/code_execution_graph.py", line 391, in code_evaluator_and_final_script_writer
    dict_llm_codes[llm_name.strip()] == llm_code.strip()
KeyError: 'gemma_3_7b'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/creditizens/langgraph/app.py", line 77, in <module>
    raise Exception(f"An error occured while running 'code_execution_flow': {e}")
Exception: An error occured while running 'code_execution_flow': 'gemma_3_7b'
```


# rendering the graph image
Here are some methods that can be used with Xray for visualizing graphs:

- **`.to_mermaid()`:**
Converts the graph into Mermaid code format. Mermaid is a popular diagram scripting language used to visualize workflows, graphs, and other structured data as text.

- **`.draw_mermaid_code()`:**
Renders the graph as Mermaid code directly into a readable diagram format in your IDE or notebook. It allows for easy copy-pasting of the code to use with Mermaid editors.

- **`.draw_mermaid_png()`:**
This method generates a PNG image from the Mermaid diagram code, providing a visual representation of the graph in image format.

- **`.draw_png()`:**
Directly generates a PNG image of the graph using its internal visualization engine, without converting it to Mermaid first.





input_variables=['query'] partial_variables={'format_instructions': 'The output should be formatted as a JSON instance that conforms to the JSON schema below.\n\nAs an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}\nthe object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\nHere is the output schema:\n```\n{"description": "Evaluate quality of Python script code created to make API call.", "properties": {"validity": {"title": "Validity", "description": "Say \'YES\' if Python script code is evaluated as well written, otherwise \'NO\'.", "default": "", "type": "string"}, "reason": {"title": "Reason", "description": "Tell reason why the code is evaluated as valid or not.", "default": "", "type": "string"}}}\n```'} template='You are an expert in Python script code review. You decide if the code is valid or not by checking if it has th eright imports, does the job required, have good indentation and check anything else that is required to check to make sure it is a valid working and executable code as it is. Answer using markdown but return the output strictly as a valid JSON object.\n{format_instructions}\n{query}\n'

# error to be handled
Structured Output Response ERROR:  Error code: 429 - {'error': {'message': 'Rate limit reached for model `mixtral-8x7b-32768` in organization `org_01hqrpzhwxfmd8pvzc25nr32hg` on : Limit 500000, Used 500153, Requested 757. Please try again in 2m37.3056s. Visit https://console.groq.com/docs/rate-limits for more information.', 'type': '', 'code': 'rate_limit_exceeded'}}

 git commit -m "we have improved some prompts, needs more prompt refinement, fixed some code issues and had rate limit of groq, will stop here for the moment or maybe use lmstudio next time. It is not consistent, we have it sometimes stopping in the middle, we have passed the parallel code execution and selection of valid codes and now are at if there is more than one valid code to chose which one will be executed. we need to work on the formatting of that code in the file as we need to get rid of python mardown tags, maybe we will ask for the code but not in markdown to see if it write fine in the file asking for python synthaxed file code."

{
  'valid': 
    [
      {'gemma_3_7b': "import requests\n\ndef get_age(name):\n    url = 'https://api.agify.io?name={}'.format(name)\n    response = requests.get(url)\n    return response.json()['age']\n\nage = get_age('Junko')\nprint('Age:', age)"},
      {'llama_3_70b': "import requests\n\ndef get_age(name):\n    url = f'https://api.agify.io?name={name}'\n    response = requests.get(url)\n    if response.status_code == 200:\n        data = response.json()\n        return data['age']\n    else:\n        return None\n\nname = 'Junko'\nprint(f'The age of {name} is {get_age(name)}')"},
      {'llama_3_8b': "import requests\n\ndef get_age(name):\n    url = f'https://api.agify.io?name={name}'\n    response = requests.get(url)\n    data = response.json()\n    return data['age']\n\nname = 'Junko'\nprint(get_age(name))"}
    ], 
  'invalid': 
    []
}

DICT[valid:LIST[DICT[llm, code]], invalid:LIST[DICT[llm, code]]]


# Analyzing the `LangChain` prompt
input_variables=['query'] 
partial_variables={
  'format_instructions': 'The output should be formatted as a JSON instance that conforms to the JSON schema below.\n\nAs an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}\nthe object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\n
  Here is the output schema:\n```\n{"description": "      Analyze Python script to determine what should be in a corresponding requirements.txt file.\\n      Answers should strictly be as a valid JSON object. Do NOT include any explanations, markdown, or non-JSON text. The script should be returned as a JSON object with the key \'script\', and the value should be a string containing the Python script. The Python code should be in a single block and fully executable. Replace any markdown delimiters (such as ``` or ```python) with an empty string (\'\').\\n      Example Bad JSON format: {\\"bad_example\\": \\"This is a bad example with issues like unescaped quotes in \'keys\' and \'values\', improper use of ```markdown``` delimiters, and mixed single/double quotes.\\"}.\\n      Example Good JSON format:\\n      {\'script\': \'import requests\\n\\n...\'}\\n       OR  \\n      {\\"good_example\\": \\"This is a good example where quotes are properly escaped, like this: \\"escaped quotes\\", and no markdown code block delimiters are used.\\"}\\nEnsure that the output is a valid JSON object and does not contain any additional text or explanation.\\"\\n    ", 

## result of analysis
- We have already used that kind of prompt in the past, here it is a bit over-complicated. LLM can be just prompted and needs example of the required output. We can use our technique asking to put answer in ```markdown ```. And add our prompts by injection. (so inject schema and query(human) to system prompt)
response_schema={
  "Requirements" : "The content of the requirements.txt file, with right versions and format of a requirements.txt file content. Do not use any markdown code block delimiters (i.e., ``` and ```python) replace those with \'\'. This value MUST be JSON serializable and deserializable, therefore, make sure it is well formatted. Ignore docker, don\'t put it in the requirement as it is already installed.",
  "Needed": "Answer \'YES\' or \'NO\' depending on if the code requires a requirements.txt file."
} 

prompt["system"]["template"]="You are an expert in Python script code review and requirements.txt file. You will be presented different Python script and will decide if it needs a requirements.txt file. If it need one you will provide the content of the corresponding requirements.txt. Strictly answer following the given schema.\nhere is the schema that you have to follow and make sure it is a proper JSON format and put it between ```markdown ``` tags to ease parsing of response: <YOUR RESPONSE SCHEMA>{response_schema}</YOUR RESPONSE SCHEMA>\nHere is user query: {query}\n"

{
  'requirements': 'requests\n', 
  'script': "#\\!/usr/bin/env python3\n\n# code from: gemma\\_3\\_7b LLM Agent\n# User initial request: None\n\n'''This code have been generated by an LLM Agent''' \n\nimport requests\n\ndef get\\_age(name):\n    url = 'https://api.agify.io?name={}'.format(name)\n    response = requests.get(url)\n    age = response.json()['age']\n    return '{} is too old now!'.format(age)\n\nprint(get\\_age('Junko'))\n", 
  'needed': 'YES'
}

documentation_steps_evaluator_and_doc_judge

### **final step in coding graph**
Step 16: {
    "inter_graph_node": {
        "messages": [
            {
                "role": "system",
                "content": "[{\"Graph Code Execution Done\": \"Find here the last 3 messages of code execution graph\", \"message -3\": \"{\\\"success\\\": {\\\"requirements\\\": \\\"./docker_agent/agents_scripts/gemma_3_7b_requirements.txt\\\", \\\"llm_name\\\": \\\"gemma_3_7b\\\"}}\", \"message -2\": \"{\\\"stdout\\\": [{\\\"script_file_path\\\": \\\"./docker_agent/agents_scripts/agent_code_execute_in_docker_gemma_3_7b.py\\\", \\\"llm_script_origin\\\": \\\"gemma_3_7b\\\", \\\"output\\\": \\\"success:The age of your friend Junko is: 56\\\\n\\\"}], \\\"stderr\\\": [], \\\"exception\\\": []}\", \"message -1\": \"{\\\"success_code_execution\\\": [{\\\"script_file_path\\\": \\\"./docker_agent/agents_scripts/agent_code_execute_in_docker_gemma_3_7b.py\\\", \\\"llm_script_origin\\\": \\\"gemma_3_7b\\\", \\\"output\\\": \\\"success:The age of your friend Junko is: 56\\\\n\\\"}]}\"}]"

I am not very funny in general and need random jokes being produced using an API that permits it, can you help me make an API script that only create random jokes? I am not good at coding so make sure that the script syntax is correct


I am not very funny guy and want to use some jokes during next very serious meeting, can you make a Python script that calls an API to get jokes. Make sure it is well formatted, indented and using only Python standard libraries to make it easy for me to execute












1.3s (10/10) FINISHED                                                                                                                                    docker:default
 => [internal] load build definition from gemma_3_7b_dockerfile                                                                                                                 0.0s
 => => transferring dockerfile: 466B                                                                                                                                            0.0s
 => [internal] load metadata for docker.io/library/python:3.9-slim                                                                                                              1.1s
 => [internal] load .dockerignore                                                                                                                                               0.0s
 => => transferring context: 2B                                                                                                                                                 0.0s
 => [1/5] FROM docker.io/library/python:3.9-slim@sha256:49f94609e5a997dc16086a66ac9664591854031d48e375945a9dbf4d1d53abbc                                                        0.0s
 => [internal] load build context                                                                                                                                               0.0s
 => => transferring context: 780B                                                                                                                                               0.0s
 => CACHED [2/5] COPY ./docker_agent/agents_scripts/gemma_3_7b_requirements.txt .                                                                                               0.0s
 => CACHED [3/5] RUN pip install --no-cache-dir -r gemma_3_7b_requirements.txt                                                                                                  0.0s
 => [4/5] COPY ./docker_agent/agents_scripts/agent_code_execute_in_docker_gemma_3_7b.py /app/agent_code_execute_in_docker_gemma_3_7b.py                                         0.0s
 => [5/5] WORKDIR /app                                                                                                                                                          0.0s
 => exporting to image                                                                                                                                                          0.1s
 => => exporting layers                                                                                                                                                         0.0s
 => => writing image sha256:b7b6088f5e464422d89e5fb485e9a87733549c50eee0d3da48797d39fe3d9a5c                                                                                    0.0s
 => => naming to docker.io/library/sandbox-python                                                                                                                               0.0s
Untagged: sandbox-python:latest



I am looking at pictures of random dog breeds. I want to create a script using Python that calls an API and get an returned dog image. Make sure that the script is well written without any errors, missing parenthesis, comma, string literal not well closes, and use only python standard librairies






























